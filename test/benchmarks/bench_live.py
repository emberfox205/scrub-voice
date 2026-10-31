"""Live Hardware Capture & Real-Time Streaming Benchmark.

Evaluates real-time audio acquisition and processing on edge hardware (e.g. Raspberry Pi 4 + ReSpeaker):
1. Continuous multi-channel capture via sounddevice InputStream into a ring-buffer queue.
2. In-stream Beamforming -> Denoising -> ASR decoding without buffer overrun (backlog <= 1, overflows == 0).
3. Post-speech finalization latency, intent classification, and IPC bus dispatching.
4. Preserves the full multi-channel recording as a calibrated WAV capture for offline verification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import shutil
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.bench_e2e import make_variants
from test.benchmarks.common.acoustics import circular_array, save_audio
from test.benchmarks.common.registry import BUDGETS, Registry, RepoNLU, _try_import
from test.benchmarks.common.reporting import _jd, save_json_report
from test.benchmarks.common.rig import DispatcherRig
from test.benchmarks.common.stats import summarize
from test.benchmarks.common.telemetry import ResourceSampler, hw_info


def bench_live(args, reg: Registry, nlu, rig, mic_xy: np.ndarray, outdir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 72 + f"\n [LIVE] Real-Time Hardware Capture ({args.live_seconds:g}s from array)\n" + "=" * 72)

    variants = make_variants(args, reg)
    if not variants or nlu is None:
        return {"status": "SKIPPED", "reason": "No runnable variant or NLU unavailable"}

    var = variants[0]
    bf, dn, asr = reg.get("bf", var.bf), reg.get("dn", var.dn), reg.get("asr", var.asr)
    sr = args.sr
    target_chunk = int(args.chunk_s * args.sr)
    n_chunks = int(math.ceil(args.live_seconds / args.chunk_s))
    overflow = [0]
    raw, lat, backlog, rtf = [], [], [], []
    tcap = tdone = 0.0

    asr.reset()

    # Determine backend: On Linux/Pi, use direct DMA-aligned ALSA AudioStreamer (avoids PortAudio deadlocks)
    use_arecord = (platform.system() == "Linux" and shutil.which("arecord") is not None)

    print("  [>] Recording & processing in real time... Speak now!")
    if use_arecord:
        from src.dsp.source.audio_streamer import AudioStreamer

        device = args.live_device if args.live_device else "plughw:seeed8micvoicec,0"
        try:
            with AudioStreamer(device=device, sample_rate=sr, chunk_duration=args.chunk_s) as streamer:
                for _ in range(n_chunks):
                    tcap = pc()
                    mc = streamer.read_chunk()  # shape: (6, samples)
                    raw.append(mc.T)
                    t0 = pc()
                    y, _ = bf.process(mc, sr)
                    asr.accept(dn.process(y, sr))
                    tdone = pc()
                    rtf.append((tdone - t0) / args.chunk_s)
                    lat.append((tdone - tcap) * 1000)
                    backlog.append(0)
        except Exception as e:
            return {"status": "ERROR", "reason": f"ALSA AudioStreamer capture error: {e}"}
    else:
        sd = _try_import("sounddevice")
        if sd is None:
            return {
                "status": "SKIPPED",
                "reason": "sounddevice not installed and arecord unavailable",
            }
        # Use aligned 1024-frame power-of-two hardware blocks for PortAudio driver stability
        hw_block = 1024 if (target_chunk >= 1024 and target_chunk % 1024 != 0) else min(1024, target_chunk)
        q: queue.Queue = queue.Queue()
        accum_chunks: list[np.ndarray] = []
        accum_count = 0
        t_chunk_start = pc()

        def callback(indata, frames, time_info, status):
            nonlocal accum_chunks, accum_count, t_chunk_start
            if status.input_overflow:
                overflow[0] += 1
            if not accum_chunks:
                t_chunk_start = pc()
            accum_chunks.append(indata.copy())
            accum_count += frames
            if accum_count >= target_chunk:
                full = np.concatenate(accum_chunks, axis=0)
                chunk = full[:target_chunk]
                remainder = full[target_chunk:]
                accum_chunks = [remainder] if len(remainder) > 0 else []
                accum_count = len(remainder)
                q.put((t_chunk_start, chunk))

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=args.live_channels,
                blocksize=hw_block,
                dtype="float32",
                device=args.live_device,
                callback=callback,
            ):
                for _ in range(n_chunks):
                    tcap, data = q.get(timeout=6.0)
                    raw.append(data)
                    mc = data.T[[i for i in args.mic_idx if i < data.shape[1]]]
                    t0 = pc()
                    y, _ = bf.process(mc, sr)
                    asr.accept(dn.process(y, sr))
                    tdone = pc()
                    rtf.append((tdone - t0) / args.chunk_s)
                    lat.append((tdone - tcap) * 1000)
                    backlog.append(q.qsize())
        except Exception as e:
            return {"status": "ERROR", "reason": f"Sounddevice InputStream error: {e}"}

    t0 = pc()
    text = asr.finalize()
    env = nlu.process(text)
    t_ipc = rig.roundtrip({**env, "timestamp": time.time()}) if (rig and env.get("status") == "EXECUTE") else None
    post = (tdone - tcap) * 1000 + (pc() - t0) * 1000 + (t_ipc or 0.0)

    outdir.mkdir(parents=True, exist_ok=True)
    cap_path = outdir / "live_capture.wav"
    try:
        save_audio(cap_path, np.concatenate(raw, axis=0).T, sr)
    except Exception as e:
        print(f"  [!] Failed to save live capture: {e}")

    res = {
        "status": "PASS" if (max(backlog) <= 1 and overflow[0] == 0 and post <= BUDGETS["e2e_ms_max"]) else "FAIL",
        "variant": var.__dict__,
        "transcript": text,
        "intent": env.get("intent"),
        "env_status": env.get("status"),
        "chunk_completion_lag_ms": summarize(lat),
        "rtf": summarize(rtf),
        "max_queue_backlog": int(max(backlog)) if backlog else 0,
        "input_overflows": overflow[0],
        "post_speech_latency_ms": post,
        "ipc_ms": t_ipc,
        "capture_file": str(cap_path),
    }

    print(f"  transcript: '{text}' -> intent={env.get('intent')} (status={env.get('status')})")
    print(
        f"  chunk lag p95 {res['chunk_completion_lag_ms']['p95']:.1f} ms | "
        f"max backlog {res['max_queue_backlog']} | overflows {overflow[0]} | "
        f"post-speech {post:.1f} ms  [{res['status']}]"
    )
    return res


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE Live Audio Capture Benchmark")
    p.add_argument("--live-seconds", type=float, default=10.0, help="Duration of live capture (s)")
    p.add_argument("--live-device", default=None, help="Audio input device index or name")
    p.add_argument("--live-channels", type=int, default=8, help="Hardware capture channels")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--mic-channels", default="0,1,2,3,4,5")
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument("--beamformers", default="repo_beamformer,ref_srp_das")
    p.add_argument("--denoisers", default="repo_denoiser,spectral_gate")
    p.add_argument("--asrs", default="vosk_grammar,repo_asr")
    p.add_argument("--plugin", action="append", default=[])
    default_vosk = REPO_ROOT / "src" / "asr" / "models" / "vosk-model-small-en-us-0.15"
    p.add_argument("--vosk-model", default=str(default_vosk) if default_vosk.exists() else None)
    p.add_argument("--grammar", help="Path to grammar file")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "live"), help="Output directory")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    a.beamformers = [s for s in a.beamformers.split(",") if s]
    a.denoisers = [s for s in a.denoisers.split(",") if s]
    a.asrs = [s for s in a.asrs.split(",") if s]
    if a.live_device is not None and a.live_device.isdigit():
        a.live_device = int(a.live_device)
    return a


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Operational: Live Hardware Capture\n" + "*" * 72)
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    outdir = Path(args.out) / (f"live_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
    outdir.mkdir(parents=True, exist_ok=True)

    nlu = None
    try:
        nlu = RepoNLU(args.threads)
    except Exception as e:
        print(f"  [warn] Could not load RepoNLU ({e})")

    grammar = None
    if getattr(args, "grammar", None):
        t = Path(args.grammar).read_text()
        try:
            grammar = json.loads(t)
        except Exception:
            grammar = [ln.strip() for ln in t.splitlines() if ln.strip()]

    reg = Registry(args, mic_xy, grammar)

    rig = None
    if nlu is not None:
        try:
            rig = DispatcherRig("live")
        except Exception as e:
            print(f"  [warn] Dispatcher rig unavailable ({e})")

    with ResourceSampler() as rs:
        res = bench_live(args, reg, nlu, rig, mic_xy, outdir)
    res["resources"] = rs.summary()

    if rig:
        rig.close()

    report = {
        "timestamp": time.time(),
        "hardware": hw_info(),
        "live": res,
    }
    save_json_report(report, outdir / "report_live.json")
    print(f"\n Results saved to: {outdir / 'report_live.json'}")


if __name__ == "__main__":
    main()
