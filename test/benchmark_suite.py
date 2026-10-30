#!/usr/bin/env python3
"""SCRUB-VOICE Benchmark Suite (Raspberry Pi 4 / ARM64 edge evaluation).

Master orchestrator for research-grade edge benchmarking:
  DSP   beamforming (DOA error, SI-SDR gain, RTF) and denoising (SI-SDR, STOI, RTF)
  ASR   WER with bootstrap CI, RTF, post-speech finalize latency
  NLU   intent accuracy, slot match, OOD false-accept rate (Wilson CIs), latency, thread scaling
  IPC   ZeroMQ transport vs raw UNIX sockets, dispatcher round trip, priority preemption,
        HWM burst behaviour, malformed-payload resilience
  E2E   streaming pipeline, post-speech latency decomposed per stage, SNR sweeps, ablations,
        head-to-head baselines (same hardware, same data), paired significance tests
  LIVE  real-time capture from the mic array for N seconds, with per-chunk deadline tracking
  SOAK  long-run stress test for RSS memory leak drift and latency drift

Modes
-----
  --mode wav   (default) offline: multichannel WAV, a labelled manifest, or --synthetic N
  --mode live  Raspberry Pi + array mic: records --live-seconds (default 10) while processing
               the stream in real time, saves the capture, then runs the offline stages on it

Examples
--------
  # Harness smoke test, no data or models needed:
  python test/benchmark_suite.py --synthetic 6 --stages dsp --scope isolated

  # Full run on a labelled dataset:
  python test/benchmark_suite.py --manifest data/bench/manifest.jsonl \\
      --noise-wav data/noise/or_noise.wav --scope both --baselines --seeds 3

  # Fast CI/CD smoke test:
  python test/benchmarks/bench_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Common package imports
from test.benchmarks.common.acoustics import (
    Item,
    NoiseSource,
    align_to,
    ang_err,
    build_items,
    build_stream,
    chunk_iter,
    circular_array,
    load_audio,
    make_sample,
    plane_wave,
    quality,
    save_audio,
    si_sdr,
    snr_label,
    synth_or_noise,
    synth_speechlike,
    unit,
)
from test.benchmarks.common.registry import (
    BUDGETS,
    DEFAULT_NLU_EVAL,
    DeepFilterNetDenoiser,
    NoiseReduceDenoiser,
    Passthrough,
    PluginBeamformer,
    PluginDenoiser,
    RefBeamformer,
    RefChannel,
    Registry,
    RepoNLU,
    SpectralGate,
    TfidfNLU,
    VoskASR,
    WhisperASR,
    _try_import,
    load_plugin,
)
from test.benchmarks.common.reporting import (
    _jd,
    make_figures,
    md_table,
    print_budget_table,
    tex_table,
    write_csvs,
    write_tables,
)
from test.benchmarks.common.rig import DispatcherRig, endpoint
from test.benchmarks.common.stats import (
    corpus_wer,
    holm,
    norm_text,
    paired_test,
    slots_match,
    summarize,
    wilson,
    word_errors,
)
from test.benchmarks.common.telemetry import ResourceSampler, get_rss_mb, hw_info

# Modular stage runners
from test.benchmarks.bench_asr import bench_asr
from test.benchmarks.bench_dsp import bench_dsp
from test.benchmarks.bench_e2e import Variant, bench_e2e, make_variants, run_utterance
from test.benchmarks.bench_ipc import bench_ipc
from test.benchmarks.bench_live import bench_live
from test.benchmarks.bench_nlu import bench_nlu
from test.benchmarks.bench_soak import bench_soak


def parse_args():
    p = argparse.ArgumentParser(
        description="SCRUB-VOICE research-grade edge benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["wav", "live"], default="wav")
    p.add_argument("--scope", choices=["isolated", "e2e", "both"], default="both")
    p.add_argument("--stages", default="dsp,asr,nlu,ipc", help="isolated stages to run")
    p.add_argument("--wav", help="single (multichannel) WAV: latency/RTF only, no labels")
    p.add_argument("--manifest", help="JSONL labelled dataset")
    p.add_argument("--synthetic", type=int, default=0, help="N synthetic speech-like clips")
    p.add_argument("--noise-wav", help="real OR noise recording used for SNR sweeps")
    p.add_argument("--snr-sweep", default="20,15,10,5,0")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--reps", type=int, default=100, help="timing repetitions for DSP chunks")
    p.add_argument("--reps-nlu", type=int, default=5)
    p.add_argument("--reps-ipc", type=int, default=500)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument(
        "--mic-channels",
        default="0,1,2,3,4,5",
        help="indices of perimeter capsules in capture",
    )
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument(
        "--diffuse-ratio",
        type=float,
        default=0.5,
        help="diffuse vs directional share of simulated noise power",
    )
    p.add_argument("--beamformers", default="ref_channel,ref_srp_das,repo_beamformer")
    p.add_argument("--denoisers", default="passthrough,spectral_gate,noisereduce,deepfilternet,repo_denoiser")
    p.add_argument("--asrs", default="vosk_grammar,vosk_open,whisper,repo_asr")
    p.add_argument("--baselines", action="store_true", help="add head-to-head baseline variants to E2E")
    p.add_argument("--plugin", action="append", default=[], help="NAME=module:attr")
    default_vosk = REPO_ROOT / "src" / "asr" / "models" / "vosk-model-small-en-us-0.15"
    p.add_argument(
        "--vosk-model",
        default=str(default_vosk) if default_vosk.exists() else None,
        help="path to Vosk model",
    )
    p.add_argument("--grammar", help="JSON list or text file of grammar phrases")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("--nlu-eval", help="JSONL: {text,intent,status[,slots]}")
    p.add_argument("--nlu-train", help="JSONL training set for TF-IDF baseline")
    p.add_argument("--nlu-thread-sweep", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpu-affinity", help="e.g. 0,1,2,3")
    p.add_argument("--vad-hangover-ms", type=float, default=0.0, help="add constant VAD hangover to E2E")
    p.add_argument("--no-dispatcher", action="store_true")
    p.add_argument("--stoi", action="store_true", help="compute STOI (needs pystoi)")
    p.add_argument("--soak-minutes", type=float, default=0.0)
    p.add_argument("--live-seconds", type=float, default=10.0)
    p.add_argument("--live-device")
    p.add_argument("--live-channels", type=int, default=8)
    p.add_argument("--out", default=str(REPO_ROOT / "results"))
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    for k in ("beamformers", "denoisers", "asrs", "stages"):
        setattr(a, k, [s for s in getattr(a, k).split(",") if s])
    if a.live_device is not None and a.live_device.isdigit():
        a.live_device = int(a.live_device)
    return a


def load_grammar(args, items) -> Optional[List[str]]:
    if args.grammar:
        t = Path(args.grammar).read_text()
        try:
            g = json.loads(t)
            return g if isinstance(g, list) else None
        except json.JSONDecodeError:
            return [ln.strip() for ln in t.splitlines() if ln.strip()]
    try:
        from src.asr.vocab import vosk_grammar_phrases

        phrases = vosk_grammar_phrases()
        if phrases:
            return phrases
    except Exception:
        pass
    words = sorted({w for it in items for w in norm_text(it.text).split()})
    if words:
        print("  [!] no --grammar given: derived a word-level grammar from test transcripts.")
        return words + ["[unk]"]
    return None


def check_requirements():
    missing = []
    for mod, pkg in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("psutil", "psutil"),
        ("zmq", "pyzmq"),
        ("orjson", "orjson"),
        ("jsonschema", "jsonschema"),
    ]:
        if _try_import(mod) is None:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing required dependencies: {', '.join(missing)}")
        print("Please install them: uv sync (or pip install -r requirements.txt)\n")
        sys.exit(1)


def main():
    check_requirements()
    args = parse_args()
    if args.cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(c) for c in args.cpu_affinity.split(",")})

    rng = np.random.default_rng(args.seed)
    snrs = [float(s) for s in args.snr_sweep.split(",") if s]
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    outdir = Path(args.out) / (f"bench_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
    outdir.mkdir(parents=True, exist_ok=True)

    print("*" * 72 + f"\n SCRUB-VOICE BENCHMARK | mode={args.mode} scope={args.scope}\n Host: {platform.platform()}\n" + "*" * 72)
    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "args": {k: v for k, v in vars(args).items()},
        "hardware": hw_info(),
        "stages": {},
    }
    noise = NoiseSource(args.noise_wav, args.sr)
    if not args.noise_wav:
        print("  [!] no --noise-wav: using SYNTHETIC OR-like noise (fine for harness checks, not for publication).")

    items = build_items(args, rng)

    nlu = None
    try:
        nlu = RepoNLU(args.threads)
    except Exception as e:
        print(f"  [!] repo NLU unavailable ({type(e).__name__}: {e}); NLU/IPC/E2E stages will be skipped.")

    grammar = load_grammar(args, items)
    reg = Registry(args, mic_xy, grammar)
    rig = None
    peak_rss = 0.0

    def run_stage(name: str, fn: Callable[[], Dict[str, Any]]):
        nonlocal peak_rss
        with ResourceSampler() as rs:
            try:
                res = fn()
            except Exception as e:
                import traceback

                traceback.print_exc()
                res = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
        res["resources"] = rs.summary()
        peak_rss = max(peak_rss, res["resources"].get("rss_peak_mb", 0.0))
        report["stages"][name] = res

    need_rig = (
        nlu is not None
        and not args.no_dispatcher
        and (args.scope in ("e2e", "both") or args.mode == "live" or args.soak_minutes)
    )
    if need_rig:
        try:
            rig = DispatcherRig("main")
        except Exception as e:
            print(f"  [!] dispatcher rig unavailable ({e}); IPC leg excluded from E2E.")

    if args.mode == "live":
        run_stage("live", lambda: bench_live(args, reg, nlu, rig, mic_xy, outdir))
        cap = outdir / "live_capture.wav"
        if cap.exists():
            args.wav = str(cap)
            items = items + build_items(
                argparse.Namespace(**{**vars(args), "manifest": None, "synthetic": 0, "max_items": 0}), rng
            )

    if not items and (args.scope != "isolated" or "dsp" in args.stages):
        print("  [!] no audio supplied (--manifest / --wav / --synthetic N): DSP uses synthetic stream, accuracy stages skipped.")

    if args.scope in ("isolated", "both"):
        if "dsp" in args.stages:
            run_stage("dsp", lambda: bench_dsp(args, reg, items, mic_xy, noise, snrs))
        if "asr" in args.stages:
            run_stage("asr", lambda: bench_asr(args, reg, items, mic_xy, noise, snrs))
        if "nlu" in args.stages:
            run_stage("nlu", lambda: bench_nlu(args, items))
        if "ipc" in args.stages:
            run_stage("ipc", lambda: bench_ipc(args, nlu))

    if args.scope in ("e2e", "both") and items:
        run_stage("e2e", lambda: bench_e2e(args, reg, nlu, rig, items, mic_xy, noise, snrs))

    if args.soak_minutes and items:
        run_stage("soak", lambda: bench_soak(args, reg, nlu, rig, items, mic_xy, noise))

    if rig:
        rig.close()

    report["component_load"] = reg.load_info
    report["skipped_components"] = reg.skipped
    report["resources"] = {"peak_rss_mb": peak_rss}

    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=1, default=_jd)

    write_csvs(report, outdir)
    write_tables(report, outdir)
    make_figures(report, outdir)
    print_budget_table(report)

    if reg.skipped:
        print("\n Skipped components:")
        for k, v in reg.skipped.items():
            print(f"   {k}: {v}")
    print(f"\n Results written to: {outdir}")


if __name__ == "__main__":
    main()
