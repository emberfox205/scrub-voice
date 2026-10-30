"""Isolated DSP Subsystem Benchmark (Beamforming & Denoising).

Evaluates:
1. Real-Time Factor (RTF) across 0.5s audio chunks (mean, p95, p99, cpu_over_wall, deadline misses).
2. Quality across SNR levels (clean, 20dB, 15dB, 10dB, 5dB, 0dB):
   - Absolute and Differential Scale-Invariant Signal-to-Distortion Ratio (SI-SDR, Delta SI-SDR).
   - Short-Time Objective Intelligibility (STOI, if pystoi is installed).
   - Direction of Arrival (DOA) error in degrees and % within 15 degrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.common.acoustics import (
    NoiseSource,
    ang_err,
    build_items,
    build_stream,
    chunk_iter,
    circular_array,
    make_sample,
    quality,
    snr_label,
)
from test.benchmarks.common.registry import BUDGETS, Registry
from test.benchmarks.common.reporting import _jd, md_table, tex_table, write_csvs
from test.benchmarks.common.stats import summarize
from test.benchmarks.common.telemetry import ResourceSampler, hw_info


def _apply_chunked(fn, x: np.ndarray, n: int):
    """Run fn(chunk) over chunks; fn may return an array or (array, azimuth)."""
    outs, azs = [], []
    for c in chunk_iter(x, n):
        r = fn(c)
        if isinstance(r, tuple):
            r, az = r
            if az is not None:
                azs.append(az)
        outs.append(r)
    return np.concatenate(outs)[: x.shape[-1]], azs


def bench_dsp(args, reg: Registry, items, mic_xy: np.ndarray, noise: NoiseSource, snrs: List[float]) -> Dict[str, Any]:
    print("\n" + "=" * 72 + "\n [DSP] Beamforming & Denoising Isolation Benchmark\n" + "=" * 72)
    rng = np.random.default_rng(args.seed)
    n = int(args.chunk_s * args.sr)
    stream = build_stream(items, args, mic_xy, noise, rng)
    chunks = [c for c in chunk_iter(stream, n)]
    bfs = {b: reg.get("bf", b) for b in args.beamformers}
    dns = {d: reg.get("dn", d) for d in args.denoisers}
    out: Dict[str, Any] = {"timing": {}, "quality": {}}

    def time_it(fn, label: str):
        for i in range(args.warmup):
            fn(chunks[i % len(chunks)])
        wall, cpu = [], []
        for i in range(args.reps):
            c = chunks[i % len(chunks)]
            t0, c0 = pc(), time.process_time()
            fn(c)
            wall.append(pc() - t0)
            cpu.append(time.process_time() - c0)
        rtf = np.asarray(wall) / args.chunk_s
        res = {
            "latency_ms": summarize(np.asarray(wall) * 1000),
            "rtf": summarize(rtf),
            "cpu_over_wall": float(np.sum(cpu) / np.sum(wall)) if np.sum(wall) > 0 else 0.0,
            "deadline_miss_rate": float(np.mean(rtf > 1.0)),
            "budget_miss_rate": float(np.mean(rtf > BUDGETS["dsp_rtf_max"])),
        }
        res["status"] = "PASS" if res["rtf"]["p95"] <= BUDGETS["dsp_rtf_max"] else "FAIL"
        out["timing"][label] = res
        print(
            f"  {label:<24} RTF mean {res['rtf']['mean']:.3f}  p95 {res['rtf']['p95']:.3f}  p99 {res['rtf']['p99']:.3f}  "
            f"lat p95 {res['latency_ms']['p95']:.1f} ms  [{res['status']}]"
        )

    for name, bf in bfs.items():
        if bf is None:
            print(f"  [skip] beamformer {name}: {reg.skipped.get('bf:' + name)}")
            continue
        time_it(lambda c, b=bf: b.process(c, args.sr), f"bf/{name}")

    for name, dn in dns.items():
        if dn is None:
            print(f"  [skip] denoiser {name}: {reg.skipped.get('dn:' + name)}")
            continue
        time_it(lambda c, d=dn: d.process(c[0], args.sr), f"dn/{name}")

    clean_items = [it for it in items if not it.real_array]
    if not clean_items:
        out["quality"]["note"] = "no clean mono references: quality metrics need a manifest of clean mono audio or --synthetic"
        return out

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        for snr in [None] + snrs:
            for it in clean_items:
                mc, clean, az = make_sample(it, snr, args, mic_xy, noise, rng)
                s_in = quality(clean, mc[0], args.sr, False)["si_sdr"]
                for name, bf in bfs.items():
                    if bf is None:
                        continue
                    y, azs = _apply_chunked(lambda c, b=bf: b.process(c, args.sr), mc, n)
                    q = quality(clean, y, args.sr, args.stoi)
                    rows.append(
                        dict(
                            kind="bf",
                            name=name,
                            snr=snr_label(snr),
                            si_sdr=q["si_sdr"],
                            d_si_sdr=(q["si_sdr"] - s_in) if snr is not None else None,
                            stoi=q["stoi"],
                            doa_err=ang_err(float(np.median(azs)), az) if azs else None,
                        )
                    )
                for name, dn in dns.items():
                    if dn is None:
                        continue
                    y, _ = _apply_chunked(lambda c, d=dn: d.process(c, args.sr), mc[0], n)
                    q = quality(clean, y, args.sr, args.stoi)
                    rows.append(
                        dict(
                            kind="dn",
                            name=name,
                            snr=snr_label(snr),
                            si_sdr=q["si_sdr"],
                            d_si_sdr=(q["si_sdr"] - s_in) if snr is not None else None,
                            stoi=q["stoi"],
                            doa_err=None,
                        )
                    )

    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["name"], r["snr"]), []).append(r)

    for (kind, name, snr), rs in groups.items():
        doa = [r["doa_err"] for r in rs if r["doa_err"] is not None]
        out["quality"][f"{kind}/{name}/{snr}"] = {
            "si_sdr": summarize([r["si_sdr"] for r in rs]),
            "delta_si_sdr": summarize([r["d_si_sdr"] for r in rs]),
            "stoi": summarize([r["stoi"] for r in rs]),
            "doa_err_deg": summarize(doa) if doa else None,
            "doa_within_15deg": float(np.mean(np.asarray(doa) <= 15)) if doa else None,
        }

    print("\n  quality (mean dSI-SDR dB vs reference-mic input; 'clean' rows show absolute SI-SDR, capped at 60):")
    for key, v in out["quality"].items():
        if isinstance(v, dict) and "si_sdr" in v:
            extra = f"  DOA med {v['doa_err_deg']['p50']:.0f} deg" if v.get("doa_err_deg") else ""
            val = (
                f"{v['delta_si_sdr']['mean']:+6.2f} dB"
                if v["delta_si_sdr"].get("n")
                else f"{v['si_sdr']['mean']:6.1f} dB (abs)"
            )
            print(f"    {key:<34} {val}{extra}")

    out["raw_rows"] = rows
    return out


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE isolated DSP benchmark")
    p.add_argument("--wav", help="single WAV file for latency/RTF only")
    p.add_argument("--manifest", help="JSONL labelled dataset with audio paths")
    p.add_argument("--synthetic", type=int, default=4, help="N synthetic speech-like clips")
    p.add_argument("--noise-wav", help="real OR noise recording used for SNR sweeps")
    p.add_argument("--snr-sweep", default="20,15,10,5,0")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--reps", type=int, default=30, help="timing repetitions for DSP chunks")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--mic-channels", default="0,1,2,3,4,5")
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument("--diffuse-ratio", type=float, default=0.5)
    p.add_argument("--beamformers", default="ref_channel,ref_srp_das,repo_beamformer")
    p.add_argument("--denoisers", default="passthrough,spectral_gate,noisereduce,deepfilternet,repo_denoiser")
    p.add_argument("--plugin", action="append", default=[])
    p.add_argument("--stoi", action="store_true")
    p.add_argument("--out", default="", help="Output directory for reports")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    a.beamformers = [s for s in a.beamformers.split(",") if s]
    a.denoisers = [s for s in a.denoisers.split(",") if s]
    a.asrs = []
    a.vosk_model = None
    a.whisper_model = "tiny.en"
    a.threads = 2
    return a


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Subsystem: DSP (Beamforming & Denoising)\n" + "*" * 72)
    rng = np.random.default_rng(args.seed)
    snrs = [float(s) for s in args.snr_sweep.split(",") if s]
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    noise = NoiseSource(args.noise_wav, args.sr)
    items = build_items(args, rng)
    reg = Registry(args, mic_xy, grammar=None)

    with ResourceSampler() as rs:
        res = bench_dsp(args, reg, items, mic_xy, noise, snrs)
    res["resources"] = rs.summary()

    if args.out:
        outdir = Path(args.out) / (f"bench_dsp_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "hardware": hw_info(),
            "stages": {"dsp": res},
        }
        with open(outdir / "report_dsp.json", "w") as f:
            json.dump(report, f, indent=2, default=_jd)
        write_csvs(report, outdir)
        print(f"\n Results saved to: {outdir}")


if __name__ == "__main__":
    main()

