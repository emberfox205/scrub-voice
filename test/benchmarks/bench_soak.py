"""Extended Soak & Burn-In Benchmark (Memory Leaks & Latency Drift).

Evaluates edge runtime stability under sustained load:
1. Continuous streaming execution across hours of synthetic or real speech.
2. Tracks process memory (RSS) and calculates leak rate (MB/h) via linear regression.
3. Detects latency degradation and pipeline drift over time (ms/h).
4. Monitors Raspberry Pi SoC thermal throttling flags and CPU temperature.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.bench_e2e import make_variants, run_utterance
from test.benchmarks.common.acoustics import (
    NoiseSource,
    build_items,
    circular_array,
    make_sample,
)
from test.benchmarks.common.registry import BUDGETS, Registry, RepoNLU
from test.benchmarks.common.reporting import _jd
from test.benchmarks.common.rig import DispatcherRig
from test.benchmarks.common.telemetry import ResourceSampler, get_rss_mb, hw_info


def bench_soak(args, reg: Registry, nlu, rig, items, mic_xy: np.ndarray, noise: NoiseSource) -> Dict[str, Any]:
    variants = make_variants(args, reg)
    if not variants or nlu is None:
        return {"status": "SKIPPED", "reason": "No runnable variant or NLU unavailable"}

    var, rng = variants[0], np.random.default_rng(7)
    duration_s = args.soak_minutes * 60.0
    end_time = pc() + duration_s
    t_start = pc()
    series: List[tuple] = []

    print(f"\n" + "=" * 72 + f"\n [SOAK] Stress & Stability Burn-in ({args.soak_minutes:g} min on '{var.name}')\n" + "=" * 72)
    print("  Streaming continuous utterances... Press Ctrl+C to abort early.\n")

    iteration = 0
    try:
        while pc() < end_time:
            for it in items:
                if pc() >= end_time:
                    break
                mc, _, _ = make_sample(it, 10.0, args, mic_xy, noise, rng)
                r = run_utterance(reg, var, nlu, rig, mc, args.sr, args.chunk_s, 0.0)
                elapsed_h = (pc() - t_start) / 3600.0
                curr_rss = get_rss_mb()
                series.append((elapsed_h, curr_rss, r["e2e_ms"]))
                iteration += 1

                if iteration % 20 == 0:
                    print(
                        f"  [{iteration:4d} iters] elapsed: {(pc() - t_start):.1f}s | "
                        f"RSS: {curr_rss:.1f} MB | last e2e: {r['e2e_ms']:.1f} ms"
                    )
    except KeyboardInterrupt:
        print("\n  [!] Soak interrupted by user. Summarizing completed iterations...")

    if not series:
        return {"status": "ERROR", "reason": "No iterations completed"}

    a = np.asarray(series)
    hours = a[:, 0]
    rss = a[:, 1]
    e2e_lat = a[:, 2]

    # Linear regression slope (drift rate)
    rss_slope = float(np.polyfit(hours, rss, 1)[0]) if len(a) > 2 and hours[-1] > 0 else 0.0
    lat_slope = float(np.polyfit(hours, e2e_lat, 1)[0]) if len(a) > 2 and hours[-1] > 0 else 0.0
    peak_rss = float(rss.max())

    passed = (rss_slope < 5.0) and (peak_rss <= BUDGETS["rss_total_mb_max"])
    status = "PASS" if passed else "FAIL"

    res = {
        "status": status,
        "variant": var.name,
        "iterations": len(a),
        "duration_minutes": round((pc() - t_start) / 60.0, 2),
        "rss_leak_slope_mb_per_h": round(rss_slope, 3),
        "rss_peak_mb": round(peak_rss, 1),
        "latency_drift_ms_per_h": round(lat_slope, 2),
        "series_downsampled": a[:: max(1, len(a) // 300)].tolist(),
    }

    print(
        f"\n  Result: [{status}] across {len(a)} utterances in {(pc() - t_start):.1f}s\n"
        f"    RSS peak: {peak_rss:.1f} MB (budget: <= {BUDGETS['rss_total_mb_max']} MB)\n"
        f"    RSS leak slope: {rss_slope:+.2f} MB/h (budget: < 5.0 MB/h)\n"
        f"    Latency drift slope: {lat_slope:+.2f} ms/h\n"
    )
    return res


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE Soak and Memory Leak Benchmark")
    p.add_argument("--soak-minutes", type=float, default=2.0, help="Duration of stress soak (min)")
    p.add_argument("--synthetic", type=int, default=4, help="N synthetic speech clips")
    p.add_argument("--manifest", help="Optional labelled dataset")
    p.add_argument("--wav", help="Optional single audio WAV")
    p.add_argument("--noise-wav", help="Noise WAV for background injection")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--mic-channels", default="0,1,2,3,4,5")
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument("--diffuse-ratio", type=float, default=0.5)
    p.add_argument("--beamformers", default="repo_beamformer,ref_srp_das")
    p.add_argument("--denoisers", default="repo_denoiser,spectral_gate")
    p.add_argument("--asrs", default="vosk_grammar,repo_asr")
    p.add_argument("--plugin", action="append", default=[])
    default_vosk = REPO_ROOT / "src" / "asr" / "models" / "vosk-model-small-en-us-0.15"
    p.add_argument("--vosk-model", default=str(default_vosk) if default_vosk.exists() else None)
    p.add_argument("--grammar", help="Path to grammar file")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-dispatcher", action="store_true")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "soak"), help="Output directory")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    a.beamformers = [s for s in a.beamformers.split(",") if s]
    a.denoisers = [s for s in a.denoisers.split(",") if s]
    a.asrs = [s for s in a.asrs.split(",") if s]
    a.baselines = False
    a.max_items = 0
    return a


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Operational: Soak & Memory Leak Test\n" + "*" * 72)
    rng = np.random.default_rng(42)
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    noise = NoiseSource(args.noise_wav, args.sr)
    items = build_items(args, rng)

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
    if nlu is not None and not args.no_dispatcher:
        try:
            rig = DispatcherRig("soak")
        except Exception as e:
            print(f"  [warn] Dispatcher rig unavailable ({e})")

    outdir = Path(args.out) / (f"soak_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
    outdir.mkdir(parents=True, exist_ok=True)

    with ResourceSampler() as rs:
        res = bench_soak(args, reg, nlu, rig, items, mic_xy, noise)
    res["resources"] = rs.summary()

    if rig:
        rig.close()

    report = {
        "timestamp": time.time(),
        "hardware": hw_info(),
        "soak": res,
    }
    with open(outdir / "report_soak.json", "w") as f:
        json.dump(report, f, indent=2, default=_jd)

    if res.get("series_downsampled"):
        with open(outdir / "soak_timeseries.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["elapsed_hours", "rss_mb", "e2e_ms"])
            w.writerows(res["series_downsampled"])

    print(f"\n Results saved to: {outdir}")


if __name__ == "__main__":
    main()

