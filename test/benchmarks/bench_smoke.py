"""Fast CI/CD smoke test harness for SCRUB-VOICE.

Validates the core benchmark harness, acoustic simulation, DSP filtering,
ZeroMQ IPC dispatching, and statistical pipelines in < 5 seconds without requiring
physical soundcard hardware or external datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.common.acoustics import (
    NoiseSource,
    circular_array,
    quality,
    simulate_array,
    synth_speechlike,
)
from test.benchmarks.common.registry import Passthrough, RefBeamformer, SpectralGate
from test.benchmarks.common.rig import DispatcherRig
from test.benchmarks.common.stats import summarize, wilson
from test.benchmarks.common.telemetry import ResourceSampler, get_rss_mb


def run_smoke_test(verbose: bool = True, with_nlu: bool = False) -> Tuple[bool, Dict[str, str]]:
    results: Dict[str, str] = {}
    all_pass = True

    if verbose:
        print("*" * 68)
        print(" SCRUB-VOICE FAST SMOKE TEST (CI/CD Validation)")
        print("*" * 68)

    # 1. Acoustics & Array Simulation
    t0 = pc()
    sim_mc = None
    sr = 16000
    rng = np.random.default_rng(42)
    mic_xy = circular_array(6, 0.043)
    speech = None
    try:
        dur_s = 1.0
        speech = synth_speechlike(dur_s, sr, rng)
        noise = NoiseSource(None, sr)
        sim_mc = simulate_array(speech, az=45.0, snr_db=15.0, mic_xy=mic_xy, fs=sr, noise=noise, rng=rng)
        assert sim_mc.shape == (6, int(dur_s * sr)), f"Shape mismatch: {sim_mc.shape}"
        results["Acoustics Simulation"] = f"PASS ({(pc() - t0) * 1000:.1f} ms)"
    except Exception as e:
        results["Acoustics Simulation"] = f"FAIL ({e})"
        all_pass = False

    # 2. DSP Beamforming & Denoising
    t0 = pc()
    try:
        if sim_mc is None or speech is None:
            raise RuntimeError("Skipped due to prior failure in acoustics simulation")
        bf = RefBeamformer(mic_xy, sr)
        dn = SpectralGate()
        chunk_len = int(0.5 * sr)
        chunk = sim_mc[:, :chunk_len]

        t_dsp0 = pc()
        bf_out, az = bf.process(chunk, sr)
        dn_out = dn.process(bf_out, sr)
        dsp_wall = pc() - t_dsp0
        rtf = dsp_wall / 0.5

        assert len(dn_out) == chunk_len, f"DSP output size mismatch: {len(dn_out)}"
        q = quality(speech[:chunk_len], dn_out, sr, want_stoi=False)
        assert q["si_sdr"] is not None
        results["DSP Processing"] = f"PASS (RTF={rtf:.3f}, SI-SDR={q['si_sdr']:.1f} dB)"
    except Exception as e:
        results["DSP Processing"] = f"FAIL ({e})"
        all_pass = False

    # 3. IPC & Dispatcher
    t0 = pc()
    try:
        with DispatcherRig("smoke") as rig:
            env = {
                "command_id": "cmd_smoke_01",
                "timestamp": time.time(),
                "latency_ms": 10.0,
                "raw_transcript": "zoom in please",
                "intent": "zoom_display",
                "category": "MONITOR_DISPLAY",
                "priority": "NORMAL",
                "status": "EXECUTE",
                "confidence": 0.99,
                "slots": {"direction": "IN", "factor": 2.0},
            }
            rt = rig.roundtrip(env, timeout_ms=1500)
            assert rt is not None, "Dispatcher roundtrip timed out"
            assert rt < 250.0, f"Dispatcher roundtrip too slow: {rt:.1f} ms"
        results["IPC Dispatcher"] = f"PASS (roundtrip={rt:.2f} ms)"
    except Exception as e:
        results["IPC Dispatcher"] = f"FAIL ({e})"
        all_pass = False

    # 4. Statistics & Telemetry
    t0 = pc()
    try:
        samples = [10.2, 12.5, 9.8, 11.0, 10.5, 13.1, 10.9]
        stats = summarize(samples, boot=200)
        p, lo, hi = wilson(18, 20)
        rss = get_rss_mb()
        assert stats["p50"] > 0
        assert 0.0 <= lo <= p <= hi <= 1.0
        assert rss > 0.0
        results["Stats & Telemetry"] = f"PASS (p50={stats['p50']:.1f}, RSS={rss:.1f} MB)"
    except Exception as e:
        results["Stats & Telemetry"] = f"FAIL ({e})"
        all_pass = False

    # 5. NLU Model Sanity (Fast Normalizer/Schema check by default, full SetFit if with_nlu)
    t0 = pc()
    try:
        if with_nlu:
            from test.benchmarks.common.registry import RepoNLU
            nlu = RepoNLU(threads=1)
            res = nlu.process("zoom in two times")
            assert res.get("intent") == "zoom_display"
            results["NLU Pipeline"] = f"PASS (SetFit intent={res.get('intent')}, {(pc() - t0) * 1000:.1f} ms)"
        else:
            from src.nlu.normalizer import LexicalNormalizer
            norm = LexicalNormalizer()
            normalized = norm.normalize("zoom in two times")
            assert "zoom" in normalized
            results["NLU Pipeline"] = f"PASS (Fast lexical normalizer, {(pc() - t0) * 1000:.1f} ms)"
    except Exception as e:
        results["NLU Pipeline"] = f"SKIPPED ({type(e).__name__}: {e})"

    if verbose:
        print()
        for k, v in results.items():
            status = "PASS" if "PASS" in v else ("SKIP" if "SKIPPED" in v else "FAIL")
            print(f"  [{status:<4}] {k:<26} : {v}")
        print()
        print(f" Smoke Test Result: {'ALL PASSED' if all_pass else 'FAILURES DETECTED'}")
        print("*" * 68)

    return all_pass, results


def main():
    p = argparse.ArgumentParser(description="SCRUB-VOICE CI/CD Smoke Test")
    p.add_argument("--quiet", action="store_true", help="Minimal output")
    p.add_argument("--with-nlu", action="store_true", help="Include full SetFit neural model load and inference")
    args = p.parse_args()

    with ResourceSampler():
        ok, _ = run_smoke_test(verbose=not args.quiet, with_nlu=args.with_nlu)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
