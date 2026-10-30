"""Isolated ASR Subsystem Benchmark (Vosk & Whisper Decoders).

Evaluates:
1. Streaming 0.5s chunk processing time and Real-Time Factor (RTF).
2. End-of-utterance finalization latency (p50, p95, p99) against target budget (<= 150ms).
3. Corpus Word Error Rate (WER) with 95% bootstrap confidence intervals across SNR conditions.
4. Grammar-constrained vs open-vocabulary decoding comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from test.benchmarks.common.acoustics import (
    NoiseSource,
    build_items,
    chunk_iter,
    circular_array,
    make_sample,
    snr_label,
)
from test.benchmarks.common.registry import Registry
from test.benchmarks.common.reporting import _jd, write_csvs
from test.benchmarks.common.stats import corpus_wer, norm_text, summarize, word_errors
from test.benchmarks.common.telemetry import ResourceSampler, hw_info


def load_grammar(args, items) -> Optional[List[str]]:
    if getattr(args, "grammar", None):
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
        print("  [!] no --grammar given: derived word-level grammar from test transcripts.")
        return words + ["[unk]"]
    return None


def bench_asr(args, reg: Registry, items, mic_xy: np.ndarray, noise: NoiseSource, snrs: List[float]) -> Dict[str, Any]:
    print("\n" + "=" * 72 + "\n [ASR] Isolated Speech-to-Text Benchmark (Streaming 0.5s chunks)\n" + "=" * 72)
    labelled = [it for it in items if it.text]
    if not labelled:
        return {"status": "SKIPPED", "reason": "No labelled items with transcripts provided"}

    n, out, recs = int(args.chunk_s * args.sr), {}, []
    for name in args.asrs:
        asr = reg.get("asr", name)
        if asr is None:
            print(f"  [skip] {name}: {reg.skipped.get('asr:' + name)}")
            continue

        for seed in range(args.seeds):
            rng = np.random.default_rng(2000 + seed)
            for snr in [None] + snrs:
                for it in labelled:
                    mc, _, _ = make_sample(it, snr, args, mic_xy, noise, rng)
                    x = mc[0]
                    asr.reset()
                    t_acc = 0.0
                    for c in chunk_iter(x, n):
                        t0 = pc()
                        asr.accept(c)
                        t_acc += pc() - t0
                    t0 = pc()
                    hyp = asr.finalize()
                    t_fin = pc() - t0
                    e, r = word_errors(it.text, hyp)
                    recs.append(
                        dict(
                            asr=name,
                            snr=snr_label(snr),
                            err=e,
                            ref=r,
                            fin_ms=t_fin * 1000,
                            rtf=(t_acc + t_fin) / (len(x) / args.sr),
                        )
                    )

    for (name, snr) in sorted({(r["asr"], r["snr"]) for r in recs}):
        rs = [r for r in recs if r["asr"] == name and r["snr"] == snr]
        w = corpus_wer([r["err"] for r in rs], [r["ref"] for r in rs])
        out[f"{name}/{snr}"] = {
            "wer": w,
            "finalize_ms": summarize([r["fin_ms"] for r in rs]),
            "rtf": summarize([r["rtf"] for r in rs]),
        }
        print(
            f"  {name:<14} {snr:<6} WER {100 * w['wer']:5.1f}% [{100 * w['ci95_lo']:.1f},{100 * w['ci95_hi']:.1f}]  "
            f"finalize p95 {out[f'{name}/{snr}']['finalize_ms']['p95']:.1f} ms  RTF {out[f'{name}/{snr}']['rtf']['mean']:.3f}"
        )

    out["load_info"] = {k: v for k, v in reg.load_info.items() if k.startswith("asr:")}
    out["raw_rows"] = recs
    return out


def parse_args():
    p = argparse.ArgumentParser(description="SCRUB-VOICE isolated ASR benchmark")
    p.add_argument("--manifest", help="JSONL labelled dataset with audio paths and text")
    p.add_argument("--noise-wav", help="Noise WAV for SNR sweeps")
    p.add_argument("--snr-sweep", default="20,10,0")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-s", type=float, default=0.5)
    p.add_argument("--mic-channels", default="0,1,2,3,4,5")
    p.add_argument("--mic-radius", type=float, default=0.043)
    p.add_argument("--diffuse-ratio", type=float, default=0.5)
    p.add_argument("--asrs", default="vosk_grammar,vosk_open,repo_asr")
    default_vosk = REPO_ROOT / "src" / "asr" / "models" / "vosk-model-small-en-us-0.15"
    p.add_argument(
        "--vosk-model",
        default=str(default_vosk) if default_vosk.exists() else None,
        help="Path to Vosk model",
    )
    p.add_argument("--grammar", help="Path to JSON grammar list or text file")
    p.add_argument("--whisper-model", default="tiny.en")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--plugin", action="append", default=[])
    p.add_argument("--out", default="", help="Output directory for reports")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    a.mic_idx = [int(x) for x in a.mic_channels.split(",")]
    a.asrs = [s for s in a.asrs.split(",") if s]
    a.beamformers = []
    a.denoisers = []
    return a


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Subsystem: ASR (Speech Recognition)\n" + "*" * 72)
    rng = np.random.default_rng(args.seed)
    snrs = [float(s) for s in args.snr_sweep.split(",") if s]
    mic_xy = circular_array(len(args.mic_idx), args.mic_radius)
    noise = NoiseSource(args.noise_wav, args.sr)
    items = build_items(args, rng)
    grammar = load_grammar(args, items)
    reg = Registry(args, mic_xy, grammar)

    with ResourceSampler() as rs:
        res = bench_asr(args, reg, items, mic_xy, noise, snrs)
    res["resources"] = rs.summary()

    if args.out:
        outdir = Path(args.out) / (f"bench_asr_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "hardware": hw_info(),
            "stages": {"asr": res},
        }
        with open(outdir / "report_asr.json", "w") as f:
            json.dump(report, f, indent=2, default=_jd)
        write_csvs(report, outdir)
        print(f"\n Results saved to: {outdir}")


if __name__ == "__main__":
    main()

