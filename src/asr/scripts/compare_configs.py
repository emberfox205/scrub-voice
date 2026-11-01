"""Which differences between configs are real? Paired cluster bootstrap on
src/asr/results/per_clip.csv (dev set): for each pair A - B, the corpus-WER
difference and the term-recall difference with a 95% interval. If the
interval contains 0, the difference can't be told apart from noise.

Uses the same command-group resampling as benchmark.py (asr/metrics.py), so
both configs are compared on identical resamples.

    python src/asr/scripts/compare_configs.py
    python src/asr/scripts/compare_configs.py --pairs "vosk-grammar:vosk-freeform"
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ASR_DIR = Path(__file__).resolve().parents[1]  # src/asr
SRC_DIR = ASR_DIR.parent                         # src/ (on sys.path, like the other modules)
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from asr.metrics import cluster_resamples, paired_diff_ci
from asr.vocab import cluster_key

DEFAULT_PAIRS = [
    # the grammar vs the same engine without it
    ("vosk-grammar", "vosk-freeform"),
    # free-form recognizer for log_event free text vs grammar alone
    ("vosk-grammar+freeform-log", "vosk-grammar"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of benchmark configs")
    parser.add_argument("--per-clip", type=Path, default=ASR_DIR / "results" / "per_clip.csv")
    parser.add_argument("--pairs", default=None, help='Comma-separated "A:B" pairs (default: built-in list)')
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    with args.per_clip.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == "dev"]
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
    for clips in by_config.values():
        clips.sort(key=lambda r: r["file"])

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else DEFAULT_PAIRS
    reference = next(iter(by_config.values()))
    ref_words = np.array([int(r["ref_words"]) for r in reference])
    terms_said = np.array([int(r["terms_said"]) for r in reference])
    clusters = [cluster_key(r["notes"], r["ref_norm"]) for r in reference]
    resamples = cluster_resamples(clusters, n=args.bootstrap, seed=0)

    def column(config: str, name: str) -> np.ndarray:
        return np.array([int(r[name]) for r in by_config[config]])

    def verdict(lo: float, hi: float) -> str:
        return "REAL" if lo > 0 or hi < 0 else "noise"

    print(f"{'A - B':78} {'corpus WER diff [95% CI]':33} term recall diff [95% CI]")
    for a, b in pairs:
        if a not in by_config or b not in by_config:
            print(f"{a} - {b}: missing in {args.per_clip.name}, skipped")
            continue
        point, lo, hi = paired_diff_ci(column(a, "errors"), column(b, "errors"), ref_words, resamples)
        matched_a, matched_b = column(a, "terms_matched"), column(b, "terms_matched")
        diffs = [(matched_a[i].sum() - matched_b[i].sum()) / terms_said[i].sum() for i in resamples]
        t_point = (matched_a.sum() - matched_b.sum()) / terms_said.sum()
        t_lo, t_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        print(
            f"{a} - {b}".ljust(78),
            f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict(lo, hi):5}",
            f"  {t_point:+.3f} [{t_lo:+.3f}, {t_hi:+.3f}] {verdict(t_lo, t_hi)}",
        )


if __name__ == "__main__":
    main()
