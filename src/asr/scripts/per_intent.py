"""Per-intent breakdown of a benchmark run: which commands fail, per config.

Reads src/asr/results/per_clip.csv (written by benchmark.py) and prints, for
every config and every intent (the manifest `notes` column):
- recall: command words said that came out right
- clean: share of clips whose command words are all right and nothing was
  invented - a stand-in for "the language understanding stage gets the
  right command and slots"

Run from the repository root:
    python src/asr/scripts/per_intent.py
    python src/asr/scripts/per_intent.py --configs vosk-grammar,faster-whisper-tiny.en-beam1+prompt
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ASR_DIR = Path(__file__).resolve().parents[1]  # src/asr


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-intent results of a benchmark run")
    parser.add_argument("--per-clip", type=Path, default=ASR_DIR / "results" / "per_clip.csv")
    parser.add_argument("--configs", default=None, help="Comma-separated config names (default: all)")
    parser.add_argument("--set", default="dev")
    args = parser.parse_args()

    with args.per_clip.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == args.set]
    configs = args.configs.split(",") if args.configs else list(dict.fromkeys(r["config"] for r in rows))
    intents = list(dict.fromkeys(r["notes"] for r in rows))

    # stats[config][intent] = [terms said, terms matched, clips, clean clips]
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for r in rows:
        said, matched, output = int(r["terms_said"]), int(r["terms_matched"]), int(r["terms_output"])
        for key in (r["notes"], "ALL"):
            s = stats[r["config"]][key]
            s[0] += said
            s[1] += matched
            s[2] += 1
            s[3] += int(matched == said and output == said)

    width = max(len(c) for c in configs)
    print(f"Set '{args.set}': recall / clean per intent (clean = all command words right, none invented)\n")
    print(f"{'intent':16} " + "  ".join(f"{i:>13}" for i in range(1, len(configs) + 1)))
    for intent in intents + ["ALL"]:
        cells = []
        for config in configs:
            said, matched, clips, clean = stats[config][intent]
            recall = f"{matched / said:.2f}" if said else " n/a"
            cells.append(f"{recall} / {clean:>2}/{clips:<2}")
        print(f"{intent:16} " + "  ".join(f"{c:>13}" for c in cells))
    print("\nColumns:")
    for i, config in enumerate(configs, start=1):
        print(f"  {i}. {config:{width}}")


if __name__ == "__main__":
    main()
