"""Benchmark every engine config on the clip sets and score the results.

Run from the repository root:
    python src/asr/scripts/benchmark.py

Each config runs in its own fresh process (src/asr/scripts/bench_worker.py),
so RAM numbers aren't distorted by models loaded earlier. Raw engine output
is saved to src/asr/results/raw/<config>.json; --score-only rescores those
files without running any engine again.

Clip sets (a missing or empty set is skipped, and the report says so):
  dev          data/asr/commands_dev/manifest.csv   commands, scored for WER/terms
  noise_only   data/asr/noise_only/manifest.csv     silence/noise, hallucinations
  non_command  data/asr/non_command/manifest.csv    speech that isn't a command

Outputs (src/asr/results/, gitignored): summary.csv, per_clip.csv,
errors_<config>.txt, output_fields.csv, machine.json.
All numbers are "dev set": these clips are also used for tuning.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil

ASR_DIR = Path(__file__).resolve().parents[1]  # src/asr
SRC_DIR = ASR_DIR.parent                         # src/ (on sys.path, like the other modules)
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from asr.bench_configs import BASELINES, CONFIGS
from asr.metrics import bootstrap_ci, clip_errors, cluster_resamples, corpus_wer
from asr.normalize import normalize
from asr.vocab import check_all_biasing_leakage, command_group, term_counts

RESULTS_DIR = ASR_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
DATA_DIR = REPO_ROOT / "data" / "asr"
SETS = {
    "dev": DATA_DIR / "commands_dev" / "manifest.csv",
    "noise_only": DATA_DIR / "noise_only" / "manifest.csv",
    "non_command": DATA_DIR / "non_command" / "manifest.csv",
}
SUMMARY_FIELDS = [
    "config", "split", "n_dev", "corpus_wer", "wer_ci_low", "wer_ci_high", "mean_clip_wer",
    "term_recall", "term_precision", "false_terms_per_clip", "unk_rate_dev",
    "n_noise", "hallucination_rate", "false_command_rate_noise", "unk_rate_noise",
    "n_non_command", "false_command_rate_non_command", "unk_rate_non_command",
    "median_latency_s", "p95_latency_s", "mean_rtf", "peak_ram_mb", "load_time_s",
    "confidence_available",
]


def machine_info() -> dict:
    versions = {}
    for pkg in ["faster-whisper", "ctranslate2", "vosk", "numpy", "jiwer"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "packages": versions,
    }


def load_sets(names: list[str]) -> list[dict]:
    clips = []
    for name in names:
        path = SETS[name]
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        if not rows:
            print(f"Set '{name}': no clips at {path.relative_to(REPO_ROOT)} - skipped.")
            continue
        print(f"Set '{name}': {len(rows)} clips")
        for row in rows:
            clips.append({
                "set": name,
                "file": row["file"],
                "reference_text": row.get("reference_text", "") or "",
                "speaker": row.get("speaker", ""),
                "notes": row.get("notes", ""),
            })
    return clips


def tree_rss_mb(proc: psutil.Process) -> float:
    """RSS of a process plus its children (on Windows the venv python.exe is
    a launcher that starts the real interpreter as a child)."""
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


def run_config(name: str, clips: list[dict], runs: int) -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.json"
    with tempfile.TemporaryDirectory() as tmp:
        clips_path = Path(tmp) / "clips.json"
        clips_path.write_text(json.dumps(clips), encoding="utf-8")
        cmd = [sys.executable, str(ASR_DIR / "scripts" / "bench_worker.py"),
               "--config", name, "--clips", str(clips_path), "--out", str(out_path), "--runs", str(runs)]
        started = time.perf_counter()
        proc = subprocess.Popen(cmd)
        ps = psutil.Process(proc.pid)
        polled_peak = 0.0
        while proc.poll() is None:
            try:
                polled_peak = max(polled_peak, tree_rss_mb(ps))
            except psutil.Error:
                pass
            time.sleep(0.02)
    if proc.returncode != 0:
        print(f"  {name}: worker failed with exit code {proc.returncode}")
        return False
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    raw["peak_rss_mb_polled"] = polled_peak
    out_path.write_text(json.dumps(raw), encoding="utf-8")
    print(f"  {name}: done in {time.perf_counter() - started:.0f}s")
    return True


def score_clip(clip: dict, corrections: bool = False) -> dict:
    ref_norm = normalize(clip["reference_text"], corrections=corrections)
    hyp_norm = normalize(clip["hypothesis_text"], corrections=corrections)
    errors, ref_words = clip_errors(ref_norm, hyp_norm)
    said, output, matched = term_counts(ref_norm, hyp_norm)
    return {
        "set": clip["set"],
        "file": clip["file"],
        "speaker": clip["speaker"],
        "notes": clip["notes"],
        "reference_text": clip["reference_text"],
        "hypothesis_text": clip["hypothesis_text"],
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "errors": errors,
        "ref_words": ref_words,
        "clip_wer": errors / ref_words if ref_words else None,
        "terms_said": said,
        "terms_output": output,
        "terms_matched": matched,
        "false_terms": output - matched,
        "has_unk": "[unk]" in clip["hypothesis_text"],
        "confidence": clip["confidence"],
        "latency_median_s": clip["latency_median_s"],
        "rtf": clip["latency_median_s"] / clip["duration_s"],
        "runs_differ": clip["runs_differ"],
    }


def mean_or_none(values) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def summarize(name: str, raw: dict, scored: list[dict], resamples: list[np.ndarray]) -> dict:
    dev = [c for c in scored if c["set"] == "dev"]
    noise = [c for c in scored if c["set"] == "noise_only"]
    non_cmd = [c for c in scored if c["set"] == "non_command"]

    errors = np.array([c["errors"] for c in dev])
    ref_words = np.array([c["ref_words"] for c in dev])
    lo, hi = bootstrap_ci(errors, ref_words, resamples)
    said = sum(c["terms_said"] for c in dev)
    output = sum(c["terms_output"] for c in dev)
    matched = sum(c["terms_matched"] for c in dev)
    latencies = [c["latency_median_s"] for c in dev]

    spoke = [c for c in dev if c["hyp_norm"]]
    with_conf = sum(c["confidence"] is not None for c in spoke)
    conf = "yes" if spoke and with_conf == len(spoke) else ("no" if with_conf == 0 else "partial")

    peak = raw.get("peak_rss_mb_self") or raw.get("peak_rss_mb_polled")
    return {
        "config": name,
        "split": "dev",
        "n_dev": len(dev),
        "corpus_wer": corpus_wer(errors, ref_words),
        "wer_ci_low": lo,
        "wer_ci_high": hi,
        "mean_clip_wer": mean_or_none(c["clip_wer"] for c in dev),
        "term_recall": matched / said if said else None,
        "term_precision": matched / output if output else None,
        "false_terms_per_clip": sum(c["false_terms"] for c in dev) / len(dev),
        "unk_rate_dev": mean_or_none(c["has_unk"] for c in dev),
        "n_noise": len(noise),
        "hallucination_rate": mean_or_none(bool(c["hyp_norm"]) for c in noise),
        "false_command_rate_noise": mean_or_none(c["terms_output"] > 0 for c in noise),
        "unk_rate_noise": mean_or_none(c["has_unk"] for c in noise),
        "n_non_command": len(non_cmd),
        "false_command_rate_non_command": mean_or_none(c["terms_output"] > 0 for c in non_cmd),
        "unk_rate_non_command": mean_or_none(c["has_unk"] for c in non_cmd),
        "median_latency_s": float(np.median(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "mean_rtf": mean_or_none(c["rtf"] for c in dev),
        "peak_ram_mb": peak,
        "load_time_s": raw["load_time_s"],
        "confidence_available": conf,
    }


def output_fields(name: str, raw: dict) -> dict:
    """Which parts of the CLAUDE.md output schema this config really fills."""
    spoke = [c for c in raw["clips"] if c["hypothesis_text"].strip()]
    words = [w for c in spoke for w in c["words"]]
    word_confs = {w[3] for w in words}
    if not words or word_confs == {None}:
        per_word = "null"
    elif len(word_confs) == 1:
        per_word = f"constant {word_confs.pop()}"
    else:
        per_word = "varies"
    return {
        "config": name,
        "confidence": "yes" if spoke and all(c["confidence"] is not None for c in spoke) else "null",
        "words": "yes" if spoke and all(c["words"] for c in spoke) else "missing on some clips",
        "timestamps": "yes" if words and all(w[1] is not None and w[2] is not None for w in words) else "no",
        "per_word_confidence": per_word,
        "runs_differ_clips": sum(c["runs_differ"] for c in raw["clips"]),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_errors_file(name: str, scored: list[dict]) -> None:
    lines = [f"{name} - every clip, reference vs hypothesis (dev set)\n"]
    for c in scored:
        score = f"{c['errors']}/{c['ref_words']} errors" if c["ref_words"] else f"{c['errors']} words output"
        lines.append(f"[{c['set']}] {c['file']}  {score}")
        lines.append(f"  ref: {c['reference_text']!r:45} -> {c['ref_norm']!r}")
        lines.append(f"  hyp: {c['hypothesis_text']!r:45} -> {c['hyp_norm']!r}\n")
    (RESULTS_DIR / f"errors_{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(summaries: list[dict]) -> None:
    cols = [
        ("config", lambda s: s["config"]),
        ("corpus WER [95% CI]", lambda s: f"{fmt(s['corpus_wer'])} [{fmt(s['wer_ci_low'])}-{fmt(s['wer_ci_high'])}]"),
        ("clip WER", lambda s: fmt(s["mean_clip_wer"])),
        ("recall", lambda s: fmt(s["term_recall"])),
        ("precision", lambda s: fmt(s["term_precision"])),
        ("false/clip", lambda s: fmt(s["false_terms_per_clip"])),
        ("halluc", lambda s: fmt(s["hallucination_rate"], 2)),
        ("falsecmd noise", lambda s: fmt(s["false_command_rate_noise"], 2)),
        ("falsecmd speech", lambda s: fmt(s["false_command_rate_non_command"], 2)),
        ("med lat", lambda s: fmt(s["median_latency_s"])),
        ("p95 lat", lambda s: fmt(s["p95_latency_s"])),
        ("RTF", lambda s: fmt(s["mean_rtf"])),
        ("RAM MB", lambda s: fmt(s["peak_ram_mb"], 0)),
        ("load s", lambda s: fmt(s["load_time_s"], 2)),
        ("conf", lambda s: s["confidence_available"]),
    ]
    cells = [[get(s) for _, get in cols] for s in summaries]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, (h, _) in enumerate(cols)]
    print(" | ".join(h.ljust(w) for (h, _), w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in cells:
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASR engine configs (dev set)")
    parser.add_argument("--configs", default=None, help="Comma-separated config names (default: all)")
    parser.add_argument("--sets", default="dev,noise_only,non_command")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per clip (median is kept)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--score-only", action="store_true", help="Rescore src/asr/results/raw/*.json, run nothing")
    args = parser.parse_args()

    names = args.configs.split(",") if args.configs else list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. Options: {list(CONFIGS)}")
    set_names = args.sets.split(",")
    if "dev" not in set_names:
        sys.exit("The dev set is required (WER and timing come from it).")

    RESULTS_DIR.mkdir(exist_ok=True)
    info = machine_info()
    (RESULTS_DIR / "machine.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Machine: {info['os']} | {info['cpu']} | {info['physical_cores']} cores "
          f"({info['logical_cores']} logical) | {info['ram_gb']} GB RAM | Python {info['python']}")

    clips = load_sets(set_names)
    dev_clips = [c for c in clips if c["set"] == "dev"]
    if not dev_clips:
        sys.exit("No dev clips yet - record data/asr/prompts/commands_dev.txt first "
                 "(src/asr/scripts/record_testset.py, see src/asr/CLAUDE.md Phase B).")
    check_all_biasing_leakage([c["reference_text"] for c in dev_clips])
    clusters = [command_group(normalize(c["reference_text"])) for c in dev_clips]
    resamples = cluster_resamples(clusters, n=args.bootstrap, seed=0)
    print(f"Bootstrap: {args.bootstrap} resamples over {len(set(clusters))} command groups "
          f"({len(dev_clips)} dev clips)")

    if not args.score_only:
        print(f"\nRunning {len(names)} config(s), {args.runs} timed run(s) per clip, one process each:")
        names = [n for n in names if run_config(n, clips, args.runs)]

    summaries, per_clip_rows, field_rows = [], [], []
    for name in names:
        raw_path = RAW_DIR / f"{name}.json"
        if not raw_path.exists():
            print(f"  {name}: no raw results at {raw_path.relative_to(REPO_ROOT)} - skipped")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        wanted_sets = set(set_names)
        field_rows.append(output_fields(name, raw))
        # Baselines also get a "+corr" row: same transcripts, scored with the
        # post-correction map (asr/normalize.py) - no extra inference.
        variants = [(name, False)] + ([(f"{name}+corr", True)] if name in BASELINES else [])
        for label, corrections in variants:
            scored = [score_clip(c, corrections) for c in raw["clips"] if c["set"] in wanted_sets]
            summaries.append(summarize(label, raw, scored, resamples))
            per_clip_rows.extend({"config": label, **c} for c in scored)
            write_errors_file(label, scored)

    if not summaries:
        sys.exit("Nothing to score.")
    write_csv(RESULTS_DIR / "summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(RESULTS_DIR / "per_clip.csv", per_clip_rows, list(per_clip_rows[0].keys()))
    write_csv(RESULTS_DIR / "output_fields.csv", field_rows, list(field_rows[0].keys()))

    print("\n=== Summary (dev set - tuned on these clips, numbers are optimistic) ===")
    print_table(summaries)
    print("\n=== Output fields per config ===")
    for row in field_rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
