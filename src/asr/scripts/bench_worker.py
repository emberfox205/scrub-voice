"""Benchmark worker: runs ONE engine config over the given clip sets in its
own process, so its RAM reading isn't mixed up with other models. Started
by benchmark.py; not meant to be run by hand (but you can, for debugging).

Timing: the model is loaded once (load time measured separately), one
warm-up transcription is run and thrown away, then every clip is
transcribed --runs times and the median latency is kept. The transcript
from the first run is the one that gets scored.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil

ASR_DIR = Path(__file__).resolve().parents[1]  # src/asr
SRC_DIR = ASR_DIR.parent                         # src/ (on sys.path, like the other modules)
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from asr.audio import TARGET_SR, load_audio
from asr.bench_configs import CONFIGS


def own_peak_rss_mb() -> float | None:
    """Exact peak memory of this process, where the OS tracks it."""
    if platform.system() == "Windows":
        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024 if platform.system() == "Linux" else kb / (1024 * 1024)
    except ImportError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one engine config over clip sets")
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--clips", required=True, type=Path, help="JSON list of {set, file, ...} clip rows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    clips = json.loads(args.clips.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    engine = CONFIGS[args.config]()
    load_time_s = time.perf_counter() - t0

    warmup_audio = load_audio(REPO_ROOT / clips[0]["file"])
    t0 = time.perf_counter()
    engine.transcribe(warmup_audio)
    warmup_s = time.perf_counter() - t0

    print(f"  [{args.config}] model loaded in {load_time_s:.1f}s, warm-up {warmup_s:.1f}s", flush=True)
    results = []
    for i, clip in enumerate(clips, start=1):
        if i % 10 == 0 or i == len(clips):
            print(f"  [{args.config}] clip {i}/{len(clips)}", flush=True)
        audio = load_audio(REPO_ROOT / clip["file"])
        latencies, texts, first = [], [], None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            result = engine.transcribe(audio)
            latencies.append(time.perf_counter() - t0)
            texts.append(result.text)
            if first is None:
                first = result
        results.append({
            **clip,
            "duration_s": len(audio) / TARGET_SR,
            "hypothesis_text": first.text,
            "confidence": first.confidence,
            "words": [[w.word, w.start, w.end, w.confidence] for w in first.words],
            "latency_runs_s": latencies,
            "latency_median_s": statistics.median(latencies),
            "runs_differ": len(set(texts)) > 1,
        })

    args.out.write_text(
        json.dumps({
            "config": args.config,
            "load_time_s": load_time_s,
            "warmup_s": warmup_s,
            "peak_rss_mb_self": own_peak_rss_mb(),
            "clips": results,
        }),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
