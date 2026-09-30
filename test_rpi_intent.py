#!/usr/bin/env python3
"""Edge Intent Classification Test & Benchmark Script for Raspberry Pi 4.

Supports:
1. Cold-start load time & RAM profiling.
2. Latency benchmark across medical intent categories (mean, min, max, p95).
3. Live interactive CLI prompt for testing real-time spoken utterances.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import psutil
except ImportError:
    psutil = None

try:
    from src.nlu.intent_classifier import IntentClassifier
except ImportError:
    from nlu.intent_classifier import IntentClassifier

# Default model directory (Downloads directory on Raspberry Pi 4)
DEFAULT_MODEL_DIR = "/home/admin/Downloads/setfit_bge-micro-v2_8k"
if not os.path.exists(DEFAULT_MODEL_DIR):
    local_fallback = os.path.join(current_dir, "models", "setfit_bge-micro-v2_8k")
    if os.path.exists(local_fallback):
        DEFAULT_MODEL_DIR = local_fallback

BENCHMARK_SAMPLES: List[Tuple[str, str]] = [
    ("Please zoom in three times magnification immediately", "ZOOM_IN"),
    ("Please zoom out to wide view on screen", "ZOOM_OUT"),
    ("Can you pan right 50 pixels now", "PAN"),
    ("Advance forward 15 slices", "SLICE_NEXT"),
    ("Now previous 1 slices please", "SLICE_PREV"),
    ("Assistant navigate to slice number 182 immediately", "GOTO_SLICE"),
    ("Please set contrast to bone window immediately", "CONTRAST_SET"),
    ("Can you open spine MRI on primary monitor", "IMAGE_OPEN"),
    ("Hey assistant hide this medical viewport", "IMAGE_CLOSE"),
    ("Can you reset viewport to default", "VIEW_RESET"),
    ("Please record event cystic duct clamped note clip placed", "LOG_EVENT"),
    ("What is the weather like today", "OUT_OF_SCOPE"),
]


def get_ram_usage_mb() -> float:
    """Return process resident memory in megabytes."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(classifier: IntentClassifier, repetitions: int = 5) -> None:
    """Run automated benchmark suite and print performance metrics."""
    print("\n" + "=" * 65)
    print(" [>] RUNNING NLU BENCHMARK ON RASPBERRY PI 4")
    print("=" * 65)

    # Warmup
    print(" Warming up CPU inference engine (2 runs)...")
    classifier.predict("warmup run one")
    classifier.predict("warmup run two")

    latencies: List[float] = []
    correct_matches = 0
    total_samples = len(BENCHMARK_SAMPLES)

    print("\nEvaluating Intent Accuracy & Latency:")
    print("-" * 65)
    print(f"{'Utterance':<35} | {'Predicted':<13} | {'Conf':<6} | {'Time'}")
    print("-" * 65)

    for utterance, expected_intent in BENCHMARK_SAMPLES:
        sample_latencies = []
        last_result = None

        for _ in range(repetitions):
            res = classifier.predict(utterance)
            sample_latencies.append(res["latency_ms"])
            last_result = res

        latencies.extend(sample_latencies)
        avg_sample_ms = sum(sample_latencies) / len(sample_latencies)

        predicted_intent = last_result["intent"]
        confidence = last_result["confidence"]

        if predicted_intent == expected_intent:
            correct_matches += 1
            status_symbol = "[OK]"
        else:
            status_symbol = "[XX]"

        print(
            f"{status_symbol} {utterance:<31} | {predicted_intent:<13} | "
            f"{confidence:>5.1%} | {avg_sample_ms:>5.1f}ms"
        )

    # Compute statistics
    latencies.sort()
    mean_latency = sum(latencies) / len(latencies)
    min_latency = latencies[0]
    max_latency = latencies[-1]
    p95_idx = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_idx, len(latencies) - 1)]

    current_ram = get_ram_usage_mb()

    print("=" * 65)
    print(" [i] BENCHMARK SUMMARY")
    print("=" * 65)
    print(f" Intent Match Rate   : {correct_matches}/{total_samples} ({correct_matches/total_samples:.1%})")
    print(f" Mean Latency        : {mean_latency:.2f} ms")
    print(f" Min Latency         : {min_latency:.2f} ms")
    print(f" Max Latency         : {max_latency:.2f} ms")
    print(f" P95 Latency         : {p95_latency:.2f} ms")
    if psutil:
        print(f" Process RAM Footprint: {current_ram:.1f} MB")
    print("=" * 65)


def run_interactive(classifier: IntentClassifier) -> None:
    """Interactive loop allowing manual utterance inputs."""
    print("\n" + "=" * 65)
    print(" [*] INTERACTIVE NLU TEST (Type 'exit' or 'quit' to stop)")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nEnter voice command > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive test.")
                break

            result = classifier.predict(user_input)
            print(f" -> Intent    : {result['intent']}")
            print(f" -> Confidence: {result['confidence']:.1%}")
            print(f" -> Latency   : {result['latency_ms']:.2f} ms")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive test.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 4 Intent Classifier Runner & Benchmark"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to SetFit model directory",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run automated benchmark suite only",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive test prompt only",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test utterance during benchmark (default: 3)",
    )

    args = parser.parse_args()

    print(f"Initializing Edge NLU System on Raspberry Pi 4...")
    ram_before = get_ram_usage_mb()
    t0 = time.perf_counter()

    try:
        classifier = IntentClassifier(model_path=args.model_dir, device="cpu")
    except Exception as e:
        print(f"\n[!] Error loading model: {e}")
        sys.exit(1)

    load_time = time.perf_counter() - t0
    ram_after = get_ram_usage_mb()

    print(f"[OK] Model loaded in {load_time:.2f}s")
    if psutil:
        print(f"[OK] Memory Delta: +{ram_after - ram_before:.1f} MB (Total RSS: {ram_after:.1f} MB)")

    if args.benchmark:
        run_benchmark(classifier, repetitions=args.runs)
    elif args.interactive:
        run_interactive(classifier)
    else:
        # Default: Run benchmark then drop into interactive mode
        run_benchmark(classifier, repetitions=args.runs)
        run_interactive(classifier)


if __name__ == "__main__":
    main()
