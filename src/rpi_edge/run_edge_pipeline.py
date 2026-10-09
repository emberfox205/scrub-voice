#!/usr/bin/env python3
"""Interactive CLI Runner for Raspberry Pi 4 Voice-to-Intent Pipeline.

Prompts surgeon/engineer for spoken utterances and outputs the
schema-validated 10-key JSON envelope with latency and confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None

# Ensure rpi_edge directory is in sys.path
edge_dir = os.path.dirname(os.path.abspath(__file__))
if edge_dir not in sys.path:
    sys.path.insert(0, edge_dir)

from pipeline import VoiceToIntentPipeline


def get_ram_usage_mb() -> float:
    """Return process resident memory in megabytes."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 4 Interactive Voice-to-Intent Pipeline Runner"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to SetFit model directory (default: auto-detected)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to allocate for PyTorch (default: 4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON only (suitable for piping into ZeroMQ / ROS2 / IPC)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" [*] INITIALIZING MEDICAL NLU PIPELINE ON RASPBERRY PI 4")
    print("=" * 65)
    ram_before = get_ram_usage_mb()
    t0 = time.perf_counter()

    try:
        pipeline = VoiceToIntentPipeline(
            model_dir=args.model_dir,
            num_threads=args.threads
        )
    except Exception as e:
        print(f"\n[!] Failed to initialize pipeline: {e}")
        sys.exit(1)

    load_time = time.perf_counter() - t0
    ram_after = get_ram_usage_mb()

    print(f"[OK] Pipeline loaded in {load_time:.2f}s using {args.threads} CPU threads")
    print(f"[OK] Model: {pipeline.model_dir}")
    print(f"[OK] Classes ({len(pipeline.labels)}): {', '.join(pipeline.labels[:5])}...")
    if psutil:
        print(f"[OK] Memory Delta: +{ram_after - ram_before:.1f} MB (Total RSS: {ram_after:.1f} MB)")

    print("\n" + "=" * 65)
    print(" [*] INTERACTIVE NLU TEST (Type 'exit', 'quit' or 'q' to stop)")
    print("=" * 65)

    while True:
        try:
            raw_input_text = input("\nEnter voice command > ").strip()
            if not raw_input_text:
                continue
            if raw_input_text.lower() in ("exit", "quit", "q"):
                print("Exiting interactive pipeline.")
                break

            envelope = pipeline.process(raw_input_text)

            if args.json:
                print(json.dumps(envelope))
            else:
                print(" " + "-" * 61)
                print(f"  Command ID  : {envelope['command_id']}")
                print(f"  Status      : {envelope['status']} | Intent: {envelope['intent']}")
                print(f"  Category    : {envelope['category']} | Priority: {envelope['priority']}")
                print(f"  Confidence  : {envelope['confidence']:.1%}")
                print(f"  Latency     : {envelope['latency_ms']:.2f} ms")
                print(f"  Slots       : {json.dumps(envelope['slots'])}")
                print(" " + "-" * 61)
                print("  Validated JSON Envelope:")
                print(json.dumps(envelope, indent=2))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive pipeline.")
            break
        except Exception as err:
            print(f"[!] Pipeline processing error: {err}")


if __name__ == "__main__":
    main()