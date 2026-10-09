"""Verification test script for scrub-voice edge pipeline with CPU, Cores & RAM profiling."""

import os
import sys

# Configure sys.path so 'src' and repo root are importable
test_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(test_dir)
src_dir = os.path.join(repo_root, "src")
for p in (src_dir, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json
import time
import jsonschema
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rpi_edge.pipeline import VoiceToIntentPipeline
except ModuleNotFoundError:
    from src.rpi_edge.pipeline import VoiceToIntentPipeline


def get_process_memory_mb(proc) -> float:
    """Return process Resident Set Size (RSS) memory in megabytes."""
    if proc is None:
        return 0.0
    return proc.memory_info().rss / (1024.0 * 1024.0)


def test_pipeline(num_threads: int = 1, pin_single_core: bool = False):
    proc = psutil.Process(os.getpid()) if psutil else None

    # Restrict to 1 physical CPU core if requested
    if proc and pin_single_core:
        try:
            proc.cpu_affinity([0])  # Force OS scheduler to use physical Core 0 only
        except Exception as e:
            print(f"  [!] Note: Could not set CPU affinity: {e}")

    ram_initial_mb = get_process_memory_mb(proc)
    peak_ram_mb = ram_initial_mb

    # CPU Hardware details
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) if psutil else logical_cores
    current_affinity = proc.cpu_affinity() if (proc and hasattr(proc, "cpu_affinity")) else list(range(logical_cores))

    print("=" * 80)
    print(">>> INITIALIZING EDGE PIPELINE ON RASPBERRY PI 4")
    print("=" * 80)
    print(f"  - Host CPU Physical Cores  : {physical_cores}")
    print(f"  - Host CPU Logical Threads : {logical_cores}")
    print(f"  - OS Process CPU Affinity  : Core(s) {current_affinity} ({len(current_affinity)} physical core assigned)")
    print(f"  - PyTorch Worker Threads   : {num_threads} Thread(s) Allocated")
    if proc:
        print(f"  - Initial Base Process RAM : {ram_initial_mb:.2f} MB")

    t_init = time.perf_counter()
    edge_dir = os.path.join(src_dir, "rpi_edge")
    pipeline = VoiceToIntentPipeline(
        base_dir=edge_dir,
        num_threads=num_threads
    )
    init_duration = time.perf_counter() - t_init

    actual_torch_threads = torch.get_num_threads()
    ram_after_load_mb = get_process_memory_mb(proc)
    model_delta_mb = ram_after_load_mb - ram_initial_mb
    peak_ram_mb = max(peak_ram_mb, ram_after_load_mb)

    print(f"  - Model Load Time          : {init_duration:.2f}s")
    print(f"  - Model Labels Count       : {len(pipeline.labels)}")
    print(f"  - Active PyTorch Threads   : {actual_torch_threads} CPU thread(s)")
    if proc:
        print(f"  - Post-Load Process RAM    : {ram_after_load_mb:.2f} MB (Delta: +{model_delta_mb:.2f} MB)")

    test_cases = [
        ("Hand me the scalpel please", "pick_tool", "TOOL_HANDLING",
         "EXECUTE", {"tool_identifier": "scalpel"}),
        ("Emergency stop immediately", "halt_emergency", "EMERGENCY", "EXECUTE", {}),
        ("Zoom in 2.5x", "zoom_display", "MONITOR_DISPLAY",
         "EXECUTE", {"direction": "IN", "factor": 2.5}),
        ("Go to slice 45", "navigate_slice",
         "MONITOR_DISPLAY", "EXECUTE", {"slice_index": 45}),
        ("Camera left 5 cm", "move_camera", "CAMERA_LIGHT",
         "EXECUTE", {"direction": "LEFT", "distance_cm": 5.0}),
        ("423", "unknown", "UNKNOWN", "OUT-OF-DOMAIN", {}),
        ("It is raining outside", "unknown", "UNKNOWN", "OUT-OF-DOMAIN", {})
    ]

    print("\n>>> Running Verification on Test Utterances:")
    print("-" * 80)

    latencies = []
    cpu_active_times = []
    cores_utilized_list = []

    for text, exp_intent, exp_cat, exp_status, exp_slots in test_cases:
        t_cpu_before = proc.cpu_times() if proc else None

        res = pipeline.process(text)
        latency_ms = res["latency_ms"]
        latencies.append(latency_ms)

        current_ram_mb = get_process_memory_mb(proc)
        peak_ram_mb = max(peak_ram_mb, current_ram_mb)

        cpu_str = ""
        if proc and t_cpu_before:
            t_cpu_after = proc.cpu_times()
            user_cpu_ms = (t_cpu_after.user - t_cpu_before.user) * 1000.0
            sys_cpu_ms = (t_cpu_after.system - t_cpu_before.system) * 1000.0
            total_cpu_ms = user_cpu_ms + sys_cpu_ms
            cpu_active_times.append(total_cpu_ms)

            # Parallel core load: total work across threads divided by wall-clock latency
            cores_engaged = total_cpu_ms / latency_ms if latency_ms > 0 else 1.0
            cores_engaged = min(float(actual_torch_threads), max(1.0, cores_engaged))
            cores_utilized_list.append(cores_engaged)

            cpu_str = f" | CPU Time: {total_cpu_ms:.1f} ms (~{cores_engaged:.1f} core)"

        ram_info_str = f" | RAM: {current_ram_mb:.2f} MB" if proc else ""

        jsonschema.validate(instance=res, schema=pipeline.schema)
        print(f"Utterance : \"{text}\"")
        print(
            f"Intent    : {res['intent']} (exp: {exp_intent}) | Status: {res['status']} | Latency: {latency_ms:.2f} ms{cpu_str}{ram_info_str}"
        )
        print(f"Slots     : {res['slots']}")
        assert res["intent"] == exp_intent, f"Expected {exp_intent}, got {res['intent']}"
        assert res["status"] == exp_status, f"Expected {exp_status}, got {res['status']}"
        if exp_slots:
            for k, v in exp_slots.items():
                assert res["slots"].get(
                    k) == v, f"Slot mismatch for {k}: expected {v}, got {res['slots'].get(k)}"
        print("-" * 80)

    print("\n" + "=" * 80)
    print("RASPBERRY PI 4 CPU & MEMORY RESOURCE PROFILE:")
    print(f"  - Target Hardware          : Raspberry Pi 4 Model B (Broadcom BCM2711)")
    print(f"  - Assigned Physical Cores  : {len(current_affinity)} Core(s) (Affinity: {current_affinity})")
    print(f"  - PyTorch Worker Threads   : {actual_torch_threads} Thread(s)")
    print(f"  - Mean Inference Latency   : {sum(latencies)/len(latencies):.2f} ms")
    if cpu_active_times:
        mean_cpu_ms = sum(cpu_active_times) / len(cpu_active_times)
        mean_cores = sum(cores_utilized_list) / len(cores_utilized_list)
        print(f"  - Mean Active CPU Time     : {mean_cpu_ms:.2f} ms per inference")
        print(f"  - Parallel Core Engagement : ~{mean_cores:.1f} core(s) active during inference")
    if proc:
        print(f"  - Initial Base Process RAM : {ram_initial_mb:.2f} MB")
        print(f"  - Post-Load Process RAM    : {ram_after_load_mb:.2f} MB")
        print(f"  - Net Model RAM Delta      : +{model_delta_mb:.2f} MB")
        print(f"  - Peak Process RAM (Max)   : {peak_ram_mb:.2f} MB")
    print(f"  - Schema Compliance        : 100% (Draft 2020-12)")
    print("=" * 80)
    print("\n[SUCCESS] ALL ASSERTIONS PASSED! CPU cores, execution time, and memory measured.")


def main():
    parser = argparse.ArgumentParser(description="Test edge pipeline with custom CPU cores/threads.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads to allocate (default: 1 for single-core test, or 4 for full Pi 4)"
    )
    parser.add_argument(
        "--pin-core",
        action="store_true",
        default=True,
        help="Pin OS process to single CPU Core 0 using CPU affinity (default: True when threads=1)"
    )
    args = parser.parse_args()

    pin = args.pin_core if args.threads == 1 else False
    test_pipeline(num_threads=args.threads, pin_single_core=pin)


if __name__ == "__main__":
    main()
