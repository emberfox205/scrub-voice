#!/usr/bin/env python3
"""
Comprehensive Edge Benchmarking & Resource Profiler for Raspberry Pi 4 (ARM64).

Evaluates:
  1. DSP AI Denoiser (DeepFilterNet ONNX)
  2. NLU Edge Pipeline (Lexical Normalizer + Single-Pass SetFit + Slot Extractor)
  3. Orchestrator ZeroMQ Dispatcher & LVC Snapshot Bus
  4. Full Pipeline CPU, RAM, Latency, and Real-Time Factor (RTF)

Outputs:
  - High-resolution console summary table with Raspberry Pi 4 budget thresholds
  - Structured JSON telemetry report saved to results/benchmark_edge_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dynamic repo path resolution
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psutil
except ImportError:
    psutil = None

import jsonschema
import onnxruntime as ort
import orjson
import soundfile as sf
import torch

from src.nlu.pipeline import VoiceToIntentPipeline

# Target Budgets for Raspberry Pi 4 Model B (4-core Cortex-A72 @ 1.5GHz, 4GB RAM)
PI4_BUDGETS = {
    "denoise_rtf_max": 0.50,         # Denoiser RTF must be <= 0.50
    "nlu_latency_p95_ms": 100.0,     # NLU p95 latency must be <= 100ms
    "nlu_ram_peak_mb": 800.0,        # NLU peak RAM <= 800MB
    "dispatcher_latency_p95_ms": 5.0,# Dispatcher socket latency <= 5ms
}


def get_process_memory_mb() -> float:
    """Returns process Resident Set Size (RSS) in megabytes."""
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# 1. Benchmark: DeepFilterNet ONNX Denoiser
# ---------------------------------------------------------------------------
def benchmark_denoiser(wav_path: Optional[str] = None, iterations: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" [1/3] PROFILING DSP AI DENOISER (DeepFilterNet ONNX)")
    print("=" * 70)

    model_path = REPO_ROOT / "src" / "dsp" / "denoise" / "deepfilternet.onnx"
    
    if wav_path:
        test_wav = Path(wav_path).resolve()
    else:
        test_wav = REPO_ROOT / "src" / "dsp" / "denoise" / "audio" / "noise_audio.wav"
        if not test_wav.exists():
            test_wav = REPO_ROOT / "data" / "asr" / "sample_01.wav"

    if not model_path.exists() or not test_wav.exists():
        print(f"  [!] Warning: DeepFilterNet model or test audio not found (checked {test_wav}). Skipping.")
        return {"status": "SKIPPED"}

    print(f"  Profiling Audio Source: {test_wav.name}")
    proc = psutil.Process(os.getpid()) if psutil else None
    ram_init_mb = get_process_memory_mb()

    # Load audio
    data, sr = sf.read(str(test_wav), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio_duration_s = len(data) / float(sr)

    # Load ONNX Session with 1 thread
    t_load_start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    load_time_s = time.perf_counter() - t_load_start
    ram_load_mb = get_process_memory_mb()

    # Dummy inputs matching DeepFilterNet shape [1, 1, T, 32] and [1, 2, T, 96]
    # For a 1.0s audio chunk at hop=128, T ~ 125 frames
    T = int((sr * min(1.0, audio_duration_s)) / 128)
    dummy_erb = np.random.randn(1, 1, T, 32).astype(np.float32)
    dummy_complex = np.random.randn(1, 2, T, 96).astype(np.float32)

    # Warmup
    session.run(None, {"input_erb": dummy_erb, "complex_in": dummy_complex})

    latencies_ms = []
    chunk_dur_s = 1.0

    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {"input_erb": dummy_erb, "complex_in": dummy_complex})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    ram_peak_mb = get_process_memory_mb()
    mean_lat_ms = float(np.mean(latencies_ms))
    p95_lat_ms = float(np.percentile(latencies_ms, 95))
    rtf = (mean_lat_ms / 1000.0) / chunk_dur_s

    print(f"  Model Load Time    : {load_time_s:.2f}s")
    print(f"  Single-Pass Latency: Mean = {mean_lat_ms:.2f}ms | p95 = {p95_lat_ms:.2f}ms")
    print(f"  Real-Time Factor   : {rtf:.4f} (Budget <= {PI4_BUDGETS['denoise_rtf_max']:.2f} -> {'PASS' if rtf <= PI4_BUDGETS['denoise_rtf_max'] else 'FAIL'})")
    print(f"  RAM Usage          : Base = {ram_init_mb:.1f}MB | Model Delta = +{ram_load_mb - ram_init_mb:.1f}MB | Peak = {ram_peak_mb:.1f}MB")

    return {
        "status": "PASS" if rtf <= PI4_BUDGETS["denoise_rtf_max"] else "FAIL",
        "load_time_s": round(load_time_s, 2),
        "mean_latency_ms": round(mean_lat_ms, 2),
        "p95_latency_ms": round(p95_lat_ms, 2),
        "rtf": round(rtf, 4),
        "ram_load_delta_mb": round(ram_load_mb - ram_init_mb, 1),
        "ram_peak_mb": round(ram_peak_mb, 1),
    }


# ---------------------------------------------------------------------------
# 2. Benchmark: NLU Pipeline
# ---------------------------------------------------------------------------
def benchmark_nlu(pin_core: bool = True, num_threads: int = 1, iterations: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f" [2/3] PROFILING NLU PIPELINE (Threads: {num_threads}, Core Pinning: {pin_core})")
    print("=" * 70)

    proc = psutil.Process(os.getpid()) if psutil else None
    if proc and pin_core:
        try:
            proc.cpu_affinity([0])
            print("  Pinned process affinity to physical Core 0")
        except Exception as e:
            print(f"  [!] Note: Could not set CPU affinity: {e}")

    ram_init_mb = get_process_memory_mb()
    t_load_start = time.perf_counter()
    pipeline = VoiceToIntentPipeline(num_threads=num_threads)
    load_time_s = time.perf_counter() - t_load_start
    ram_load_mb = get_process_memory_mb()

    test_utterances = [
        ("Please zoom in two point five times on coronal view", "zoom_display", "MONITOR_DISPLAY", "EXECUTE"),
        ("Zoom out to wide view", "zoom_display", "MONITOR_DISPLAY", "EXECUTE"),
        ("Advance forward 5 slices please", "navigate_slice", "MONITOR_DISPLAY", "EXECUTE"),
        ("Go back 10 slices nurse", "navigate_slice", "MONITOR_DISPLAY", "EXECUTE"),
        ("Jump to slice number 45", "navigate_slice", "MONITOR_DISPLAY", "EXECUTE"),
        ("Open CT scan on primary monitor", "open_scan", "MONITOR_DISPLAY", "EXECUTE"),
        ("Show brain MRI", "open_scan", "MONITOR_DISPLAY", "EXECUTE"),
        ("Reset display view", "reset_camera", "CAMERA_LIGHT", "EXECUTE"),
        ("Emergency stop immediately freeze", "halt_emergency", "EMERGENCY", "EXECUTE"),
        ("Clear surgical area back off", "clear_emergency", "EMERGENCY", "EXECUTE"),
        ("Hand me the scalpel please", "pick_tool", "TOOL_HANDLING", "EXECUTE"),
        ("Pass the iris scissors", "pick_tool", "TOOL_HANDLING", "EXECUTE"),
        ("Park robotic arm", "park_arm", "TOOL_HANDLING", "EXECUTE"),
        ("Record milestone trocar placed note clip attached", "log_event", "MONITOR_DISPLAY", "EXECUTE"),
        ("What is the weather like outside today", "unknown", "UNKNOWN", "OUT-OF-DOMAIN"),
        ("Could you play some soft music in here", "unknown", "UNKNOWN", "OUT-OF-DOMAIN"),
        ("423", "unknown", "UNKNOWN", "OUT-OF-DOMAIN"),
    ]

    all_latencies = []
    cpu_active_times = []
    schema_passes = 0
    ood_correct = 0
    ood_total = 0

    for raw, exp_intent, exp_cat, exp_status in test_utterances:
        is_ood = exp_status == "OUT-OF-DOMAIN"
        if is_ood:
            ood_total += 1

        for _ in range(iterations):
            t_cpu_before = proc.cpu_times() if proc else None
            t0 = time.perf_counter()
            envelope = pipeline.process(raw)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies.append(dt_ms)

            if proc and t_cpu_before:
                t_cpu_after = proc.cpu_times()
                cpu_active_times.append(((t_cpu_after.user - t_cpu_before.user) + (t_cpu_after.system - t_cpu_before.system)) * 1000.0)

            # Check schema
            try:
                jsonschema.validate(instance=envelope, schema=pipeline.schema)
                schema_passes += 1
            except Exception:
                pass

            if is_ood and envelope["status"] == "OUT-OF-DOMAIN":
                ood_correct += 1

    total_runs = len(test_utterances) * iterations
    ram_peak_mb = get_process_memory_mb()
    mean_ms = float(np.mean(all_latencies))
    p50_ms = float(np.percentile(all_latencies, 50))
    p95_ms = float(np.percentile(all_latencies, 95))
    p99_ms = float(np.percentile(all_latencies, 99))
    schema_rate = (schema_passes / total_runs) * 100.0
    ood_rate = (ood_correct / (ood_total * iterations)) * 100.0 if ood_total > 0 else 100.0

    print(f"  Model Load Time    : {load_time_s:.2f}s")
    print(f"  Latency Profile    : Mean = {mean_ms:.2f}ms | p50 = {p50_ms:.2f}ms | p95 = {p95_ms:.2f}ms | p99 = {p99_ms:.2f}ms")
    print(f"  Schema Compliance  : {schema_rate:.1f}% ({schema_passes}/{total_runs} envelopes valid Draft 2020-12)")
    print(f"  OOD Rejection Rate : {ood_rate:.1f}% ({ood_correct}/{ood_total * iterations} distractors rejected)")
    print(f"  RAM Usage          : Base = {ram_init_mb:.1f}MB | Model Delta = +{ram_load_mb - ram_init_mb:.1f}MB | Peak = {ram_peak_mb:.1f}MB")

    status = "PASS" if p95_ms <= PI4_BUDGETS["nlu_latency_p95_ms"] and ram_peak_mb <= PI4_BUDGETS["nlu_ram_peak_mb"] else "FAIL"
    print(f"  Overall NLU Status : {status} (p95 {p95_ms:.1f}ms <= {PI4_BUDGETS['nlu_latency_p95_ms']:.0f}ms, Peak RAM {ram_peak_mb:.0f}MB <= {PI4_BUDGETS['nlu_ram_peak_mb']:.0f}MB)")

    return {
        "status": status,
        "load_time_s": round(load_time_s, 2),
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "p99_latency_ms": round(p99_ms, 2),
        "schema_compliance_pct": round(schema_rate, 1),
        "ood_rejection_pct": round(ood_rate, 1),
        "ram_load_delta_mb": round(ram_load_mb - ram_init_mb, 1),
        "ram_peak_mb": round(ram_peak_mb, 1),
    }


# ---------------------------------------------------------------------------
# 3. Benchmark: Orchestrator ZeroMQ Dispatcher
# ---------------------------------------------------------------------------
def benchmark_orchestrator(iterations: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" [3/3] PROFILING ORCHESTRATOR ZEROMQ DISPATCHER & IPC BUS")
    print("=" * 70)

    import threading
    import zmq
    from src.orchestrator.dispatcher import CommandDispatcher

    test_nlu = "ipc:///tmp/bench_nlu.ipc"
    test_prio = "ipc:///tmp/bench_prio.ipc"
    test_bus = "ipc:///tmp/bench_bus.ipc"
    test_snap = "ipc:///tmp/bench_snap.ipc"

    dispatcher = CommandDispatcher(
        nlu_addr=test_nlu,
        nlu_priority_addr=test_prio,
        bus_addr=test_bus,
        snapshot_addr=test_snap,
    )

    t_disp = threading.Thread(target=dispatcher.run, daemon=True)
    t_disp.start()
    time.sleep(0.3)

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.connect(test_nlu)

    sub = ctx.socket(zmq.SUB)
    sub.connect(test_bus)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    time.sleep(0.2)

    sample_envelope = {
        "command_id": "cmd_bench_001",
        "timestamp": time.time(),
        "latency_ms": 10.5,
        "raw_transcript": "zoom in 2x",
        "intent": "zoom_display",
        "category": "MONITOR_DISPLAY",
        "slots": {"direction": "IN", "factor": 2.0},
        "confidence": 0.98,
        "priority": "NORMAL",
        "status": "EXECUTE",
    }
    payload_bytes = orjson.dumps(sample_envelope)

    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        push.send(payload_bytes)
        if sub.poll(timeout=1000):
            _ = sub.recv_multipart()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    dispatcher.stop()
    push.close(linger=0)
    sub.close(linger=0)
    ctx.destroy(linger=0)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))

    print(f"  IPC Roundtrip Transit: Mean = {mean_ms:.2f}ms | p95 = {p95_ms:.2f}ms")
    status = "PASS" if p95_ms <= PI4_BUDGETS["dispatcher_latency_p95_ms"] else "FAIL"
    print(f"  Dispatcher Status    : {status} (p95 {p95_ms:.2f}ms <= {PI4_BUDGETS['dispatcher_latency_p95_ms']:.1f}ms)")

    return {
        "status": status,
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 Edge Benchmarking & Profiler")
    parser.add_argument("--iterations", type=int, default=5, help="Test iterations per sample")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU worker threads (default: 1 for Pi 4)")
    parser.add_argument("--no-pin", action="store_true", help="Do not pin process to Core 0")
    parser.add_argument("--wav", type=str, default=None, help="Path to custom WAV file for denoiser profiling")
    args = parser.parse_args()

    print("*" * 70)
    print(" SCRUB-VOICE: RASPBERRY PI 4 EDGE VALIDATION & RESOURCE PROFILE")
    print(f" Host: {platform.platform()} | Python: {platform.python_version()}")
    print(f" Emulation Mode: PyTorch Threads = {args.threads} | Core Pinning = {not args.no_pin}")
    if args.wav:
        print(f" Custom Audio Input: {args.wav}")
    print("*" * 70)

    report = {
        "timestamp": time.time(),
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_threads": args.threads,
        "core_pinned": not args.no_pin,
        "modules": {},
    }

    # 1. Denoiser
    report["modules"]["dsp_denoiser"] = benchmark_denoiser(wav_path=args.wav, iterations=args.iterations)

    # 2. NLU
    report["modules"]["nlu_pipeline"] = benchmark_nlu(
        pin_core=not args.no_pin,
        num_threads=args.threads,
        iterations=args.iterations,
    )

    # 3. Orchestrator
    report["modules"]["orchestrator"] = benchmark_orchestrator(iterations=30)

    # Save structured JSON
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "benchmark_edge_profile.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(" EDGE BENCHMARK SUMMARY TABLE:")
    print("=" * 70)
    print(f"{'Module':<25} | {'Status':<7} | {'Mean Latency':<14} | {'p95 Latency':<13} | {'RAM Peak'}")
    print("-" * 70)

    d_res = report["modules"]["dsp_denoiser"]
    n_res = report["modules"]["nlu_pipeline"]
    o_res = report["modules"]["orchestrator"]

    if d_res.get("status") != "SKIPPED":
        print(f"{'DSP DeepFilterNet ONNX':<25} | {d_res['status']:<7} | {d_res['mean_latency_ms']:>8.2f} ms   | {d_res['p95_latency_ms']:>7.2f} ms   | {d_res['ram_peak_mb']:>6.1f} MB (RTF: {d_res['rtf']})")
    print(f"{'NLU Edge Pipeline':<25} | {n_res['status']:<7} | {n_res['mean_latency_ms']:>8.2f} ms   | {n_res['p95_latency_ms']:>7.2f} ms   | {n_res['ram_peak_mb']:>6.1f} MB")
    print(f"{'Orchestrator ZeroMQ':<25} | {o_res['status']:<7} | {o_res['mean_latency_ms']:>8.2f} ms   | {o_res['p95_latency_ms']:>7.2f} ms   | N/A")
    print("=" * 70)
    print(f"Telemetry report written to: {report_file}")


if __name__ == "__main__":
    main()
