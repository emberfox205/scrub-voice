"""ZeroMQ IPC and Dispatcher Benchmark.

Measures:
1. Transport latency & throughput: POSIX IPC, TCP loopback, and AF_UNIX socketpairs (256B, 4KB, 32KB).
2. Dispatcher round-trip latency (PUSH NLU -> Dispatcher -> PUB Command Bus).
3. Priority preemption: Emergency channel prioritization over backlogged normal commands.
4. High-Water-Mark (HWM) burst load: Backpressure handling under queue saturation.
5. Fuzzing / Malformed payload recovery: Deadlock immunity and error containment.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.common.registry import BUDGETS
from test.benchmarks.common.reporting import _jd
from test.benchmarks.common.rig import DispatcherRig, endpoint
from test.benchmarks.common.stats import summarize
from test.benchmarks.common.telemetry import ResourceSampler, get_rss_mb, hw_info

FALLBACK_NORMAL_ENVELOPE = {
    "command_id": "cmd_bench_normal",
    "timestamp": 0.0,
    "latency_ms": 15.0,
    "raw_transcript": "zoom in two times",
    "intent": "zoom_display",
    "category": "MONITOR_DISPLAY",
    "priority": "NORMAL",
    "status": "EXECUTE",
    "confidence": 0.95,
    "slots": {"direction": "IN", "factor": 2.0},
}

FALLBACK_EMERGENCY_ENVELOPE = {
    "command_id": "cmd_bench_emergency",
    "timestamp": 0.0,
    "latency_ms": 10.0,
    "raw_transcript": "emergency stop",
    "intent": "halt_emergency",
    "category": "EMERGENCY",
    "priority": "HIGH",
    "status": "EXECUTE",
    "confidence": 0.99,
    "slots": {},
}


def _transport_latency(kind: str, payload: int, n: int) -> Dict[str, Any]:
    """Measure one-way latency (sender timestamp in payload) and throughput."""
    body = b"x" * max(0, payload - 8)
    lats: List[float] = []

    if kind.startswith("zmq"):
        import zmq

        ctx = zmq.Context()
        pull, push = ctx.socket(zmq.PULL), ctx.socket(zmq.PUSH)
        if kind == "zmq_ipc":
            ep = endpoint("tp")
            pull.bind(ep)
        else:
            ep = f"tcp://127.0.0.1:{pull.bind_to_random_port('tcp://127.0.0.1')}"
        push.connect(ep)
        time.sleep(0.1)

        for _ in range(n):
            push.send(struct.pack("d", pc()) + body)
            m = pull.recv()
            lats.append((pc() - struct.unpack("d", m[:8])[0]) * 1000)

        t0 = pc()
        th = threading.Thread(target=lambda: [push.send(b"y" * payload) for _ in range(n * 4)])
        th.start()
        for _ in range(n * 4):
            pull.recv()
        thr = (n * 4) / (pc() - t0)
        th.join()

        push.close(linger=0)
        pull.close(linger=0)
        ctx.destroy(linger=0)

        if kind == "zmq_ipc" and ep.startswith("ipc://"):
            sock_p = ep[len("ipc://"):]
            if os.path.exists(sock_p):
                try:
                    os.unlink(sock_p)
                except OSError:
                    pass
    else:
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        for _ in range(n):
            a.sendall(struct.pack("d", pc()) + body)
            data = b.recv(payload + 16)
            lats.append((pc() - struct.unpack("d", data[:8])[0]) * 1000)

        t0 = pc()
        th = threading.Thread(target=lambda: [a.sendall(b"y" * payload) for _ in range(n * 4)])
        th.start()
        for _ in range(n * 4):
            b.recv(1 << 16)
        thr = (n * 4) / (pc() - t0)
        th.join()

        a.close()
        b.close()

    return {"latency_ms": summarize(lats), "throughput_msgs_per_s": thr}


def bench_ipc(args_or_reps: Any = 500, nlu_instance=None, reps: Optional[int] = None) -> Dict[str, Any]:
    """Execute the full IPC and Dispatcher benchmark suite."""
    if reps is not None:
        reps_val = reps
    elif hasattr(args_or_reps, "reps_ipc"):
        reps_val = args_or_reps.reps_ipc
    elif isinstance(args_or_reps, int):
        reps_val = args_or_reps
    else:
        reps_val = 500
    reps = reps_val

    print("\n" + "=" * 72 + "\n [IPC] ZeroMQ transport + dispatcher\n" + "=" * 72)
    try:
        import zmq
    except ImportError:
        return {"status": "SKIPPED", "reason": "pyzmq not installed"}

    out: Dict[str, Any] = {"transport": {}}
    kinds = ["zmq_ipc", "zmq_tcp", "unix_socketpair"] if os.name == "posix" else ["zmq_tcp"]

    for kind in kinds:
        for size in (256, 4096, 32768):
            r = _transport_latency(kind, size, reps)
            out["transport"][f"{kind}/{size}B"] = r
            print(
                f"  {kind:<16} {size:>6} B  p50 {r['latency_ms']['p50']:.3f}  p99 {r['latency_ms']['p99']:.3f} ms   "
                f"{r['throughput_msgs_per_s']:>9,.0f} msg/s"
            )

    ipc_key = "zmq_ipc/256B" if "zmq_ipc/256B" in out["transport"] else "zmq_tcp/256B"
    out["transport_status"] = (
        "PASS"
        if (
            out["transport"][ipc_key]["latency_ms"]["p95"] <= BUDGETS["ipc_ms_max"]
            and out["transport"][ipc_key]["throughput_msgs_per_s"] > 5000
        )
        else "FAIL"
    )
    print(f"  transport budget (p95<={BUDGETS['ipc_ms_max']} ms, >5k msg/s): {out['transport_status']}")

    if nlu_instance is not None:
        try:
            normal = dict(nlu_instance.process("zoom in two times"))
            emergency = dict(nlu_instance.process("emergency stop"))
        except Exception as e:
            print(f"  [warn] nlu instance failed ({e}), falling back to pre-validated envelopes.")
            normal = dict(FALLBACK_NORMAL_ENVELOPE)
            emergency = dict(FALLBACK_EMERGENCY_ENVELOPE)
    else:
        normal = dict(FALLBACK_NORMAL_ENVELOPE)
        emergency = dict(FALLBACK_EMERGENCY_ENVELOPE)

    try:
        rig = DispatcherRig("rt")
    except Exception as e:
        out["dispatcher"] = {"status": "SKIPPED", "reason": f"{type(e).__name__}: {e}"}
        print(f"  [skip] dispatcher: {e}")
        return out

    d: Dict[str, Any] = {}

    # 1. Roundtrip latency
    lat = [rig.roundtrip({**normal, "timestamp": time.time()}) for _ in range(reps)]
    valid_lat = [x for x in lat if x is not None]
    d["roundtrip_ms"] = summarize(valid_lat)
    d["roundtrip_timeouts"] = int(sum(x is None for x in lat))
    d["roundtrip_status"] = (
        "PASS" if d["roundtrip_ms"].get("p95", 1e9) <= BUDGETS["dispatch_ms_max"] else "FAIL"
    )
    print(
        f"  dispatcher round trip p50 {d['roundtrip_ms']['p50']:.3f}  p95 {d['roundtrip_ms']['p95']:.3f}  "
        f"p99 {d['roundtrip_ms']['p99']:.3f} ms  [{d['roundtrip_status']}]"
    )
    rig.close()

    # 2. Priority Preemption
    rig = DispatcherRig("pre", start=False)
    for i in range(45):
        rig.push.send(rig.orjson.dumps({**normal, "command_id": f"cmd_n{i}", "timestamp": time.time()}))
    rig.push_prio.send(rig.orjson.dumps({**emergency, "command_id": "cmd_EMG", "timestamp": time.time()}))
    time.sleep(0.3)
    t_start = pc()
    rig.start()
    order, t_emg = [], None
    deadline = pc() + 3
    while pc() < deadline and rig.sub.poll(200):
        parts = rig.sub.recv_multipart()
        body = parts[1] if len(parts) > 1 else b"{}"
        try:
            cid = rig.orjson.loads(body).get("command_id")
        except Exception:
            cid = None
        order.append(cid)
        if cid == "cmd_EMG" and t_emg is None:
            t_emg = (pc() - t_start) * 1000
    d["preemption"] = {
        "emergency_rank": order.index("cmd_EMG") if "cmd_EMG" in order else None,
        "published": len(order),
        "emergency_publish_ms_after_loop_start": t_emg,
        "note": "rank 0 = emergency published before every backlogged normal command",
    }
    d["preemption_status"] = "PASS" if d["preemption"]["emergency_rank"] == 0 else "FAIL"
    print(
        f"  priority preemption: emergency rank {d['preemption']['emergency_rank']} of {len(order)}  [{d['preemption_status']}]"
    )
    rig.close()

    # 3. HWM Burst
    rig = DispatcherRig("hwm")
    r0, sent, again = get_rss_mb(), 0, 0
    t0 = pc()
    for i in range(200):
        try:
            rig.push.send(
                rig.orjson.dumps({**normal, "command_id": f"cmd_b{i}", "timestamp": time.time()}), flags=zmq.NOBLOCK
            )
            sent += 1
        except zmq.Again:
            again += 1
    burst_s = pc() - t0
    got = 0
    while rig.sub.poll(500):
        rig.sub.recv_multipart()
        got += 1
    d["hwm_burst"] = {
        "attempted": 200,
        "sent": sent,
        "sender_side_drops": again,
        "published": got,
        "burst_duration_s": round(burst_s, 3),
        "rss_delta_mb": round(get_rss_mb() - r0, 1),
        "dispatcher_alive": rig.thread.is_alive(),
    }
    print(
        f"  HWM burst: sent {sent}/200, published {got}, sender drops {again}, dispatcher alive {rig.thread.is_alive()}"
    )

    # 4. Malformed Payloads & Deadlock Recovery
    bad = [
        b"",
        b"{",
        b"\xff\xfe\x00garbage",
        b"[]",
        b'{"intent": 5}',
        b'{"command_id": "x"}',
        b"null",
        b'{"a":' + b"1" * 5000 + b"}",
    ]
    rig.drain()
    for p in bad:
        rig.push.send(p)
    time.sleep(0.5)
    topics = []
    while rig.sub.poll(300):
        topics.append(rig.sub.recv_multipart()[0].decode(errors="replace"))
    ok_after = rig.roundtrip({**normal, "timestamp": time.time()}) is not None
    d["malformed"] = {
        "injected": len(bad),
        "bus_messages": len(topics),
        "ERROR_topic": topics.count("ERROR"),
        "non_error_leaks": len(topics) - topics.count("ERROR"),
        "alive": rig.thread.is_alive(),
        "valid_msg_after": ok_after,
    }
    d["malformed_status"] = (
        "PASS"
        if (rig.thread.is_alive() and ok_after and d["malformed"]["non_error_leaks"] == 0)
        else "FAIL"
    )
    print(
        f"  malformed payloads: ERROR-topic {d['malformed']['ERROR_topic']}/{len(bad)}, "
        f"leaks {d['malformed']['non_error_leaks']}, recovered {ok_after}  [{d['malformed_status']}]"
    )
    rig.close()

    out["dispatcher"] = d
    return out


def parse_args():
    p = argparse.ArgumentParser(description="ZeroMQ transport and dispatcher benchmarks")
    p.add_argument("--reps-ipc", type=int, default=500, help="Repetitions for latency and round-trip")
    p.add_argument("--out", default="", help="Optional output directory for JSON report")
    p.add_argument("--tag", default="", help="Optional tag for output directory")
    p.add_argument("--use-repo-nlu", action="store_true", help="Instantiate RepoNLU to generate envelopes")
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    print("*" * 72 + "\n SCRUB-VOICE BENCHMARK | Subsystem: IPC & Dispatcher\n" + "*" * 72)
    nlu = None
    if args.use_repo_nlu:
        try:
            from test.benchmarks.common.registry import RepoNLU

            nlu = RepoNLU(args.threads)
        except Exception as e:
            print(f"  [warn] Could not load RepoNLU ({e}), using default valid envelopes.")

    with ResourceSampler() as rs:
        results = bench_ipc(reps=args.reps_ipc, nlu_instance=nlu)
    results["resources"] = rs.summary()

    if args.out:
        outdir = Path(args.out) / (f"bench_ipc_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": time.time(),
            "hardware": hw_info(),
            "ipc": results,
        }
        with open(outdir / "report_ipc.json", "w") as f:
            json.dump(report, f, indent=2, default=_jd)
        print(f"\n Results saved to: {outdir / 'report_ipc.json'}")


if __name__ == "__main__":
    main()
