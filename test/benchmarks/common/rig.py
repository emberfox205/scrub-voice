"""ZeroMQ dispatcher harness fixture for isolated and end-to-end benchmarks."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from time import perf_counter as pc
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PORT = [5900]


def endpoint(tag: str) -> str:
    """Generate an isolated endpoint URL (IPC on POSIX, TCP on Windows)."""
    if os.name == "posix":
        return f"ipc:///tmp/bench_{os.getpid()}_{tag}.ipc"
    _PORT[0] += 1
    return f"tcp://127.0.0.1:{_PORT[0]}"


class DispatcherRig:
    """Manages an in-process CommandDispatcher with push/sub testing sockets."""

    def __init__(self, tag: str, start: bool = True):
        import orjson
        import zmq
        from src.orchestrator.dispatcher import CommandDispatcher

        self.zmq, self.orjson = zmq, orjson
        self.tag = tag
        self.ad = {k: endpoint(f"{tag}_{k}") for k in ("nlu", "prio", "bus", "snap")}
        self.disp = CommandDispatcher(
            nlu_addr=self.ad["nlu"],
            nlu_priority_addr=self.ad["prio"],
            bus_addr=self.ad["bus"],
            snapshot_addr=self.ad["snap"],
        )
        self.ctx = zmq.Context()
        self.push = self.ctx.socket(zmq.PUSH)
        self.push_prio = self.ctx.socket(zmq.PUSH)
        self.sub = self.ctx.socket(zmq.SUB)

        self.push.connect(self.ad["nlu"])
        self.push_prio.connect(self.ad["prio"])
        self.sub.connect(self.ad["bus"])
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self.thread = threading.Thread(target=self.disp.run, daemon=True)
        self.started = False
        if start:
            self.start()

    def start(self):
        """Start the dispatcher background thread."""
        self.thread.start()
        self.started = True
        time.sleep(0.4)

    def roundtrip(self, env: Dict[str, Any], timeout_ms: int = 1000) -> Optional[float]:
        """Send an envelope through the dispatcher and wait for bus publish. Returns roundtrip ms."""
        t0 = pc()
        self.push.send(self.orjson.dumps(env))
        if self.sub.poll(timeout_ms):
            self.sub.recv_multipart()
            return (pc() - t0) * 1000.0
        return None

    def drain(self):
        """Drain any pending messages from the bus subscriber socket."""
        while self.sub.poll(50):
            self.sub.recv_multipart()

    def close(self):
        """Stop dispatcher thread, close sockets, and destroy ZeroMQ context."""
        try:
            self.disp.stop()
        except Exception:
            pass
        for s in (self.push, self.push_prio, self.sub):
            try:
                s.close(linger=0)
            except Exception:
                pass
        try:
            self.ctx.destroy(linger=0)
        except Exception:
            pass

        # Cleanup IPC socket files on POSIX
        if os.name == "posix":
            for addr in self.ad.values():
                if addr.startswith("ipc://"):
                    p = addr[len("ipc://"):]
                    if os.path.exists(p):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def fresh_env(nlu, text: str) -> Dict[str, Any]:
    """Helper to generate a timestamped envelope using an NLU instance."""
    env = dict(nlu.process(text))
    env["timestamp"] = time.time()
    return env


if __name__ == "__main__":
    print("[test] Verifying DispatcherRig...")
    with DispatcherRig("test_rig") as rig:
        test_payload = {
            "command_id": "cmd_test_001",
            "timestamp": time.time(),
            "latency_ms": 12.0,
            "raw_transcript": "zoom in please",
            "intent": "zoom_display",
            "category": "MONITOR_DISPLAY",
            "priority": "NORMAL",
            "status": "EXECUTE",
            "confidence": 0.98,
            "slots": {"direction": "IN", "factor": 2.0},
        }
        rt = rig.roundtrip(test_payload)
        print(f"  Roundtrip latency: {rt:.2f} ms")
        assert rt is not None, "Failed roundtrip"
    print("[test] DispatcherRig verified successfully.")
