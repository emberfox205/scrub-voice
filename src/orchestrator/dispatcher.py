#!/usr/bin/env python3
"""
Scrub-Voice Central Command Dispatcher.

Ingests parsed voice commands from the NLU pipeline via ZeroMQ (PULL),
validates against schemas/command_schema.json, and publishes verified
commands to downstream subsystems (PUB) partitioned by category topics.

Maintains an in-memory Last Value Cache (LVC) of the current surgical state
and exposes a ZeroMQ REP snapshot server for newly connected or reconnecting monitors.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jsonschema
import orjson
import zmq

# Dynamic path resolution according to AGENTS.md
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitor.handlers import handle_command
from src.monitor.state import MonitorState

DEFAULT_SCHEMA_PATH = REPO_ROOT / "schemas" / "command_schema.json"
DEFAULT_NLU_ADDR = os.getenv("SCRUB_NLU_ADDR", "ipc:///tmp/scrub_nlu.ipc")
DEFAULT_NLU_PRIORITY_ADDR = os.getenv("SCRUB_NLU_PRIORITY_ADDR", "ipc:///tmp/scrub_nlu_priority.ipc")
DEFAULT_BUS_ADDR = os.getenv("SCRUB_BUS_ADDR", "ipc:///tmp/scrub_bus.ipc")
DEFAULT_SNAPSHOT_ADDR = os.getenv("SCRUB_SNAPSHOT_ADDR", "ipc:///tmp/scrub_snapshot.ipc")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Orchestrator] %(message)s",
)
logger = logging.getLogger("orchestrator")


def clean_ipc_socket(addr: str) -> None:
    """Removes stale Unix domain socket file if present."""
    if addr.startswith("ipc://"):
        sock_path = Path(addr[6:])
        sock_path.unlink(missing_ok=True)
        sock_path.parent.mkdir(parents=True, exist_ok=True)


class CommandDispatcher:
    """
    Central ZeroMQ message broker, schema gatekeeper, and LVC snapshot server.
    """

    def __init__(
        self,
        nlu_addr: str = DEFAULT_NLU_ADDR,
        nlu_priority_addr: str = DEFAULT_NLU_PRIORITY_ADDR,
        bus_addr: str = DEFAULT_BUS_ADDR,
        snapshot_addr: str = DEFAULT_SNAPSHOT_ADDR,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.nlu_addr = nlu_addr
        self.nlu_priority_addr = nlu_priority_addr
        self.bus_addr = bus_addr
        self.snapshot_addr = snapshot_addr
        self.schema_path = Path(schema_path)

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Command schema not found at {self.schema_path}")

        self.schema: Dict[str, Any] = orjson.loads(self.schema_path.read_bytes())
        self.validator = jsonschema.Draft202012Validator(self.schema)
        logger.info("Loaded schema from %s (backend: orjson)", self.schema_path)

        self.context = zmq.Context()
        self.pull_sock: Optional[zmq.Socket] = None
        self.priority_pull_sock: Optional[zmq.Socket] = None
        self.pub_sock: Optional[zmq.Socket] = None
        self.rep_sock: Optional[zmq.Socket] = None
        self._running = False
        self._stopped = False
        self._stop_lock = threading.Lock()

        # Intraoperative state cache (LVC - Last Value Cache)
        self.monitor_state = MonitorState()
        self.state: Dict[str, Any] = self.monitor_state.to_dict()

    def setup_sockets(self) -> None:
        """Initializes and binds ZeroMQ sockets for dual-channel routing and snapshot serving."""
        clean_ipc_socket(self.nlu_addr)
        clean_ipc_socket(self.nlu_priority_addr)
        clean_ipc_socket(self.bus_addr)
        clean_ipc_socket(self.snapshot_addr)

        # 1. Normal command channel (PULL)
        self.pull_sock = self.context.socket(zmq.PULL)
        self.pull_sock.setsockopt(zmq.RCVHWM, 50)
        self.pull_sock.bind(self.nlu_addr)
        logger.info("NLU normal channel bound to %s (PULL, HWM=50)", self.nlu_addr)

        # 2. High-priority / emergency channel (PULL)
        self.priority_pull_sock = self.context.socket(zmq.PULL)
        self.priority_pull_sock.setsockopt(zmq.RCVHWM, 50)
        self.priority_pull_sock.bind(self.nlu_priority_addr)
        logger.info("NLU priority channel bound to %s (PULL, HWM=50)", self.nlu_priority_addr)

        # 3. Downstream distribution bus (PUB)
        self.pub_sock = self.context.socket(zmq.PUB)
        self.pub_sock.setsockopt(zmq.SNDHWM, 100)
        self.pub_sock.bind(self.bus_addr)
        logger.info("Command bus bound to %s (PUB)", self.bus_addr)

        # 4. State Snapshot Server (REP)
        self.rep_sock = self.context.socket(zmq.REP)
        self.rep_sock.setsockopt(zmq.SNDTIMEO, 1000)
        self.rep_sock.setsockopt(zmq.RCVTIMEO, 1000)
        self.rep_sock.bind(self.snapshot_addr)
        logger.info("Snapshot server bound to %s (REP)", self.snapshot_addr)

    def get_state(self) -> Dict[str, Any]:
        """Returns a deep-copied snapshot of current state cache."""
        return copy.deepcopy(self.state)

    def update_state(self, payload: Dict[str, Any]) -> None:
        """Updates internal state cache based on incoming command using registered handlers."""
        handle_command(self.monitor_state, payload)
        self.state = self.monitor_state.to_dict()

    def validate_command(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates payload against schemas/command_schema.json.
        Returns (is_valid, error_message).
        """
        errors = sorted(self.validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            first = errors[0]
            return False, f"{first.message} (path: {list(first.path)})"
        return True, None

    def _send_error(self, err_type: str, details: str, cmd_id: str = "unknown") -> None:
        """Emits structured error frame to ERROR topic."""
        if self.pub_sock:
            err = orjson.dumps({
                "error": err_type,
                "command_id": cmd_id,
                "details": details,
                "timestamp": time.time(),
            })
            self.pub_sock.send_multipart([b"ERROR", err])

    def route_message(self, raw_msg: str | bytes) -> bool:
        """
        Parses, validates, updates state cache, and publishes a command payload to the appropriate topic.
        Accepts UTF-8 bytes or string and emits raw bytes to ZeroMQ.
        """
        if self.pub_sock is None:
            raise RuntimeError("Sockets not initialized. Call setup_sockets() first.")

        try:
            payload = orjson.loads(raw_msg)
        except Exception as err:
            logger.error("Failed to decode JSON payload: %s", err)
            self._send_error("JSON_DECODE_ERROR", str(err))
            return False

        cmd_id = payload.get("command_id", "cmd_unknown")
        intent = payload.get("intent", "unknown")
        category = payload.get("category", "UNKNOWN")
        status = payload.get("status", "UNKNOWN")
        priority = payload.get("priority", "NORMAL")

        is_valid, err_msg = self.validate_command(payload)
        if not is_valid:
            logger.warning("Rejected invalid command [%s]: %s", cmd_id, err_msg)
            self._send_error("SCHEMA_VALIDATION_ERROR", err_msg or "Invalid schema", cmd_id)
            return False

        # Update orchestrator state cache
        self.update_state(payload)

        payload_bytes = orjson.dumps(payload)

        if status == "EXECUTE":
            topic = category.encode("utf-8")
            self.pub_sock.send_multipart([topic, payload_bytes])
            logger.info("Dispatched [EXECUTE] cmd=%s intent=%s topic=%s", cmd_id, intent, category)

            if category == "EMERGENCY" or priority == "HIGH":
                self.pub_sock.send_multipart([b"GLOBAL_EMERGENCY", payload_bytes])
                logger.warning("Broadcast [GLOBAL_EMERGENCY] cmd=%s intent=%s", cmd_id, intent)
        else:
            self.pub_sock.send_multipart([b"FEEDBACK", payload_bytes])
            self.pub_sock.send_multipart([category.encode("utf-8"), payload_bytes])
            logger.info("Diverted [%s] cmd=%s to FEEDBACK topic", status, cmd_id)

        return True

    def run(self) -> None:
        """Main dispatcher event loop with strict priority scheduling and snapshot serving."""
        self.setup_sockets()
        self._running = True

        poller = zmq.Poller()
        if self.pull_sock:
            poller.register(self.pull_sock, zmq.POLLIN)
        if self.priority_pull_sock:
            poller.register(self.priority_pull_sock, zmq.POLLIN)
        if self.rep_sock:
            poller.register(self.rep_sock, zmq.POLLIN)

        logger.info("Dispatcher is active (Dual-Channel Priority Polling & Snapshot REP enabled)...")

        try:
            while self._running:
                socks = dict(poller.poll(timeout=100))

                # 1. High-Priority Channel: DRAIN ALL EMERGENCY MESSAGES FIRST
                while self._running and self.priority_pull_sock in socks:
                    try:
                        raw_msg = self.priority_pull_sock.recv(zmq.NOBLOCK)
                        logger.info("Ingested command from [PRIORITY-CHANNEL]")
                        self.route_message(raw_msg)
                    except zmq.Again:
                        break

                # 2. Normal Channel: Only read when priority queue has been drained
                if self._running and self.pull_sock in socks and socks[self.pull_sock] == zmq.POLLIN:
                    try:
                        raw_msg = self.pull_sock.recv(zmq.NOBLOCK)
                        self.route_message(raw_msg)
                    except zmq.Again:
                        pass

                # 3. Snapshot Requests (REQ/REP)
                if self._running and self.rep_sock in socks and socks[self.rep_sock] == zmq.POLLIN:
                    try:
                        _ = self.rep_sock.recv(zmq.NOBLOCK)
                        state_bytes = orjson.dumps(self.get_state())
                        self.rep_sock.send(state_bytes)
                        logger.debug("Served LVC state snapshot to client")
                    except zmq.Again:
                        pass
                    except zmq.ZMQError as e:
                        logger.warning("Snapshot REP error: %s (resetting socket)", e)
                        try:
                            poller.unregister(self.rep_sock)
                            self.rep_sock.close(linger=0)
                        except Exception:
                            pass
                        clean_ipc_socket(self.snapshot_addr)
                        self.rep_sock = self.context.socket(zmq.REP)
                        self.rep_sock.setsockopt(zmq.SNDTIMEO, 1000)
                        self.rep_sock.setsockopt(zmq.RCVTIMEO, 1000)
                        self.rep_sock.bind(self.snapshot_addr)
                        poller.register(self.rep_sock, zmq.POLLIN)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down dispatcher...")
        except zmq.ZMQError as err:
            if self._running:
                logger.error("ZeroMQ error in dispatcher loop: %s", err)
        finally:
            self.stop()

    def stop(self) -> None:
        """Closes sockets and context cleanly."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._running = False
            try:
                self.context.destroy(linger=0)
            except Exception:
                pass
            self.pull_sock = self.priority_pull_sock = self.pub_sock = self.rep_sock = None

            clean_ipc_socket(self.nlu_addr)
            clean_ipc_socket(self.nlu_priority_addr)
            clean_ipc_socket(self.bus_addr)
            clean_ipc_socket(self.snapshot_addr)
            logger.info("Dispatcher stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Scrub-Voice Central Command Dispatcher")
    parser.add_argument("--nlu-addr", default=DEFAULT_NLU_ADDR, help="NLU normal channel ZMQ URI")
    parser.add_argument("--nlu-priority-addr", default=DEFAULT_NLU_PRIORITY_ADDR, help="NLU priority channel ZMQ URI")
    parser.add_argument("--bus-addr", default=DEFAULT_BUS_ADDR, help="Command bus ZMQ URI")
    parser.add_argument("--snapshot-addr", default=DEFAULT_SNAPSHOT_ADDR, help="State snapshot ZMQ URI")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to command_schema.json")
    args = parser.parse_args()

    dispatcher = CommandDispatcher(
        nlu_addr=args.nlu_addr,
        nlu_priority_addr=args.nlu_priority_addr,
        bus_addr=args.bus_addr,
        snapshot_addr=args.snapshot_addr,
        schema_path=Path(args.schema),
    )

    def handle_sig(sig, frame):
        dispatcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    dispatcher.run()


if __name__ == "__main__":
    main()
