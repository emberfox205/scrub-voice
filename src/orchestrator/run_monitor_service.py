"""Unified Orchestrator & Intraoperative Monitor Web Service.

Combines the Scrub-Voice central CommandDispatcher (ZeroMQ PULL/PUB/REP broker)
with the FastAPI WebSocket monitor server (clinical dashboard on http://0.0.0.0:8000).

Lifecycle:
1. Starts CommandDispatcher in a background worker thread.
2. Serves the FastAPI + WebSocket clinical dashboard via Uvicorn.
3. Handles SIGINT/SIGTERM gracefully, closing ZeroMQ sockets and HTTP server cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitor.server import create_app
from src.orchestrator.dispatcher import (
    DEFAULT_BUS_ADDR,
    DEFAULT_NLU_ADDR,
    DEFAULT_NLU_PRIORITY_ADDR,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SNAPSHOT_ADDR,
    CommandDispatcher,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MonitorService] %(message)s",
)
logger = logging.getLogger("orchestrator.service")


class MonitorService:
    """Manages the lifecycle of both CommandDispatcher and FastAPI Monitor Gateway."""

    def __init__(
        self,
        nlu_addr: str = DEFAULT_NLU_ADDR,
        nlu_priority_addr: str = DEFAULT_NLU_PRIORITY_ADDR,
        bus_addr: str = DEFAULT_BUS_ADDR,
        snapshot_addr: str = DEFAULT_SNAPSHOT_ADDR,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.nlu_addr = nlu_addr
        self.nlu_priority_addr = nlu_priority_addr
        self.bus_addr = bus_addr
        self.snapshot_addr = snapshot_addr
        self.schema_path = schema_path
        self.host = host
        self.port = port

        self.dispatcher = CommandDispatcher(
            nlu_addr=self.nlu_addr,
            nlu_priority_addr=self.nlu_priority_addr,
            bus_addr=self.bus_addr,
            snapshot_addr=self.snapshot_addr,
            schema_path=self.schema_path,
        )
        self.dispatcher_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None
        self._stopped = False

    def start(self) -> None:
        logger.info("Starting CommandDispatcher background thread...")
        self.dispatcher_thread = threading.Thread(
            target=self.dispatcher.run, daemon=True, name="CommandDispatcherThread"
        )
        self.dispatcher_thread.start()

        # Allow dispatcher sockets to bind
        time.sleep(0.3)

        logger.info(f"Starting FastAPI intraoperative monitor web server on http://{self.host}:{self.port}...")
        app = create_app(
            bus_addr=self.bus_addr,
            snapshot_addr=self.snapshot_addr,
            fetch_snapshot_on_startup=True,
        )

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stopping MonitorService...")

        if self.server:
            self.server.should_exit = True

        if self.dispatcher:
            self.dispatcher.stop()

        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=2.0)

        logger.info("MonitorService stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Unified Orchestrator & Intraoperative Monitor Web Service")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--nlu-addr", default=DEFAULT_NLU_ADDR, help="NLU normal channel ZMQ URI")
    parser.add_argument("--nlu-priority-addr", default=DEFAULT_NLU_PRIORITY_ADDR, help="NLU priority channel ZMQ URI")
    parser.add_argument("--bus-addr", default=DEFAULT_BUS_ADDR, help="Command bus ZMQ URI")
    parser.add_argument("--snapshot-addr", default=DEFAULT_SNAPSHOT_ADDR, help="State snapshot ZMQ URI")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to command_schema.json")
    args = parser.parse_args()

    service = MonitorService(
        nlu_addr=args.nlu_addr,
        nlu_priority_addr=args.nlu_priority_addr,
        bus_addr=args.bus_addr,
        snapshot_addr=args.snapshot_addr,
        schema_path=Path(args.schema),
        host=args.host,
        port=args.port,
    )

    def handle_sig(sig, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        service.start()
    finally:
        service.stop()


if __name__ == "__main__":
    main()

