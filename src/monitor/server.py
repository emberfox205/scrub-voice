"""
FastAPI + WebSocket Server for Scrub-Voice Intraoperative Monitor.
Serves the offline clinical dashboard, bridges ZeroMQ state to browser
clients via WebSockets, and provides REST status endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Set

import orjson
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Dynamic path resolution according to AGENTS.md
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitor.bridge import DEFAULT_BUS_ADDR, DEFAULT_SNAPSHOT_ADDR, MonitorBridge

STATIC_DIR = CURRENT_DIR / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MonitorServer] %(message)s",
)
logger = logging.getLogger("monitor.server")


class ConnectionManager:
    """
    Manages active WebSocket connections with dedicated per-client outbound queues.
    Guarantees:
    1. Single-writer invariant per WebSocket (prevents Starlette ASGI concurrent send crashes).
    2. Head-of-line blocking elimination: slow or lagging browser clients do not delay others.
    3. Thread-safe and task-safe lifecycle management.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        self.max_queue_size = max_queue_size
        self._queues: Dict[WebSocket, asyncio.Queue[str]] = {}

    def connect(self, ws: WebSocket) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue_size)
        self._queues[ws] = queue
        return queue

    def disconnect(self, ws: WebSocket) -> None:
        self._queues.pop(ws, None)

    def broadcast(self, message: str) -> None:
        """Pushes message to all client queues; drops oldest frame if a client queue is full."""
        for ws, queue in list(self._queues.items()):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("WebSocket queue full for client, dropping oldest frame (slow consumer)")
                try:
                    _ = queue.get_nowait()
                    queue.put_nowait(message)
                except Exception:
                    pass

    @property
    def active_connections(self) -> Set[WebSocket]:
        return set(self._queues.keys())

    def __len__(self) -> int:
        return len(self._queues)


def create_app(
    bus_addr: str = DEFAULT_BUS_ADDR,
    snapshot_addr: str = DEFAULT_SNAPSHOT_ADDR,
    fetch_snapshot_on_startup: bool = True,
) -> FastAPI:
    """Application factory for the Monitor web server and WebSocket bridge."""
    bridge = MonitorBridge(bus_addr=bus_addr, snapshot_addr=snapshot_addr)
    manager = ConnectionManager(max_queue_size=100)

    async def broadcast_event(topic: str, command: Dict[str, Any], state_dict: Dict[str, Any]) -> None:
        """Broadcasts state and command updates to all connected WebSockets via isolated queues."""
        if len(manager) == 0:
            return

        payload_str = orjson.dumps({
            "type": "command",
            "topic": topic,
            "command": command,
            "state": state_dict,
        }).decode("utf-8")

        manager.broadcast(payload_str)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        bridge.set_event_loop(loop)
        bridge.add_listener(broadcast_event)
        bridge.start(fetch_initial_snapshot=fetch_snapshot_on_startup)
        logger.info("Monitor server lifespan started.")
        yield
        bridge.remove_listener(broadcast_event)
        bridge.stop()
        logger.info("Monitor server lifespan ended.")

    app = FastAPI(title="Scrub-Voice Monitor Gateway", lifespan=lifespan)
    app.state.bridge = bridge
    app.state.manager = manager
    app.state.active_connections = manager.active_connections

    # HTTP middleware adds anti-caching headers for clinical dashboard and REST endpoints.
    # Note: Starlette HTTP middleware intercepts standard HTTP requests only; WebSockets bypass it.
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static") or request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return {"error": "Dashboard UI not found", "path": str(index_file)}
        return FileResponse(index_file)

    @app.get("/api/state")
    async def get_state():
        """Returns the current intraoperative monitor state."""
        return bridge.state.to_dict()

    @app.get("/api/health")
    async def health_check():
        """Returns monitor bridge connectivity and active subscriber counts."""
        return {
            "status": "healthy",
            "active_clients": len(manager),
            "emergency_active": bridge.state.emergency_active,
            "bus_addr": bridge.bus_addr,
            "snapshot_addr": bridge.snapshot_addr,
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        Full-duplex WebSocket channel for streaming real-time voice and state events.
        Uses a dedicated sender task per connection to strictly prevent concurrent write collisions.
        """
        await websocket.accept()
        queue = manager.connect(websocket)
        logger.info("WebSocket client connected. Total clients: %d", len(manager))

        # Enqueue full initial state snapshot as first message to guarantee correct ordering
        snapshot_msg = orjson.dumps({
            "type": "snapshot",
            "state": bridge.state.to_dict(),
        }).decode("utf-8")
        await queue.put(snapshot_msg)

        async def _sender_loop() -> None:
            """Dedicated sender coroutine ensures strictly sequential writes to the socket."""
            try:
                while True:
                    msg = await queue.get()
                    await websocket.send_text(msg)
                    queue.task_done()
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as send_err:
                logger.debug("WebSocket send error: %s", send_err)

        sender_task = asyncio.create_task(_sender_loop(), name="ws-sender")

        try:
            while True:
                data = await websocket.receive_text()
                # Respond to client heartbeats via outbound queue to prevent concurrent write collisions
                if data == "ping":
                    try:
                        queue.put_nowait("pong")
                    except asyncio.QueueFull:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("WebSocket receiver exception: %s", e)
        finally:
            sender_task.cancel()
            manager.disconnect(websocket)
            logger.info("WebSocket client disconnected. Remaining clients: %d", len(manager))

    return app


# Default application instance for uvicorn
app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Scrub-Voice Monitor Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/WebSocket port")
    parser.add_argument("--bus-addr", default=DEFAULT_BUS_ADDR, help="Command bus ZMQ URI")
    parser.add_argument("--snapshot-addr", default=DEFAULT_SNAPSHOT_ADDR, help="Snapshot server ZMQ URI")
    args = parser.parse_args()

    custom_app = create_app(bus_addr=args.bus_addr, snapshot_addr=args.snapshot_addr)
    uvicorn.run(custom_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
