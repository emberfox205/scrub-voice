"""
ZeroMQ to WebSocket Bridge for Scrub-Voice Monitor.
Connects to the Orchestrator's PUB bus and REQ snapshot server, maintains
the local MonitorState, and broadcasts updates to UI listeners.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import orjson
import zmq

from src.monitor.handlers import handle_command
from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.bridge")

DEFAULT_BUS_ADDR = os.getenv("SCRUB_BUS_ADDR", "ipc:///tmp/scrub_bus.ipc")
DEFAULT_SNAPSHOT_ADDR = os.getenv("SCRUB_SNAPSHOT_ADDR", "ipc:///tmp/scrub_snapshot.ipc")


class MonitorBridge:
    """
    Subscribes to Orchestrator's command bus, synchronizes state via snapshot server,
    and forwards events to registered asynchronous callbacks (e.g. WebSockets).
    """

    def __init__(
        self,
        bus_addr: str = DEFAULT_BUS_ADDR,
        snapshot_addr: str = DEFAULT_SNAPSHOT_ADDR,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.bus_addr = bus_addr
        self.snapshot_addr = snapshot_addr
        self.loop = loop
        self.state = MonitorState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str, Dict[str, Any], Dict[str, Any]], Any]] = []
        self._recent_cmd_ids: deque = deque(maxlen=100)
        self._context = zmq.Context()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Sets the asyncio event loop for threadsafe callback dispatching."""
        self.loop = loop

    def add_listener(self, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], Any]) -> None:
        """Registers a callback (topic, command_payload, state_dict)."""
        self._callbacks.append(callback)

    def remove_listener(self, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], Any]) -> None:
        """Unregisters a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def fetch_snapshot(self, timeout_ms: int = 1500) -> bool:
        """
        Queries the orchestrator snapshot server (REQ/REP) to hydrate the initial state.
        Returns True if successful, False if timed out or failed.
        """
        req_sock = self._context.socket(zmq.REQ)
        req_sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        req_sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        req_sock.setsockopt(zmq.LINGER, 0)

        try:
            req_sock.connect(self.snapshot_addr)
            req_sock.send(b"SNAPSHOT")
            raw_state = req_sock.recv()
            state_dict = orjson.loads(raw_state)
            self.state = MonitorState.from_dict(state_dict)
            logger.info("Synchronized initial state snapshot from %s", self.snapshot_addr)
            return True
        except Exception as err:
            logger.warning("Could not fetch state snapshot from %s: %s (using defaults)", self.snapshot_addr, err)
            return False
        finally:
            req_sock.close()

    def _sub_worker(self) -> None:
        """Worker thread for subscribing to ZeroMQ command bus."""
        sub_sock = self._context.socket(zmq.SUB)
        sub_sock.setsockopt(zmq.RCVHWM, 200)
        sub_sock.setsockopt(zmq.RCVTIMEO, 250)  # Allows regular check of self._running
        sub_sock.connect(self.bus_addr)
        sub_sock.setsockopt(zmq.SUBSCRIBE, b"")  # Listen to all topics

        logger.info("MonitorBridge subscriber connected to %s", self.bus_addr)

        while self._running:
            try:
                parts = sub_sock.recv_multipart()
                if len(parts) < 2:
                    continue
                topic = parts[0].decode("utf-8", errors="replace")
                payload = orjson.loads(parts[1])

                # Deduplicate multi-topic broadcasts of identical command_id
                cmd_id = payload.get("command_id")
                if cmd_id and cmd_id in self._recent_cmd_ids:
                    logger.debug("Skipping duplicate delivery for command %s on topic %s", cmd_id, topic)
                    continue
                if cmd_id:
                    self._recent_cmd_ids.append(cmd_id)

                # Update local state via intent handlers
                handle_command(self.state, payload)
                state_dict = self.state.to_dict()

                # Dispatch to all listeners
                self._dispatch_event(topic, payload, state_dict)
            except zmq.Again:
                continue
            except Exception as err:
                if self._running:
                    logger.error("Error processing ZMQ message: %s", err, exc_info=True)

        sub_sock.close(linger=0)
        logger.info("MonitorBridge subscriber loop terminated.")

    def _dispatch_event(self, topic: str, command: Dict[str, Any], state_dict: Dict[str, Any]) -> None:
        """Dispatches an event to registered callbacks across thread boundaries."""
        for cb in list(self._callbacks):
            try:
                if asyncio.iscoroutinefunction(cb):
                    if self.loop and self.loop.is_running():
                        fut = asyncio.run_coroutine_threadsafe(cb(topic, command, state_dict), self.loop)
                        fut.add_done_callback(lambda f: self._log_future_exception(f))
                else:
                    if self.loop and self.loop.is_running():
                        self.loop.call_soon_threadsafe(cb, topic, command, state_dict)
                    else:
                        cb(topic, command, state_dict)
            except Exception as e:
                logger.error("Callback dispatch error: %s", e)

    @staticmethod
    def _log_future_exception(f: Any) -> None:
        try:
            if not f.cancelled():
                exc = f.exception()
                if exc:
                    logger.error("Async callback raised error: %s", exc)
        except Exception:
            pass

    def start(self, fetch_initial_snapshot: bool = True) -> None:
        """Starts the bridge and subscription worker."""
        if self._running:
            return

        if fetch_initial_snapshot:
            self.fetch_snapshot()

        self._running = True
        self._thread = threading.Thread(target=self._sub_worker, name="MonitorBridgeWorker", daemon=True)
        self._thread.start()
        logger.info("MonitorBridge started successfully.")

    def stop(self) -> None:
        """Stops the subscription worker and terminates context."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            self._context.destroy(linger=0)
        except Exception:
            pass
        logger.info("MonitorBridge stopped.")
