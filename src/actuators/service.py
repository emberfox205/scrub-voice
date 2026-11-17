#!/usr/bin/env python3
"""
ZeroMQ Actuator Service Daemon for Scrub-Voice.

Subscribes to Orchestrator's PUB command bus on the 'TOOL_HANDLING' topic,
maintains the persistent ActuatorContext HAL interfaces, and executes
validated surgical tool retrieval and parking missions via the actuator handler registry.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import orjson
except ImportError:
    import json
    class _OrjsonCompat:
        @staticmethod
        def dumps(obj, *args, **kwargs):
            return json.dumps(obj).encode("utf-8")
        @staticmethod
        def loads(obj, *args, **kwargs):
            return json.loads(obj)
    orjson = _OrjsonCompat()

import zmq

from src.actuators.context import ActuatorContext, OperationalMode, RobotState
from src.actuators.handlers import handle_actuator_command

logger = logging.getLogger("actuators.service")

DEFAULT_BUS_ADDR = os.getenv("SCRUB_BUS_ADDR", "ipc:///tmp/scrub_bus.ipc")


class ActuatorService:
    """
    Subscribes to Orchestrator command bus for TOOL_HANDLING topic,
    manages the persistent ActuatorContext, and dispatches incoming
    commands to the actuator handler registry.
    """

    def __init__(
        self,
        bus_addr: str = DEFAULT_BUS_ADDR,
        context: Optional[ActuatorContext] = None,
        robot_ip: Optional[str] = None,
        gripper_ip: Optional[str] = None,
        sim: bool = False,
        headless: Optional[bool] = None,
        waypoints_path: Optional[Path] = None,
        topic: str = "TOOL_HANDLING",
    ) -> None:
        self.bus_addr = bus_addr
        self.topic = topic
        self._owns_context = context is None
        self.context = context or ActuatorContext(
            robot_ip=robot_ip,
            gripper_ip=gripper_ip,
            sim=sim,
            headless=headless,
            waypoints_path=waypoints_path,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recent_cmd_ids: deque = deque(maxlen=100)
        self._zmq_context: Optional[zmq.Context] = None
        self._callbacks: List[Callable[[Dict[str, Any], Any], None]] = []
        self._callbacks_lock = threading.Lock()

    def add_result_listener(self, cb: Callable[[Dict[str, Any], Any], None]) -> None:
        """Adds a callback to receive (command, result) notifications upon completion."""
        with self._callbacks_lock:
            self._callbacks.append(cb)

    def remove_result_listener(self, cb: Callable[[Dict[str, Any], Any], None]) -> None:
        """Removes a previously added callback."""
        with self._callbacks_lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def _sub_worker(self) -> None:
        """Worker thread loop polling ZeroMQ bus and executing actuator handlers."""
        assert self._zmq_context is not None
        sub_sock = self._zmq_context.socket(zmq.SUB)
        sub_sock.rcvhwm = 200
        sub_sock.rcvtimeo = 250  # Allows periodic check of self._running
        sub_sock.connect(self.bus_addr)
        sub_sock.subscribe(self.topic)

        logger.info("ActuatorService subscriber connected to %s (topic: %s)", self.bus_addr, self.topic)

        while self._running:
            try:
                parts = sub_sock.recv_multipart()
                if len(parts) < 2:
                    continue
                recv_topic = parts[0].decode("utf-8", errors="replace")
                payload = orjson.loads(parts[1])

                cmd_id = payload.get("command_id")
                if cmd_id and cmd_id in self._recent_cmd_ids:
                    logger.debug("Skipping duplicate command %s on topic %s", cmd_id, recv_topic)
                    continue
                if cmd_id:
                    self._recent_cmd_ids.append(cmd_id)

                logger.info(
                    "ActuatorService received command [%s] on topic %s: %s",
                    cmd_id, recv_topic, payload.get("intent")
                )
                result = handle_actuator_command(self.context, payload)

                with self._callbacks_lock:
                    cbs = list(self._callbacks)
                for cb in cbs:
                    try:
                        cb(payload, result)
                    except Exception as e:
                        logger.error("Error in ActuatorService result listener: %s", e)

            except zmq.Again:
                continue
            except Exception as err:
                if self._running:
                    logger.error("Error processing ZMQ message in ActuatorService: %s", err, exc_info=True)

        sub_sock.close(linger=0)
        logger.info("ActuatorService subscriber loop terminated.")

    def start(self) -> None:
        """Starts the background subscriber thread."""
        if self._running:
            logger.warning("ActuatorService is already running.")
            return

        self._running = True
        self._zmq_context = zmq.Context()
        self._thread = threading.Thread(target=self._sub_worker, name="ActuatorServiceSub", daemon=True)
        self._thread.start()
        logger.info("ActuatorService started successfully.")

    def stop(self) -> None:
        """Stops the subscriber thread and optionally releases context resources."""
        if not self._running:
            return

        logger.info("Stopping ActuatorService...")
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._zmq_context and not self._zmq_context.closed:
            self._zmq_context.term()
            self._zmq_context = None

        if self._owns_context:
            self.context.close()

        logger.info("ActuatorService stopped.")

    def is_alive(self) -> bool:
        """Checks if the subscriber thread is running."""
        return bool(self._running and self._thread and self._thread.is_alive())
