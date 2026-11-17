#!/usr/bin/env python3
"""
Integration Tests for Actuator Service ZeroMQ Subscriber Daemon.

Verifies ZeroMQ bus connectivity, topic filtering (TOOL_HANDLING),
command deduplication, and dispatching to actuator handlers.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import zmq

from src.actuators.context import ActuatorContext
from src.actuators.service import ActuatorService

REPO_ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS_PATH = REPO_ROOT / "src" / "actuators" / "waypoints.json"
TEST_BUS_ADDR = "ipc:///tmp/test_actuator_bus.ipc"


class TestActuatorService(unittest.TestCase):
    def setUp(self):
        # Clean previous socket if existing
        for p in Path("/tmp").glob("test_actuator_bus*.ipc"):
            p.unlink(missing_ok=True)

        self.zmq_context = zmq.Context()
        self.pub_sock = self.zmq_context.socket(zmq.PUB)
        self.pub_sock.bind(TEST_BUS_ADDR)

        # Mock HAL context
        self.mock_rtde_c = MagicMock()
        self.mock_rtde_r = MagicMock()
        self.mock_gripper = MagicMock()
        self.context = ActuatorContext(
            waypoints_path=WAYPOINTS_PATH,
            rtde_c=self.mock_rtde_c,
            rtde_r=self.mock_rtde_r,
            gripper=self.mock_gripper,
        )

        self.service = ActuatorService(
            bus_addr=TEST_BUS_ADDR,
            context=self.context,
            topic="TOOL_HANDLING",
        )
        self.service.start()
        # Allow subscription to connect and stabilize
        time.sleep(0.3)

    def tearDown(self):
        self.service.stop()
        self.pub_sock.close(linger=0)
        self.zmq_context.term()
        for p in Path("/tmp").glob("test_actuator_bus*.ipc"):
            p.unlink(missing_ok=True)

    def test_service_lifecycle_and_thread(self):
        """Service starts worker thread and reports alive status."""
        self.assertTrue(self.service.is_alive())

    def test_message_ingestion_and_dispatch(self):
        """Publishes a TOOL_HANDLING command and verifies delivery to result listener."""
        received_events = []

        def listener(cmd, result):
            received_events.append((cmd, result))

        self.service.add_result_listener(listener)

        payload = {
            "command_id": "cmd_service_test_001",
            "timestamp": time.time(),
            "intent": "park_arm",
            "category": "TOOL_HANDLING",
            "status": "EXECUTE",
            "slots": {},
        }
        self.pub_sock.send_multipart([b"TOOL_HANDLING", orjson.dumps(payload)])

        # Wait for dispatch
        for _ in range(30):
            if received_events:
                break
            time.sleep(0.05)

        self.assertEqual(len(received_events), 1)
        cmd, res = received_events[0]
        self.assertEqual(cmd["command_id"], "cmd_service_test_001")
        self.assertTrue(res.get("success", False))
        self.assertEqual(res.get("intent"), "park_arm")

    def test_command_deduplication(self):
        """Duplicate deliveries with identical command_id are processed only once."""
        received_events = []

        def listener(cmd, result):
            received_events.append((cmd, result))

        self.service.add_result_listener(listener)

        payload = {
            "command_id": "cmd_duplicate_001",
            "timestamp": time.time(),
            "intent": "park_arm",
            "category": "TOOL_HANDLING",
            "status": "EXECUTE",
            "slots": {},
        }
        # Send twice in rapid succession
        self.pub_sock.send_multipart([b"TOOL_HANDLING", orjson.dumps(payload)])
        time.sleep(0.05)
        self.pub_sock.send_multipart([b"TOOL_HANDLING", orjson.dumps(payload)])

        time.sleep(0.3)
        self.assertEqual(len(received_events), 1)

    def test_topic_filtering_ignores_unsubscribed_topics(self):
        """Messages published on other topics (e.g. MONITOR_DISPLAY) are ignored."""
        received_events = []

        def listener(cmd, result):
            received_events.append((cmd, result))

        self.service.add_result_listener(listener)

        payload = {
            "command_id": "cmd_monitor_001",
            "timestamp": time.time(),
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "status": "EXECUTE",
            "slots": {"scan_type": "ct"},
        }
        self.pub_sock.send_multipart([b"MONITOR_DISPLAY", orjson.dumps(payload)])

        time.sleep(0.3)
        self.assertEqual(len(received_events), 0)


if __name__ == "__main__":
    unittest.main()
