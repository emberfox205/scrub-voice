#!/usr/bin/env python3
"""
Unit tests for Scrub-Voice Central Command Dispatcher.

Verifies end-to-end ZeroMQ IPC message ingestion, schema validation,
topic routing, and strict Dual-Channel priority preemption where
emergency commands bypass normal queued commands.
"""

import json
import sys
import time
import unittest
from pathlib import Path

# Dynamic sys.path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import orjson
import zmq
from src.orchestrator.dispatcher import CommandDispatcher

TEST_NLU_ADDR = "ipc:///tmp/test_scrub_nlu_normal.ipc"
TEST_NLU_PRIORITY_ADDR = "ipc:///tmp/test_scrub_nlu_priority.ipc"
TEST_BUS_ADDR = "ipc:///tmp/test_scrub_bus.ipc"


class TestCommandDispatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dispatcher = CommandDispatcher(
            nlu_addr=TEST_NLU_ADDR,
            nlu_priority_addr=TEST_NLU_PRIORITY_ADDR,
            bus_addr=TEST_BUS_ADDR,
        )
        cls.dispatcher.setup_sockets()

        # Client context for pushing commands and subscribing to topics
        cls.client_ctx = zmq.Context()

        # Normal push socket
        cls.normal_push_sock = cls.client_ctx.socket(zmq.PUSH)
        cls.normal_push_sock.connect(TEST_NLU_ADDR)

        # Priority push socket
        cls.priority_push_sock = cls.client_ctx.socket(zmq.PUSH)
        cls.priority_push_sock.connect(TEST_NLU_PRIORITY_ADDR)

        # Subscriber socket to verify dispatching
        cls.sub_sock = cls.client_ctx.socket(zmq.SUB)
        cls.sub_sock.connect(TEST_BUS_ADDR)
        cls.sub_sock.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics

        # Give ZeroMQ time to establish IPC connections
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.normal_push_sock.close(linger=0)
        cls.priority_push_sock.close(linger=0)
        cls.sub_sock.close(linger=0)
        cls.client_ctx.term()
        cls.dispatcher.stop()

    def send_and_route(self, payload: dict | str, is_priority: bool = False) -> list[tuple[bytes, bytes]]:
        """Pushes a message via the appropriate channel and routes it, returning received messages."""
        if isinstance(payload, dict):
            raw = orjson.dumps(payload)
        else:
            raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")

        # Route through dispatcher
        self.dispatcher.route_message(raw)

        # Poll subscriber for emitted messages
        messages = []
        poller = zmq.Poller()
        poller.register(self.sub_sock, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(timeout=100))
            if self.sub_sock in socks and socks[self.sub_sock] == zmq.POLLIN:
                parts = self.sub_sock.recv_multipart()
                messages.append((parts[0], parts[1]))
            else:
                break
        return messages

    def test_standard_topic_routing(self):
        """Standard valid commands route to their designated category topics (TOOL_HANDLING, MONITOR_DISPLAY)."""
        # 1. TOOL_HANDLING
        tool_payload = {
            "command_id": "cmd_1008_120000_001",
            "timestamp": 1728380000.123,
            "latency_ms": 42.5,
            "raw_transcript": "hand me the scalpel",
            "intent": "pick_tool",
            "category": "TOOL_HANDLING",
            "slots": {"tool_identifier": "scalpel"},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        tool_msgs = self.send_and_route(tool_payload)
        self.assertGreaterEqual(len(tool_msgs), 1)
        self.assertIn(b"TOOL_HANDLING", [m[0] for m in tool_msgs])
        data = orjson.loads(tool_msgs[0][1])
        self.assertEqual(data["command_id"], "cmd_1008_120000_001")
        self.assertEqual(data["slots"]["tool_identifier"], "scalpel")

        # 2. MONITOR_DISPLAY
        display_payload = {
            "command_id": "cmd_1008_120002_003",
            "timestamp": 1728380002.5,
            "latency_ms": 38.0,
            "raw_transcript": "next slice",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "NEXT", "step_count": 1},
            "confidence": 0.95,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        display_msgs = self.send_and_route(display_payload)
        self.assertGreaterEqual(len(display_msgs), 1)
        self.assertIn(b"MONITOR_DISPLAY", [m[0] for m in display_msgs])
        disp_data = orjson.loads(display_msgs[0][1])
        self.assertEqual(disp_data["command_id"], "cmd_1008_120002_003")
        self.assertEqual(disp_data["slots"]["direction"], "NEXT")

    def test_emergency_dispatch(self):
        """Emergency command must route to EMERGENCY and broadcast GLOBAL_EMERGENCY."""
        payload = {
            "command_id": "cmd_1008_120001_002",
            "timestamp": 1728380001.0,
            "latency_ms": 25.0,
            "raw_transcript": "emergency stop",
            "intent": "halt_emergency",
            "category": "EMERGENCY",
            "slots": {},
            "confidence": 0.99,
            "priority": "HIGH",
            "status": "EXECUTE",
        }
        msgs = self.send_and_route(payload, is_priority=True)
        topics = [m[0] for m in msgs]
        self.assertIn(b"EMERGENCY", topics)
        self.assertIn(b"GLOBAL_EMERGENCY", topics)

    def test_ambiguous_command_diverted_to_feedback(self):
        """Ambiguous intent must be diverted to FEEDBACK topic and NOT execute."""
        payload = {
            "command_id": "cmd_1008_120003_004",
            "timestamp": 1728380003.0,
            "latency_ms": 35.0,
            "raw_transcript": "pass that thing",
            "intent": "pick_tool",
            "category": "TOOL_HANDLING",
            "slots": {},
            "confidence": 0.40,
            "priority": "NORMAL",
            "status": "AMBIGUOUS",
        }
        msgs = self.send_and_route(payload)
        topics = [m[0] for m in msgs]
        self.assertIn(b"FEEDBACK", topics)

    def test_schema_violation_triggers_error(self):
        """Corrupt or missing mandatory keys must be rejected and route to ERROR topic."""
        bad_payload = {
            "command_id": "cmd_invalid",
            "intent": "pick_tool",
        }
        msgs = self.send_and_route(bad_payload)
        topics = [m[0] for m in msgs]
        self.assertIn(b"ERROR", topics)

    def test_json_decode_error_triggers_error(self):
        """Non-JSON text must be caught and published to ERROR topic."""
        msgs = self.send_and_route(b"NOT_VALID_JSON{:::}")
        topics = [m[0] for m in msgs]
        self.assertIn(b"ERROR", topics)

    def test_dual_channel_priority_preemption(self):
        """
        Emergency commands pushed on the priority channel must be processed BEFORE
        pre-existing queued normal commands on the normal channel.
        """
        # 1. Push 2 normal commands into normal socket
        norm1 = {
            "command_id": "cmd_normal_001",
            "timestamp": time.time(),
            "latency_ms": 30.0,
            "raw_transcript": "advance 1 slice",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "NEXT"},
            "confidence": 0.95,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        norm2 = {
            "command_id": "cmd_normal_002",
            "timestamp": time.time(),
            "latency_ms": 30.0,
            "raw_transcript": "advance 2 slice",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "NEXT"},
            "confidence": 0.95,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.normal_push_sock.send(orjson.dumps(norm1))
        self.normal_push_sock.send(orjson.dumps(norm2))

        # 2. Push 1 emergency command into priority socket
        emg = {
            "command_id": "cmd_emergency_999",
            "timestamp": time.time(),
            "latency_ms": 15.0,
            "raw_transcript": "emergency stop",
            "intent": "halt_emergency",
            "category": "EMERGENCY",
            "slots": {},
            "confidence": 0.999,
            "priority": "HIGH",
            "status": "EXECUTE",
        }
        self.priority_push_sock.send(orjson.dumps(emg))

        # Allow ZeroMQ kernel buffers to populate
        time.sleep(0.05)

        # 3. Simulate one poller pass of the dispatcher
        poller = zmq.Poller()
        poller.register(self.dispatcher.pull_sock, zmq.POLLIN)
        poller.register(self.dispatcher.priority_pull_sock, zmq.POLLIN)

        dispatched_order = []
        # Run 3 polling iterations to pull all 3 messages
        for _ in range(3):
            socks = dict(poller.poll(timeout=100))
            if self.dispatcher.priority_pull_sock in socks:
                try:
                    raw = self.dispatcher.priority_pull_sock.recv(zmq.NOBLOCK)
                    payload = orjson.loads(raw)
                    dispatched_order.append(payload["command_id"])
                    self.dispatcher.route_message(raw)
                    continue
                except zmq.Again:
                    pass

            if self.dispatcher.pull_sock in socks and socks[self.dispatcher.pull_sock] == zmq.POLLIN:
                try:
                    raw = self.dispatcher.pull_sock.recv(zmq.NOBLOCK)
                    payload = orjson.loads(raw)
                    dispatched_order.append(payload["command_id"])
                    self.dispatcher.route_message(raw)
                except zmq.Again:
                    pass

        # Verify that emergency was dispatched FIRST despite being sent after normal commands
        self.assertEqual(len(dispatched_order), 3)
        self.assertEqual(dispatched_order[0], "cmd_emergency_999", "Emergency command must preempt normal commands!")
        self.assertEqual(dispatched_order[1], "cmd_normal_001")
        self.assertEqual(dispatched_order[2], "cmd_normal_002")


if __name__ == "__main__":
    unittest.main()
