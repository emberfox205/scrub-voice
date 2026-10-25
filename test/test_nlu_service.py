#!/usr/bin/env python3
"""
Unit and integration tests for the standalone NLU ZeroMQ Service Daemon (src/nlu/run_nlu.py).
Verifies:
1. PULL ingestion on SCRUB_ASR_ADDR.
2. Dual-channel PUSH routing (NORMAL to SCRUB_NLU_ADDR vs HIGH to SCRUB_PRIO_ADDR).
3. 10-key JSON schema compliance.
4. Correct intent classification and slot extraction.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsonschema
import orjson
import zmq

from src.nlu.run_nlu import NLUService

TEST_ASR_ADDR = "ipc:///tmp/test_nlu_asr.ipc"
TEST_NLU_ADDR = "ipc:///tmp/test_nlu_normal.ipc"
TEST_PRIO_ADDR = "ipc:///tmp/test_nlu_prio.ipc"


class TestNLUService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = REPO_ROOT / "schemas" / "command_schema.json"
        with open(schema_path, "rb") as f:
            cls.schema = orjson.loads(f.read())
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

        # 1. Initialize NLU service daemon
        cls.service = NLUService(
            asr_addr=TEST_ASR_ADDR,
            nlu_addr=TEST_NLU_ADDR,
            priority_addr=TEST_PRIO_ADDR,
            num_threads=1,
        )

        # 2. Setup mock downstream listeners (Orchestrator simulation: PULL sockets)
        cls.context = zmq.Context()
        cls.normal_pull = cls.context.socket(zmq.PULL)
        cls.normal_pull.bind(TEST_NLU_ADDR)

        cls.prio_pull = cls.context.socket(zmq.PULL)
        cls.prio_pull.bind(TEST_PRIO_ADDR)

        # 3. Start NLU service in background thread
        cls.srv_thread = threading.Thread(target=cls.service.run, daemon=True, name="NLUServiceThread")
        cls.srv_thread.start()
        time.sleep(0.5)

        # 4. Setup mock upstream ASR sender (PUSH socket)
        cls.asr_push = cls.context.socket(zmq.PUSH)
        cls.asr_push.connect(TEST_ASR_ADDR)
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.service.stop()
        cls.srv_thread.join(timeout=2.0)
        cls.asr_push.close(linger=0)
        cls.normal_pull.close(linger=0)
        cls.prio_pull.close(linger=0)
        cls.context.destroy(linger=0)

    def test_01_normal_zoom_command_routing(self):
        """Verify normal command routes to normal PUSH channel with 10-key schema envelope."""
        # Simulated ASR JSON payload
        asr_payload = {
            "text": "please zoom in 2.5x on coronal view thank you",
            "confidence": 0.95,
        }
        self.asr_push.send(orjson.dumps(asr_payload))

        # Expect envelope on normal PULL socket
        self.assertTrue(self.normal_pull.poll(timeout=2000), "Timed out waiting for NLU response")
        raw_env = self.normal_pull.recv()
        envelope = orjson.loads(raw_env)

        # Validate schema
        self.validator.validate(instance=envelope)

        self.assertEqual(envelope["intent"], "zoom_display")
        self.assertEqual(envelope["category"], "MONITOR_DISPLAY")
        self.assertEqual(envelope["priority"], "NORMAL")
        self.assertEqual(envelope["status"], "EXECUTE")
        self.assertEqual(envelope["slots"]["factor"], 2.5)
        self.assertEqual(envelope["slots"]["direction"], "IN")

    def test_02_emergency_priority_routing(self):
        """Verify emergency command routes to priority PUSH channel with HIGH priority."""
        asr_payload = {
            "text": "emergency stop immediately freeze",
            "confidence": 0.98,
        }
        self.asr_push.send(orjson.dumps(asr_payload))

        # Expect envelope on priority PULL socket
        self.assertTrue(self.prio_pull.poll(timeout=2000), "Timed out waiting for Priority NLU response")
        raw_env = self.prio_pull.recv()
        envelope = orjson.loads(raw_env)

        # Validate schema
        self.validator.validate(instance=envelope)

        self.assertEqual(envelope["intent"], "halt_emergency")
        self.assertEqual(envelope["category"], "EMERGENCY")
        self.assertEqual(envelope["priority"], "HIGH")
        self.assertEqual(envelope["status"], "EXECUTE")

    def test_03_out_of_domain_rejection(self):
        """Verify non-command conversational chatter is safely categorized as OUT-OF-DOMAIN."""
        asr_payload = {
            "text": "the weather outside is really stormy today",
            "confidence": 0.90,
        }
        self.asr_push.send(orjson.dumps(asr_payload))

        self.assertTrue(self.normal_pull.poll(timeout=2000))
        envelope = orjson.loads(self.normal_pull.recv())

        self.validator.validate(instance=envelope)
        self.assertEqual(envelope["intent"], "unknown")
        self.assertEqual(envelope["status"], "OUT-OF-DOMAIN")


if __name__ == "__main__":
    unittest.main()
