"""Integration test for Unified Orchestrator & Intraoperative Monitor Web Service.
"""

import threading
import time
import unittest
from pathlib import Path

import httpx
import orjson
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.run_monitor_service import MonitorService

TEST_NLU_ADDR = "ipc:///tmp/test_mon_nlu.ipc"
TEST_NLU_PRIO_ADDR = "ipc:///tmp/test_mon_nlu_prio.ipc"
TEST_BUS_ADDR = "ipc:///tmp/test_mon_bus.ipc"
TEST_SNAPSHOT_ADDR = "ipc:///tmp/test_mon_snapshot.ipc"
TEST_PORT = 8877


class TestMonitorService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = MonitorService(
            nlu_addr=TEST_NLU_ADDR,
            nlu_priority_addr=TEST_NLU_PRIO_ADDR,
            bus_addr=TEST_BUS_ADDR,
            snapshot_addr=TEST_SNAPSHOT_ADDR,
            host="127.0.0.1",
            port=TEST_PORT,
        )
        cls.srv_thread = threading.Thread(target=cls.service.start, daemon=True)
        cls.srv_thread.start()

        # Wait for HTTP server to become responsive
        client = httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=5.0)
        started = False
        for _ in range(30):
            try:
                res = client.get("/api/health")
                if res.status_code == 200:
                    started = True
                    break
            except Exception:
                time.sleep(0.2)
        client.close()
        if not started:
            raise RuntimeError("Monitor service HTTP server failed to start within 6 seconds")

        cls.context = zmq.Context()
        cls.nlu_push = cls.context.socket(zmq.PUSH)
        cls.nlu_push.connect(TEST_NLU_ADDR)
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.nlu_push.close(linger=0)
        cls.context.term()
        cls.service.stop()
        cls.srv_thread.join(timeout=3.0)

    def test_01_health_and_state_api(self):
        with httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=3.0) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "healthy")

            res_state = client.get("/api/state")
            self.assertEqual(res_state.status_code, 200)
            state = res_state.json()
            self.assertIn("zoom_factor", state)
            self.assertEqual(state["zoom_factor"], 1.0)

    def test_02_nlu_command_updates_state(self):
        envelope = {
            "timestamp": time.time(),
            "command_id": "cmd_zoom_in_001",
            "intent": "zoom_display",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "IN", "factor": 2.0},
            "confidence": 0.96,
            "latency_ms": 14.5,
            "raw_transcript": "zoom in please",
            "status": "EXECUTE",
            "priority": "NORMAL",
        }
        self.nlu_push.send(orjson.dumps(envelope))
        time.sleep(0.3)

        with httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=3.0) as client:
            res = client.get("/api/state")
            self.assertEqual(res.status_code, 200)
            state = res.json()
            self.assertEqual(state["zoom_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
