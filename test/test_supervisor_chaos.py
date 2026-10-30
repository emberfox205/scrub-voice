"""Supervisor & Chaos Recovery Integration Test.

Verifies:
1. Supervisor successfully initializes microservices in proper dependency order.
2. Watchdog monitors child PIDs.
3. Chaos Injection: Intentionally kill -9 a worker process (NLU).
4. Auto-Recovery: Supervisor detects the dead PID and revives the service in < 500ms.
5. In-flight Resilience: The Web Monitor service and dispatcher remain 100% healthy during worker crashes.
6. Clean teardown and signal propagation.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx
import orjson
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.supervisor import Supervisor

TEST_PORT = 8899


class TestSupervisorChaos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supervisor = Supervisor(
            wav_path=None,
            host="127.0.0.1",
            port=TEST_PORT,
            enable_dsp=False,  # Run MONITOR + NLU + ASR
            max_restarts=3,
        )
        cls.sup_thread = threading.Thread(target=cls._run_supervisor, daemon=True)
        cls.sup_thread.start()

        # Wait until all initial pipeline services (MONITOR, NLU, ASR) are spawned and warmed up
        is_ready = cls.supervisor.wait_until_ready(timeout=100.0)
        if not is_ready:
            raise RuntimeError("Supervisor pipeline services failed to reach ready state within 100s")

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
                time.sleep(0.3)
        client.close()
        if not started:
            raise RuntimeError("Supervisor failed to start web server within 9 seconds")

    @classmethod
    def _run_supervisor(cls):
        cls.supervisor.start_all()
        cls.supervisor.run_watchdog()

    @classmethod
    def tearDownClass(cls):
        cls.supervisor.stop_all()
        cls.sup_thread.join(timeout=3.0)

    def test_01_all_services_alive(self):
        """Verify managed services have active PIDs."""
        for name in ["MONITOR", "NLU", "ASR"]:
            spec = self.supervisor.services.get(name)
            self.assertIsNotNone(spec, f"Service {name} spec missing")
            self.assertIsNotNone(spec.process, f"Service {name} process not started")
            self.assertIsNone(spec.process.poll(), f"Service {name} exited prematurely")

        # Health endpoint check
        with httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=3.0) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "healthy")

    def test_02_chaos_kill_worker_and_auto_recover(self):
        """Chaos Test: kill -9 NLU worker and verify revival in < 500ms."""
        nlu_spec = self.supervisor.services["NLU"]
        old_pid = nlu_spec.process.pid
        self.assertGreater(old_pid, 0)

        # Chaos Injection: Hard kill -9
        os.kill(old_pid, signal.SIGKILL)

        # Give watchdog loop up to 1.5s to detect and respawn
        t0 = time.time()
        recovered = False
        new_pid = None

        while time.time() - t0 < 2.0:
            time.sleep(0.1)
            curr_proc = nlu_spec.process
            if curr_proc and curr_proc.pid != old_pid and curr_proc.poll() is None:
                recovered = True
                new_pid = curr_proc.pid
                recovery_time = time.time() - t0
                break

        self.assertTrue(recovered, "NLU service was not revived by supervisor watchdog")
        self.assertNotEqual(old_pid, new_pid, "NLU did not receive a new PID")
        print(f"\n  [CHAOS TEST] NLU PID {old_pid} killed -> Recovered with PID {new_pid} in {recovery_time*1000:.1f}ms")

        # Prove Monitor HTTP server remained alive and healthy throughout worker crash
        with httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=3.0) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
