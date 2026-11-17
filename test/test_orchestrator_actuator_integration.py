#!/usr/bin/env python3
"""
End-to-End Orchestrator-to-Actuator Integration Test.

Verifies the entire execution flow:
NLU PUSH -> CommandDispatcher (schema validation & bus routing) -> ActuatorService (SUB)
-> ActuatorContext & Intent Handlers.
Also verifies successive command execution with persistent connectivity.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import zmq

from src.actuators.context import ActuatorContext, RobotState
from src.actuators.pipeline import MissionResult
from src.actuators.service import ActuatorService
from src.orchestrator.dispatcher import CommandDispatcher

REPO_ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS_PATH = REPO_ROOT / "src" / "actuators" / "waypoints.json"

TEST_NLU_ADDR = "ipc:///tmp/test_integ_nlu.ipc"
TEST_PRIO_ADDR = "ipc:///tmp/test_integ_prio.ipc"
TEST_BUS_ADDR = "ipc:///tmp/test_integ_bus.ipc"
TEST_SNAP_ADDR = "ipc:///tmp/test_integ_snap.ipc"


class TestOrchestratorActuatorIntegration(unittest.TestCase):
    def setUp(self):
        # Clean IPC sockets
        for p in Path("/tmp").glob("test_integ_*.ipc"):
            p.unlink(missing_ok=True)

        # 1. Initialize CommandDispatcher
        self.dispatcher = CommandDispatcher(
            nlu_addr=TEST_NLU_ADDR,
            nlu_priority_addr=TEST_PRIO_ADDR,
            bus_addr=TEST_BUS_ADDR,
            snapshot_addr=TEST_SNAP_ADDR,
        )
        self.dispatcher_thread = threading.Thread(target=self.dispatcher.run, daemon=True)
        self.dispatcher_thread.start()

        # 2. Client PUSH socket simulating NLU output
        self.client_ctx = zmq.Context()
        self.nlu_push = self.client_ctx.socket(zmq.PUSH)
        self.nlu_push.connect(TEST_NLU_ADDR)

        # 3. Persistent mock HAL
        self.mock_rtde_c = MagicMock()
        self.mock_rtde_r = MagicMock()
        self.mock_gripper = MagicMock()
        self.mock_gripper.close.return_value = {"grip_detected": True, "width_mm": 5.0}

        self.context = ActuatorContext(
            waypoints_path=WAYPOINTS_PATH,
            rtde_c=self.mock_rtde_c,
            rtde_r=self.mock_rtde_r,
            gripper=self.mock_gripper,
        )

        # 4. ActuatorService listening to bus
        self.actuator_service = ActuatorService(
            bus_addr=TEST_BUS_ADDR,
            context=self.context,
            topic="TOOL_HANDLING",
        )
        self.actuator_service.start()

        # Allow ZeroMQ handshakes
        time.sleep(0.3)

    def tearDown(self):
        self.actuator_service.stop()
        self.nlu_push.close(linger=0)
        self.client_ctx.term()
        self.dispatcher.stop()
        if self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=2.0)
        for p in Path("/tmp").glob("test_integ_*.ipc"):
            p.unlink(missing_ok=True)

    @patch("src.actuators.handlers.tool_handlers.ToolRetrievalPipeline.execute")
    def test_end_to_end_pick_tool_and_successive_park(self, mock_pipeline):
        """
        Verifies end-to-end command flow from NLU socket through Dispatcher to Actuator,
        and verifies successive command execution on the persistent connection.
        """
        mock_pipeline.return_value = MissionResult(
            success=True,
            target="scalpel",
            phase="completed",
            message="Tool retrieval completed",
            grip_detected=True,
            handover_released=True,
            elapsed_time_s=0.8,
        )

        results = []

        def on_result(cmd, res):
            results.append((cmd, res))

        self.actuator_service.add_result_listener(on_result)

        # Step 1: Send 'pick_tool' command via NLU socket
        cmd1 = {
            "command_id": "cmd_integ_001",
            "timestamp": round(time.time(), 3),
            "latency_ms": 15.0,
            "raw_transcript": "hand me the scalpel",
            "intent": "pick_tool",
            "category": "TOOL_HANDLING",
            "slots": {"tool_identifier": "scalpel"},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.nlu_push.send(orjson.dumps(cmd1))

        # Wait for Step 1 completion
        for _ in range(40):
            if len(results) >= 1:
                break
            time.sleep(0.05)

        self.assertEqual(len(results), 1)
        res1 = results[0][1]
        self.assertTrue(res1["success"])
        self.assertEqual(res1["tool"], "scalpel")
        self.assertEqual(self.context.state, RobotState.IDLE)
        self.assertEqual(self.context.held_tool, "scalpel")

        # Step 2: Send successive 'park_arm' command without reconnecting
        t_start_cmd2 = time.perf_counter()
        cmd2 = {
            "command_id": "cmd_integ_002",
            "timestamp": round(time.time(), 3),
            "latency_ms": 10.0,
            "raw_transcript": "park arm",
            "intent": "park_arm",
            "category": "TOOL_HANDLING",
            "slots": {},
            "confidence": 0.99,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.nlu_push.send(orjson.dumps(cmd2))

        # Wait for Step 2 completion
        for _ in range(40):
            if len(results) >= 2:
                break
            time.sleep(0.05)

        t_elapsed_cmd2 = time.perf_counter() - t_start_cmd2

        self.assertEqual(len(results), 2)
        res2 = results[1][1]
        self.assertTrue(res2["success"])
        self.assertEqual(res2["intent"], "park_arm")
        self.assertEqual(self.context.state, RobotState.IDLE)
        self.assertIsNone(self.context.held_tool)

        # Successive command should be received and executed in well under 1 second (< 200ms)
        self.assertLess(t_elapsed_cmd2, 1.0)
        self.mock_rtde_c.moveJ.assert_called()


if __name__ == "__main__":
    unittest.main()
