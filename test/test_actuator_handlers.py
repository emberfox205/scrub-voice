#!/usr/bin/env python3
"""
Unit Tests for Actuator Context and Modular Intent Handlers.

Verifies state machine safety gating (IDLE/BUSY/FAULT), fault recovery via park_arm,
operational mode switching, and handler execution with mock HAL interfaces.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.actuators.context import ActuatorContext, OperationalMode, RobotState
from src.actuators.handlers import (
    ACTUATOR_HANDLERS,
    handle_actuator_command,
)
from src.actuators.pipeline import MissionResult

REPO_ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS_PATH = REPO_ROOT / "src" / "actuators" / "waypoints.json"


class TestActuatorContextAndHandlers(unittest.TestCase):
    def setUp(self):
        # Create mock HAL interfaces
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

    def test_initial_context_state(self):
        """Context should initialize in IDLE state and TOOL_HANDLING mode."""
        self.assertEqual(self.context.state, RobotState.IDLE)
        self.assertEqual(self.context.mode, OperationalMode.TOOL_HANDLING)
        self.assertIsNone(self.context.current_task)
        self.assertIsNone(self.context.held_tool)
        self.assertIsNone(self.context.last_error)

        status = self.context.get_status()
        self.assertEqual(status["state"], "IDLE")
        self.assertEqual(status["mode"], "TOOL_HANDLING")
        self.assertFalse(status["is_busy"])

    def test_busy_safety_gating(self):
        """Commands must be rejected while the robot is BUSY executing a task."""
        self.context.transition_to(RobotState.BUSY, task_name="pick_scalpel")
        self.assertTrue(self.context.get_status()["is_busy"])

        can_run, reason = self.context.can_execute("pick_tool")
        self.assertFalse(can_run)
        self.assertIn("BUSY", reason)

        # Dispatch through registry while BUSY should be safely rejected
        cmd = {
            "command_id": "cmd_test_001",
            "intent": "pick_tool",
            "category": "TOOL_HANDLING",
            "status": "EXECUTE",
            "slots": {"tool_identifier": "scalpel"},
        }
        res = handle_actuator_command(self.context, cmd)
        self.assertFalse(res.get("success", False))
        self.assertTrue(res.get("rejected", False))
        self.assertIn("BUSY", res.get("reason", ""))

    def test_mode_switching_interlock(self):
        """Mode switches are allowed when idle & empty, but rejected if an instrument is held."""
        # 1. Allowed when empty
        ok, msg = self.context.switch_mode(OperationalMode.CAMERA_LIGHT)
        self.assertTrue(ok)
        self.assertEqual(self.context.mode, OperationalMode.CAMERA_LIGHT)

        # In CAMERA_LIGHT mode, tool handling commands must be gated
        can_run, reason = self.context.can_execute("pick_tool")
        self.assertFalse(can_run)
        self.assertIn("CAMERA_LIGHT", reason)

        # Switch back to TOOL_HANDLING
        ok, _ = self.context.switch_mode(OperationalMode.TOOL_HANDLING)
        self.assertTrue(ok)

        # 2. Rejected when instrument held
        self.context.held_tool = "scalpel"
        ok, msg = self.context.switch_mode(OperationalMode.CAMERA_LIGHT)
        self.assertFalse(ok)
        self.assertIn("surgical tool 'scalpel' held", msg)
        self.assertEqual(self.context.mode, OperationalMode.TOOL_HANDLING)

    def test_fault_state_and_recovery_via_park_arm(self):
        """In FAULT state, pick_tool is rejected, but park_arm executes and recovers IDLE state."""
        self.context.transition_to(RobotState.FAULT, error="Grasp verification failed")
        self.assertEqual(self.context.state, RobotState.FAULT)
        self.assertEqual(self.context.last_error, "Grasp verification failed")

        # pick_tool must be rejected during FAULT
        can_run, reason = self.context.can_execute("pick_tool")
        self.assertFalse(can_run)
        self.assertIn("FAULT", reason)

        cmd_pick = {
            "command_id": "cmd_test_fault_pick",
            "intent": "pick_tool",
            "status": "EXECUTE",
            "slots": {"tool_identifier": "forceps"},
        }
        res_pick = handle_actuator_command(self.context, cmd_pick)
        self.assertFalse(res_pick["success"])
        self.assertTrue(res_pick["rejected"])
        self.assertIn("FAULT", res_pick["reason"])

        # park_arm must be allowed during FAULT for recovery
        can_park, park_reason = self.context.can_execute("park_arm")
        self.assertTrue(can_park)
        self.assertIsNone(park_reason)

        cmd_park = {
            "command_id": "cmd_test_fault_park",
            "intent": "park_arm",
            "status": "EXECUTE",
            "slots": {},
        }
        res_park = handle_actuator_command(self.context, cmd_park)
        self.assertTrue(res_park["success"])
        self.assertEqual(res_park["intent"], "park_arm")

        # Verify recovery to IDLE state
        self.assertEqual(self.context.state, RobotState.IDLE)
        self.assertIsNone(self.context.last_error)
        self.assertIsNone(self.context.held_tool)
        self.mock_rtde_c.moveJ.assert_called()
        self.mock_gripper.close.assert_called()

    def test_handle_command_status_validation(self):
        """Commands with status != EXECUTE must be safely rejected."""
        for status in ("AMBIGUOUS", "OUT-OF-DOMAIN", "ERROR"):
            cmd = {
                "command_id": f"cmd_test_{status}",
                "intent": "pick_tool",
                "status": status,
                "slots": {"tool_identifier": "scalpel"},
            }
            res = handle_actuator_command(self.context, cmd)
            self.assertFalse(res["success"])
            self.assertTrue(res["rejected"])
            self.assertIn(status, res["reason"])

    def test_handle_command_unregistered_intent(self):
        """Unregistered intents must be safely rejected without throwing."""
        cmd = {
            "command_id": "cmd_test_unknown",
            "intent": "unknown_action",
            "status": "EXECUTE",
            "slots": {},
        }
        res = handle_actuator_command(self.context, cmd)
        self.assertFalse(res["success"])
        self.assertTrue(res["rejected"])
        self.assertIn("Unregistered intent", res["reason"])

    def test_pick_tool_validation_slots(self):
        """pick_tool validates missing or invalid tool identifiers and normalizes aliases."""
        # 1. Missing slot
        cmd_missing = {
            "command_id": "cmd_pick_missing",
            "intent": "pick_tool",
            "status": "EXECUTE",
            "slots": {},
        }
        res_missing = handle_actuator_command(self.context, cmd_missing)
        self.assertFalse(res_missing["success"])
        self.assertIn("Missing required slot", res_missing["reason"])

        # 2. Unknown tool
        cmd_unknown = {
            "command_id": "cmd_pick_unknown",
            "intent": "pick_tool",
            "status": "EXECUTE",
            "slots": {"tool_identifier": "laser_pointer"},
        }
        res_unknown = handle_actuator_command(self.context, cmd_unknown)
        self.assertFalse(res_unknown["success"])
        self.assertIn("Unknown tool", res_unknown["reason"])

    @patch("src.actuators.handlers.tool_handlers.ToolRetrievalPipeline.execute")
    def test_pick_tool_successful_execution(self, mock_pipeline):
        """Successful pick_tool updates held_tool and restores state to IDLE."""
        mock_pipeline.return_value = MissionResult(
            success=True,
            target="scalpel",
            phase="completed",
            message="Mission executed successfully",
            grip_detected=True,
            handover_released=True,
            elapsed_time_s=1.25,
        )

        cmd = {
            "command_id": "cmd_pick_scalpel",
            "intent": "pick_tool",
            "status": "EXECUTE",
            "slots": {"tool_identifier": "knife"},  # Alias for scalpel
        }
        res = handle_actuator_command(self.context, cmd)
        self.assertTrue(res["success"])
        self.assertEqual(res["tool"], "scalpel")
        self.assertEqual(self.context.held_tool, "scalpel")
        self.assertEqual(self.context.state, RobotState.IDLE)
        mock_pipeline.assert_called_once()

    @patch("src.actuators.handlers.tool_handlers.ToolRetrievalPipeline.execute")
    def test_pick_tool_failure_enters_fault_state(self, mock_pipeline):
        """Failed pick_tool transitions context to FAULT state with error recorded."""
        mock_pipeline.return_value = MissionResult(
            success=False,
            target="scissors",
            phase="grasp_failed",
            message="Grasp verification failed: grip not detected",
            grip_detected=False,
            handover_released=False,
            elapsed_time_s=2.5,
        )

        cmd = {
            "command_id": "cmd_pick_fail",
            "intent": "pick_tool",
            "status": "EXECUTE",
            "slots": {"tool_identifier": "scissors"},
        }
        res = handle_actuator_command(self.context, cmd)
        self.assertFalse(res["success"])
        self.assertEqual(self.context.state, RobotState.FAULT)
        self.assertIn("Grasp verification failed", self.context.last_error)


if __name__ == "__main__":
    unittest.main()
