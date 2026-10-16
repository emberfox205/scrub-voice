#!/usr/bin/env python3
"""
Unit tests for Scrub-Voice Monitor Modular Intent Handlers.
Verifies state transitions, parameter validation, slot clamping,
safety interlocks, and schema compliance.
"""

import sys
import unittest
from pathlib import Path

# Dynamic sys.path setup according to AGENTS.md
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsonschema
import orjson

from src.monitor.state import MonitorState
from src.monitor.handlers import handle_command, HANDLERS


class TestMonitorHandlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = REPO_ROOT / "schemas" / "command_schema.json"
        with open(schema_path, "rb") as f:
            cls.schema = orjson.loads(f.read())
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def setUp(self):
        self.state = MonitorState()

    def test_handler_registration(self):
        """Verify that all 9 canonical intents are registered in the registry."""
        expected_intents = [
            "open_scan",
            "close_scan",
            "zoom_display",
            "navigate_slice",
            "log_event",
            "halt_emergency",
            "clear_emergency",
            "pick_tool",
            "park_arm",
        ]
        for intent in expected_intents:
            self.assertIn(intent, HANDLERS)

    # --- Scan Handlers ---

    def test_switch_scan_modality_schema_compliance(self):
        """
        Switching active diagnostic scans across modalities (CT -> MRI -> X-Ray)
        must strictly adhere to schemas/command_schema.json.
        Each switch must reset viewport slice to 1 and zoom to 1.0x.
        """
        # 1. Open CT scan
        ct_cmd = {
            "command_id": "cmd_1011_120101_001",
            "timestamp": 1728648001.0,
            "latency_ms": 25.4,
            "raw_transcript": "open CT scan",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "ct"},
            "confidence": 0.99,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(ct_cmd)
        state = handle_command(self.state, ct_cmd)
        self.assertEqual(state.active_scan, "ct")
        self.assertEqual(state.current_slice, 1)
        self.assertEqual(state.zoom_factor, 1.0)

        # Mutate viewport state (clinician zoomed and paged through CT slices)
        state.current_slice = 55
        state.zoom_factor = 3.5

        # 2. Switch CT -> MRI scan
        mri_cmd = {
            "command_id": "cmd_1011_120102_002",
            "timestamp": 1728648006.0,
            "latency_ms": 22.1,
            "raw_transcript": "switch to MRI",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "mri"},
            "confidence": 0.985,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(mri_cmd)
        state = handle_command(state, mri_cmd)
        self.assertEqual(state.active_scan, "mri")
        self.assertEqual(state.current_slice, 1)
        self.assertEqual(state.zoom_factor, 1.0)

        # Mutate viewport state again
        state.current_slice = 18
        state.zoom_factor = 2.0

        # 3. Switch MRI -> X-Ray scan
        xray_cmd = {
            "command_id": "cmd_1011_120103_003",
            "timestamp": 1728648011.0,
            "latency_ms": 20.8,
            "raw_transcript": "display X-ray",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "xray"},
            "confidence": 0.975,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(xray_cmd)
        state = handle_command(state, xray_cmd)
        self.assertEqual(state.active_scan, "xray")
        self.assertEqual(state.current_slice, 1)
        self.assertEqual(state.zoom_factor, 1.0)

        # 4. Invalid or None modality must fail schema and leave active scan unaltered
        invalid_cmd = {
            "command_id": "cmd_1011_120104_004",
            "timestamp": 1728648015.0,
            "latency_ms": 19.0,
            "raw_transcript": "open ultrasound",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "ultrasound"},
            "confidence": 0.90,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(invalid_cmd)

        state = handle_command(state, invalid_cmd)
        self.assertEqual(state.active_scan, "xray")

        # None slot rejection
        none_cmd = {
            "intent": "open_scan",
            "status": "EXECUTE",
            "slots": {"scan_type": None},
        }
        state = handle_command(state, none_cmd)
        self.assertEqual(state.active_scan, "xray")

    def test_close_scan_behavior(self):
        """close_scan closes matching scan or any active scan if slot is omitted."""
        # 1. Matching modality closes active scan
        self.state.active_scan = "mri"
        self.state.current_slice = 15
        self.state.zoom_factor = 2.5
        state = handle_command(self.state, {"intent": "close_scan", "status": "EXECUTE", "slots": {"scan_type": "mri"}})
        self.assertIsNone(state.active_scan)
        self.assertEqual(state.current_slice, 1)
        self.assertEqual(state.zoom_factor, 1.0)

        # 2. Mismatched modality does NOT close active scan
        state.active_scan = "ct"
        state = handle_command(state, {"intent": "close_scan", "status": "EXECUTE", "slots": {"scan_type": "mri"}})
        self.assertEqual(state.active_scan, "ct")

        # 3. Omitted or empty modality closes active scan
        state = handle_command(state, {"intent": "close_scan", "status": "EXECUTE", "slots": {}})
        self.assertIsNone(state.active_scan)

    def test_zoom_display_and_clamping(self):
        """zoom_display multiplies on IN, divides on OUT, defaults to 2.0x, and clamps [1.0, 10.0]."""
        # Default zoom (factor omitted defaults to 2.0x IN)
        state = handle_command(self.state, {"intent": "zoom_display", "status": "EXECUTE", "slots": {}})
        self.assertEqual(state.zoom_factor, 2.0)

        # Zoom in by 1.5x -> 3.0x
        state = handle_command(state, {"intent": "zoom_display", "status": "EXECUTE", "slots": {"direction": "IN", "factor": 1.5}})
        self.assertEqual(state.zoom_factor, 3.0)

        # Zoom out by 2.0x -> 1.5x
        state = handle_command(state, {"intent": "zoom_display", "status": "EXECUTE", "slots": {"direction": "OUT", "factor": 2.0}})
        self.assertEqual(state.zoom_factor, 1.5)

        # Max clamping at 10.0x
        state = handle_command(state, {"intent": "zoom_display", "status": "EXECUTE", "slots": {"direction": "IN", "factor": 100.0}})
        self.assertEqual(state.zoom_factor, 10.0)

        # Min clamping at 1.0x
        state = handle_command(state, {"intent": "zoom_display", "status": "EXECUTE", "slots": {"direction": "OUT", "factor": 100.0}})
        self.assertEqual(state.zoom_factor, 1.0)

    def test_navigate_slice_direct_and_relative(self):
        """navigate_slice handles direct index, relative NEXT/PREVIOUS, and minimum clamping to 1."""
        # Direct navigation
        state = handle_command(self.state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"slice_index": 45}})
        self.assertEqual(state.current_slice, 45)

        # Clamping direct negative index to 1
        state = handle_command(state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"slice_index": -10}})
        self.assertEqual(state.current_slice, 1)

        # Relative navigation: NEXT + 10 -> 11
        state = handle_command(state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"direction": "NEXT", "step_count": 10}})
        self.assertEqual(state.current_slice, 11)

        # Relative navigation: PREVIOUS - 4 -> 7
        state = handle_command(state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"direction": "PREVIOUS", "step_count": 4}})
        self.assertEqual(state.current_slice, 7)

        # Relative overshoot clamps to 1
        state = handle_command(state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"direction": "PREVIOUS", "step_count": 50}})
        self.assertEqual(state.current_slice, 1)

        # Fallback when slice_index is None
        state = handle_command(state, {"intent": "navigate_slice", "status": "EXECUTE", "slots": {"slice_index": None, "direction": "NEXT", "step_count": 5}})
        self.assertEqual(state.current_slice, 6)

    # --- Event Handlers ---

    def test_log_event_lifecycle(self):
        """log_event appends milestones with comment, enforces FIFO cap of 50, and handles empty inputs."""
        cmd = {
            "intent": "log_event",
            "status": "EXECUTE",
            "command_id": "cmd_101",
            "timestamp": 1728381000.0,
            "slots": {"event_name": "insufflation complete", "comment": "15 mmHg achieved"},
        }
        state = handle_command(self.state, cmd)
        self.assertEqual(len(state.recent_milestones), 1)
        m = state.recent_milestones[0]
        self.assertEqual(m["event_name"], "insufflation complete")
        self.assertEqual(m["comment"], "15 mmHg achieved")

        # Empty or None event_name is rejected
        state = handle_command(state, {"intent": "log_event", "status": "EXECUTE", "slots": {"event_name": "   "}})
        state = handle_command(state, {"intent": "log_event", "status": "EXECUTE", "slots": {"event_name": None}})
        self.assertEqual(len(state.recent_milestones), 1)

        # None comment defaults to empty string
        state = handle_command(state, {"intent": "log_event", "status": "EXECUTE", "slots": {"event_name": "artery clamped", "comment": None}})
        self.assertEqual(state.recent_milestones[-1]["comment"], "")

        # FIFO eviction at 50 records
        for i in range(55):
            state = handle_command(state, {"intent": "log_event", "status": "EXECUTE", "slots": {"event_name": f"step_{i}"}})
        self.assertEqual(len(state.recent_milestones), 50)
        self.assertEqual(state.recent_milestones[-1]["event_name"], "step_54")

    # --- Emergency Handlers ---

    def test_emergency_halt_and_clear(self):
        """halt_emergency engages brake interlock; clear_emergency initiates +Z cavity evacuation."""
        self.assertFalse(self.state.emergency_active)

        state = handle_command(self.state, {"intent": "halt_emergency", "status": "EXECUTE", "slots": {}})
        self.assertTrue(state.emergency_active)
        self.assertEqual(state.telemetry.status, "EMERGENCY HALT (LOCKED)")

        state = handle_command(state, {"intent": "clear_emergency", "status": "EXECUTE", "slots": {}})
        self.assertFalse(state.emergency_active)
        self.assertEqual(state.telemetry.status, "EVACUATING CAVITY (+Z ASCENT)")
        self.assertGreaterEqual(state.telemetry.z_mm, 250.0)

    # --- Tool Handlers ---

    def test_tool_handling_and_telemetry(self):
        """pick_tool tracks held tool and updates telemetry; park_arm clears held tool and resets telemetry."""
        self.assertIsNone(self.state.held_tool)

        # Valid pick_tool updates held tool and telemetry
        state = handle_command(self.state, {"intent": "pick_tool", "status": "EXECUTE", "slots": {"tool_identifier": "scalpel"}})
        self.assertEqual(state.held_tool, "scalpel")
        self.assertEqual(state.telemetry.gripper_status, "HOLDING SCALPEL")
        self.assertEqual(state.telemetry.status, "OPERATIONAL")

        # Invalid tool is rejected
        state = handle_command(state, {"intent": "pick_tool", "status": "EXECUTE", "slots": {"tool_identifier": "laser_pointer"}})
        self.assertEqual(state.held_tool, "scalpel")

        # park_arm clears held tool and returns gripper to standby
        state = handle_command(state, {"intent": "park_arm", "status": "EXECUTE", "slots": {}})
        self.assertIsNone(state.held_tool)
        self.assertEqual(state.telemetry.gripper_status, "STANDBY / OFFLINE")
        self.assertEqual(state.telemetry.status, "PARKED / STANDBY")

    # --- Safety & Edge Cases ---

    def test_safety_and_edge_cases(self):
        """Non-EXECUTE commands, unknown intents, non-dict payloads, and null dictionaries are handled safely."""
        # Non-EXECUTE status does not alter state
        state = handle_command(self.state, {"intent": "open_scan", "status": "AMBIGUOUS", "slots": {"scan_type": "ct"}})
        self.assertIsNone(state.active_scan)

        # Unknown intent does not crash
        state = handle_command(state, {"intent": "unrecognized_intent", "status": "EXECUTE", "slots": {}})
        self.assertIsNotNone(state)

        # Corrupt command types return state untouched
        state = handle_command(state, "not a dictionary")
        self.assertEqual(state, self.state)

        # MonitorState.from_dict handles null/malformed dictionary gracefully
        malformed = {
            "telemetry": None,
            "recent_milestones": None,
            "current_slice": None,
            "zoom_factor": None,
            "last_updated": None,
            "last_command": None,
        }
        hydrated = MonitorState.from_dict(malformed)
        self.assertEqual(hydrated.current_slice, 1)
        self.assertEqual(hydrated.zoom_factor, 1.0)
        self.assertEqual(hydrated.recent_milestones, [])
        self.assertEqual(hydrated.telemetry.status, "STANDBY / OFFLINE")


if __name__ == "__main__":
    unittest.main()
