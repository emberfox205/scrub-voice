#!/usr/bin/env python3
"""
Scrub-Voice Mock Command Publisher.

Simulates the upstream NLU speech recognition pipeline by generating
schema-compliant command JSON payloads and pushing them to the Central
Command Dispatcher via ZeroMQ (PUSH socket).

Provides realistic surgical presets for testing downstream subsystems
(UR10e robot arm, RG-2 gripper, endoscope camera, and monitor UI).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import orjson
import zmq

# Dynamic path resolution according to AGENTS.md
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCHEMA_PATH = REPO_ROOT / "schemas" / "command_schema.json"
DEFAULT_NLU_ADDR = os.getenv("SCRUB_NLU_ADDR", "ipc:///tmp/scrub_nlu.ipc")
DEFAULT_NLU_PRIORITY_ADDR = os.getenv("SCRUB_NLU_PRIORITY_ADDR", "ipc:///tmp/scrub_nlu_priority.ipc")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MockPublisher] %(message)s",
)
logger = logging.getLogger("mock_publisher")



class CommandFactory:
    """Generates schema-compliant command payloads with dynamic timestamps and sequence IDs."""

    _seq_counter = itertools.count(1)

    @classmethod
    def create(
        cls,
        intent: str,
        category: str,
        raw_transcript: str,
        slots: Optional[Dict[str, Any]] = None,
        confidence: float = 0.98,
        priority: str = "NORMAL",
        status: str = "EXECUTE",
        latency_ms: float = 38.5,
    ) -> Dict[str, Any]:
        """Constructs a single payload adhering strictly to schemas/command_schema.json."""
        now = datetime.now()
        seq = next(cls._seq_counter)
        cmd_id = f"cmd_{now.strftime('%m%d_%H%M%S')}_{seq:03d}"

        return {
            "command_id": cmd_id,
            "timestamp": time.time(),
            "latency_ms": round(latency_ms, 2),
            "raw_transcript": raw_transcript,
            "intent": intent,
            "category": category,
            "slots": slots if slots is not None else {},
            "confidence": round(confidence, 4),
            "priority": priority,
            "status": status,
        }


# ==============================================================================
# SURGICAL PRESET DEFINITIONS
# ==============================================================================

PRESETS: Dict[str, List[Dict[str, Any]]] = {
    # --------------------------------------------------------------------------
    # Preset 1: Routine Laparoscopic Setup & Incision
    # --------------------------------------------------------------------------
    "routine": [
        CommandFactory.create(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            raw_transcript="open the patient's CT scan",
            slots={"scan_type": "ct"},
            confidence=0.99,
        ),
        CommandFactory.create(
            intent="zoom_display",
            category="MONITOR_DISPLAY",
            raw_transcript="zoom in by two point five times",
            slots={"direction": "IN", "factor": 2.5},
            confidence=0.97,
        ),
        CommandFactory.create(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            raw_transcript="advance five slices",
            slots={"direction": "NEXT", "step_count": 5},
            confidence=0.98,
        ),
        CommandFactory.create(
            intent="log_event",
            category="MONITOR_DISPLAY",
            raw_transcript="log milestone: abdominal insufflation completed, vitals steady",
            slots={"event_name": "abdominal insufflation completed", "comment": "vitals steady"},
            confidence=0.96,
        ),
        CommandFactory.create(
            intent="pick_tool",
            category="TOOL_HANDLING",
            raw_transcript="pass the scalpel",
            slots={"tool_identifier": "scalpel"},
            confidence=0.99,
        ),
        CommandFactory.create(
            intent="pick_tool",
            category="TOOL_HANDLING",
            raw_transcript="hand me the hemostat clamp",
            slots={"tool_identifier": "hemostat"},
            confidence=0.98,
        ),
        CommandFactory.create(
            intent="park_arm",
            category="TOOL_HANDLING",
            raw_transcript="park arm outside the surgical area",
            slots={},
            confidence=0.99,
        ),
    ],

    # --------------------------------------------------------------------------
    # Preset 2: Endoscopic Camera Wand Jogging & Positioning
    # --------------------------------------------------------------------------
    "camera": [
        CommandFactory.create(
            intent="move_camera",
            category="CAMERA_LIGHT",
            raw_transcript="camera forward ten centimeters",
            slots={"direction": "FORWARD", "distance_cm": 10.0},
            confidence=0.97,
        ),
        CommandFactory.create(
            intent="tilt_camera",
            category="CAMERA_LIGHT",
            raw_transcript="tilt camera down twenty degrees",
            slots={"direction": "DOWN", "angle_deg": 20.0},
            confidence=0.96,
        ),
        CommandFactory.create(
            intent="move_camera",
            category="CAMERA_LIGHT",
            raw_transcript="camera left five centimeters",
            slots={"direction": "LEFT", "distance_cm": 5.0},
            confidence=0.98,
        ),
        CommandFactory.create(
            intent="reset_camera",
            category="CAMERA_LIGHT",
            raw_transcript="reset camera to home overhead pose",
            slots={},
            confidence=0.99,
        ),
        CommandFactory.create(
            intent="log_event",
            category="MONITOR_DISPLAY",
            raw_transcript="record event: bile duct successfully visualized",
            slots={"event_name": "bile duct successfully visualized"},
            confidence=0.95,
        ),
    ],

    # --------------------------------------------------------------------------
    # Preset 3: Emergency Interruption & Safety Interlock Test
    # --------------------------------------------------------------------------
    "emergency": [
        CommandFactory.create(
            intent="pick_tool",
            category="TOOL_HANDLING",
            raw_transcript="pass the iris scissors",
            slots={"tool_identifier": "scissors"},
            confidence=0.99,
        ),
        # Immediate high-priority halt during tool motion
        CommandFactory.create(
            intent="halt_emergency",
            category="EMERGENCY",
            raw_transcript="emergency stop freeze robot",
            slots={},
            confidence=0.995,
            priority="HIGH",
        ),
        # Rapid vertical ascent to evacuate surgical cavity
        CommandFactory.create(
            intent="clear_emergency",
            category="EMERGENCY",
            raw_transcript="emergency clear evacuate cavity",
            slots={},
            confidence=0.995,
            priority="HIGH",
        ),
        CommandFactory.create(
            intent="park_arm",
            category="TOOL_HANDLING",
            raw_transcript="park arm to standby",
            slots={},
            confidence=0.98,
        ),
    ],

    # --------------------------------------------------------------------------
    # Preset 4: Ambiguous Speech & Out-of-Domain Filtering (Negative Testing)
    # --------------------------------------------------------------------------
    "noisy": [
        # Out-of-domain conversational speech
        CommandFactory.create(
            intent="unknown",
            category="UNKNOWN",
            raw_transcript="what time does the next operation start",
            slots={},
            confidence=0.92,
            status="OUT-OF-DOMAIN",
        ),
        # Ambiguous command (missing required entity)
        CommandFactory.create(
            intent="pick_tool",
            category="TOOL_HANDLING",
            raw_transcript="pass that thing over there",
            slots={},
            confidence=0.32,
            status="AMBIGUOUS",
        ),
        # Ambiguous scan request (missing scan modality)
        CommandFactory.create(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            raw_transcript="open the scan image",
            slots={},
            confidence=0.45,
            status="AMBIGUOUS",
        ),
        # Recovery with valid zoom
        CommandFactory.create(
            intent="zoom_display",
            category="MONITOR_DISPLAY",
            raw_transcript="zoom in",
            slots={"direction": "IN", "factor": 2.0},
            confidence=0.98,
            status="EXECUTE",
        ),
    ],

    # --------------------------------------------------------------------------
    # Preset 5: Full Tour (Exercises All 12 Canonical Intents in Sequence)
    # --------------------------------------------------------------------------
    "full": [
        CommandFactory.create("open_scan", "MONITOR_DISPLAY", "open MRI scan", {"scan_type": "mri"}),
        CommandFactory.create("zoom_display", "MONITOR_DISPLAY", "zoom in 3x", {"direction": "IN", "factor": 3.0}),
        CommandFactory.create("navigate_slice", "MONITOR_DISPLAY", "go to slice 45", {"slice_index": 45}),
        CommandFactory.create("close_scan", "MONITOR_DISPLAY", "close active scan", {"scan_type": "mri"}),
        CommandFactory.create("move_camera", "CAMERA_LIGHT", "camera up 5 cm", {"direction": "UP", "distance_cm": 5.0}),
        CommandFactory.create("tilt_camera", "CAMERA_LIGHT", "tilt right 15 deg", {"direction": "RIGHT", "angle_deg": 15.0}),
        CommandFactory.create("reset_camera", "CAMERA_LIGHT", "center camera view", {}),
        CommandFactory.create("pick_tool", "TOOL_HANDLING", "give me the forceps", {"tool_identifier": "forceps"}),
        CommandFactory.create("park_arm", "TOOL_HANDLING", "park the arm", {}),
        CommandFactory.create("log_event", "MONITOR_DISPLAY", "log event: procedure finished", {"event_name": "procedure finished"}),
        CommandFactory.create("halt_emergency", "EMERGENCY", "halt robot", {}, priority="HIGH"),
        CommandFactory.create("clear_emergency", "EMERGENCY", "clear emergency area", {}, priority="HIGH"),
    ],

    # --------------------------------------------------------------------------
    # Preset 6: Modality Switching (CT -> MRI -> X-Ray)
    # --------------------------------------------------------------------------
    "switch_scans": [
        CommandFactory.create(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            raw_transcript="open the patient's CT scan",
            slots={"scan_type": "ct"},
            confidence=0.99,
        ),
        CommandFactory.create(
            intent="zoom_display",
            category="MONITOR_DISPLAY",
            raw_transcript="zoom in 2x",
            slots={"direction": "IN", "factor": 2.0},
            confidence=0.98,
        ),
        CommandFactory.create(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            raw_transcript="advance 12 slices",
            slots={"direction": "NEXT", "step_count": 12},
            confidence=0.98,
        ),
        CommandFactory.create(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            raw_transcript="switch to MRI scan",
            slots={"scan_type": "mri"},
            confidence=0.99,
        ),
        CommandFactory.create(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            raw_transcript="display chest X-ray",
            slots={"scan_type": "xray"},
            confidence=0.97,
        ),
    ],
}


class MockCommandPublisher:
    """ZeroMQ PUSH publisher that sends mock surgical command presets."""

    def __init__(
        self,
        nlu_addr: str = DEFAULT_NLU_ADDR,
        nlu_priority_addr: str = DEFAULT_NLU_PRIORITY_ADDR,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.nlu_addr = nlu_addr
        self.nlu_priority_addr = nlu_priority_addr
        self.schema_path = Path(schema_path)

        # Validate schema existence
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Command schema not found at {self.schema_path}")

        schema = orjson.loads(self.schema_path.read_bytes())
        self.validator = jsonschema.Draft202012Validator(schema)
        logger.info("Loaded schema from %s for preset validation", self.schema_path)

        # ZeroMQ setup: Dual PUSH sockets (Normal + Priority)
        self.context = zmq.Context()

        # 1. Normal command channel
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.setsockopt(zmq.SNDHWM, 50)
        self.push_sock.setsockopt(zmq.SNDTIMEO, 1000)
        self.push_sock.connect(self.nlu_addr)
        logger.info("Mock Publisher normal channel connected to %s (PUSH, HWM=50)", self.nlu_addr)

        # 2. High-priority / emergency channel
        self.priority_push_sock = self.context.socket(zmq.PUSH)
        self.priority_push_sock.setsockopt(zmq.SNDHWM, 50)
        self.priority_push_sock.setsockopt(zmq.SNDTIMEO, 1000)
        self.priority_push_sock.connect(self.nlu_priority_addr)
        logger.info("Mock Publisher priority channel connected to %s (PUSH, HWM=50)", self.nlu_priority_addr)

    def validate_preset(self, payload: Dict[str, Any]) -> None:
        """Ensures every mock payload strictly conforms to the schema before sending."""
        self.validator.validate(payload)

    def send_command(self, payload: Dict[str, Any]) -> None:
        """Validates and pushes a single command payload over the appropriate ZeroMQ channel."""
        # Refresh timestamp and command_id for live freshness
        now = datetime.now()
        seq = int(payload["command_id"].split("_")[-1]) if "_" in payload["command_id"] else 1
        payload["timestamp"] = time.time()
        payload["command_id"] = f"cmd_{now.strftime('%m%d_%H%M%S')}_{seq:03d}"

        self.validate_preset(payload)
        raw_bytes = orjson.dumps(payload)

        is_priority = (
            payload.get("priority") == "HIGH"
            or payload.get("category") == "EMERGENCY"
        )
        target_sock = self.priority_push_sock if is_priority else self.push_sock
        channel_name = "PRIORITY-CHANNEL" if is_priority else "NORMAL-CHANNEL"

        target_sock.send(raw_bytes)

        logger.info(
            "Pushed [%s] via %-16s cmd=%s intent=%-16s cat=%-15s prio=%s",
            payload.get("status"),
            channel_name,
            payload["command_id"],
            payload["intent"],
            payload["category"],
            payload.get("priority", "NORMAL"),
        )

    def send_preset(self, preset_name: str, interval_sec: float = 0.5) -> int:
        """Sends an entire sequence of commands in a preset with configurable interval."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Options: {list(PRESETS.keys())}")

        commands = PRESETS[preset_name]
        logger.info("Sending preset '%s' (%d commands, interval=%.2fs)...", preset_name, len(commands), interval_sec)

        count = 0
        for cmd in commands:
            self.send_command(cmd)
            count += 1
            if interval_sec > 0:
                time.sleep(interval_sec)

        logger.info("Preset '%s' completed successfully (%d commands sent).", preset_name, count)
        return count

    def close(self) -> None:
        """Closes the ZeroMQ sockets and context cleanly."""
        self.push_sock.close(linger=100)
        self.priority_push_sock.close(linger=100)
        self.context.term()
        logger.info("Mock Publisher closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Scrub-Voice Mock Command Publisher")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()) + ["interactive"],
        default="routine",
        help="Command preset sequence to send",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.8,
        help="Delay in seconds between commands in a preset",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously loop the preset until Ctrl+C",
    )
    parser.add_argument(
        "--addr",
        default=DEFAULT_NLU_ADDR,
        help="ZeroMQ normal NLU socket address to push to",
    )
    parser.add_argument(
        "--priority-addr",
        default=DEFAULT_NLU_PRIORITY_ADDR,
        help="ZeroMQ priority NLU socket address to push to",
    )
    args = parser.parse_args()

    pub = MockCommandPublisher(nlu_addr=args.addr, nlu_priority_addr=args.priority_addr)


    try:
        if args.preset == "interactive":
            print("\n=======================================================")
            print("  Scrub-Voice Interactive Mock Publisher Console")
            print("=======================================================")
            print("  Available Presets:")
            for idx, p in enumerate(PRESETS.keys(), 1):
                print(f"    {idx}. {p:<12} ({len(PRESETS[p])} commands)")
            print("    0. Exit")
            print("=======================================================\n")

            presets_list = list(PRESETS.keys())
            while True:
                try:
                    choice = input("Select preset (number or name) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not choice or choice in ("0", "exit", "quit"):
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(presets_list):
                    selected = presets_list[int(choice) - 1]
                elif choice in PRESETS:
                    selected = choice
                else:
                    print(f"Unknown preset '{choice}'. Try again.")
                    continue

                pub.send_preset(selected, interval_sec=args.interval)
                print()
        else:
            while True:
                pub.send_preset(args.preset, interval_sec=args.interval)
                if not args.loop:
                    break
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
