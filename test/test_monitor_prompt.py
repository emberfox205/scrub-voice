#!/usr/bin/env python3
# ==============================================================================
# HELPER / RUN COMMANDS:
#   # 1. Interactive quick menu prompt console:
#   python test/test_monitor_prompt.py
#
#   # 2. Free-form voice/text prompt directly from CLI:
#   python test/test_monitor_prompt.py "open ct scan"
#   python test/test_monitor_prompt.py "zoom in 2.5x"
#   python test/test_monitor_prompt.py "emergency stop immediately"
#
#   # 3. Direct argument flags:
#   python test/test_monitor_prompt.py --open-scan ct --slice 45 --zoom 2.0
#
#   # 4. Automated clinical demo preset:
#   python test/test_monitor_prompt.py --preset routine
# ==============================================================================
"""Interactive Intraoperative Monitor Prompt Interface.

Allows testers and presenters to dispatch standardized, schema-compliant JSON
command envelopes to the running Orchestrator and Monitor Service via ZeroMQ.
Covers all 9 visual monitor commands:
  - DICOM Viewport : open_scan, close_scan, zoom_display, navigate_slice
  - EMR Logging    : log_event
  - Tool Handling  : pick_tool, park_arm
  - Safety Interlock: halt_emergency, clear_emergency

Can be run:
  1. Interactively with numbered quick menu: python test/test_monitor_prompt.py
  2. With free-form natural language text:   python test/test_monitor_prompt.py "zoom in 3x"
  3. With direct command-line arguments:    python test/test_monitor_prompt.py --zoom-in 3.5
  4. With automated multi-step presets:     python test/test_monitor_prompt.py --preset routine

Watch the clinical dashboard at http://localhost:8000 react in real time!
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
import orjson
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCHEMA_PATH = REPO_ROOT / "schemas" / "command_schema.json"
DEFAULT_NLU_ADDR = os.getenv("SCRUB_NLU_ADDR", "ipc:///tmp/scrub_nlu.ipc")
DEFAULT_PRIO_ADDR = os.getenv("SCRUB_NLU_PRIORITY_ADDR", "ipc:///tmp/scrub_nlu_priority.ipc")

# Pre-load Schema
with open(DEFAULT_SCHEMA_PATH, "rb") as f:
    SCHEMA = orjson.loads(f.read())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)

SEQ_COUNTER = itertools.count(1)


def make_envelope(
    intent: str,
    category: str,
    transcript: str,
    slots: Optional[Dict[str, Any]] = None,
    priority: str = "NORMAL",
    confidence: float = 0.98,
    status: str = "EXECUTE",
) -> Dict[str, Any]:
    """Builds and validates a 10-key schema payload."""
    seq = next(SEQ_COUNTER)
    now = datetime.now()
    cmd_id = f"cmd_{now.strftime('%m%d_%H%M%S')}_{seq:03d}"
    payload = {
        "command_id": cmd_id,
        "timestamp": round(time.time(), 3),
        "latency_ms": 12.5,
        "raw_transcript": transcript,
        "intent": intent,
        "category": category,
        "slots": slots if slots is not None else {},
        "confidence": round(confidence, 2),
        "priority": priority,
        "status": status,
    }
    VALIDATOR.validate(payload)
    return payload


def parse_prompt(text: str) -> Optional[Dict[str, Any]]:
    """Parses natural text or shorthand into a schema-compliant command envelope."""
    clean = text.strip().lower()

    # 1. Emergency Commands (HIGH Priority, EMERGENCY category)
    if any(k in clean for k in ["halt", "emergency stop", "freeze", "stop robot", "abort", "e-stop", "hard stop"]):
        return make_envelope(
            intent="halt_emergency",
            category="EMERGENCY",
            transcript=text,
            priority="HIGH",
        )
    if any(k in clean for k in ["clear emergency", "clear area", "evacuate", "resume", "clear alarm", "back off"]):
        return make_envelope(
            intent="clear_emergency",
            category="EMERGENCY",
            transcript=text,
            priority="HIGH",
        )

    # 2. Open Scan (MONITOR_DISPLAY category)
    m_open = re.search(r"(?:open|show|display|load|bring up|view)\s+(?:the\s+)?(ct|mri|xray|x-ray)(?:\s+scan|\s+view)?", clean)
    if m_open:
        scan_type = m_open.group(1).replace("-", "")
        return make_envelope(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"scan_type": scan_type},
        )
    for s in ["ct", "mri", "xray"]:
        if s in clean and any(v in clean for v in ["open", "show", "display", "load"]):
            return make_envelope(
                intent="open_scan",
                category="MONITOR_DISPLAY",
                transcript=text,
                slots={"scan_type": s},
            )

    # 3. Close Scan (MONITOR_DISPLAY category)
    if any(k in clean for k in ["close scan", "close display", "hide scan", "dismiss scan", "close viewer", "close ct", "close mri", "close xray"]):
        slots: Dict[str, Any] = {}
        for s in ["ct", "mri", "xray"]:
            if s in clean:
                slots["scan_type"] = s
                break
        return make_envelope(
            intent="close_scan",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots=slots,
        )

    # 4. Zoom Display (MONITOR_DISPLAY category)
    # Supports: "zoom in 2x", "zoom in 3.5", "zoom out 2x", "zoom in", "zoom out", "zoom 4x", "magnify 2x"
    m_zoom = re.search(r"(?:zoom|magnify|enlarge|shrink)\s*(in|out)?\s*([0-9.]+)?\s*(?:x|times)?", clean)
    if m_zoom or "zoom" in clean:
        direction = "OUT" if "out" in clean else "IN"
        factor = 2.0
        if m_zoom and m_zoom.group(2):
            try:
                factor = float(m_zoom.group(2))
            except ValueError:
                factor = 2.0
        # Schema clamps factor between 1.1 and 10.0
        factor = round(max(1.1, min(10.0, factor)), 2)
        return make_envelope(
            intent="zoom_display",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": direction, "factor": factor},
        )

    # 5. Navigate Slice (MONITOR_DISPLAY category)
    # Direct jump: "slice 45", "goto slice 45", "go to slice 20", "jump to slice 12"
    m_slice_num = re.search(r"(?:goto\s+slice|go\s+to\s+slice|jump\s+to\s+slice|slice)\s*([0-9]+)", clean)
    if m_slice_num:
        idx = max(1, int(m_slice_num.group(1)))
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"slice_index": idx},
        )
    # Relative advance (NEXT): "next slice", "advance 5 slices", "next 3 slices", "forward 2"
    m_next = re.search(r"(?:advance|next|forward)\s*(?:slice|slices)?\s*([0-9]+)?", clean)
    if m_next and any(k in clean for k in ["next", "advance", "forward"]):
        step = 1
        if m_next.group(1):
            try:
                step = max(1, int(m_next.group(1)))
            except ValueError:
                step = 1
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": "NEXT", "step_count": step},
        )
    # Relative rewind (PREVIOUS): "previous slice", "prev slice", "back 3 slices", "rewind 2"
    m_prev = re.search(r"(?:previous|prev|back|rewind)\s*(?:slice|slices)?\s*([0-9]+)?", clean)
    if m_prev and any(k in clean for k in ["prev", "previous", "back", "rewind"]):
        step = 1
        if m_prev.group(1):
            try:
                step = max(1, int(m_prev.group(1)))
            except ValueError:
                step = 1
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": "PREVIOUS", "step_count": step},
        )

    # 6. Tool Handling (TOOL_HANDLING category)
    TOOL_MAP = {
        "scalpel": "scalpel", "knife": "scalpel", "blade": "scalpel",
        "scissors": "scissors", "iris": "scissors", "metz": "scissors", "metzenbaum": "scissors",
        "hemostat": "hemostat", "clamp": "hemostat", "snap": "hemostat",
        "forceps": "forceps", "tweezers": "forceps", "pickups": "forceps",
    }
    for alias, canonical in TOOL_MAP.items():
        if alias in clean and any(v in clean for v in ["pick", "hand", "pass", "give", "need"]):
            return make_envelope(
                intent="pick_tool",
                category="TOOL_HANDLING",
                transcript=text,
                slots={"tool_identifier": canonical},
            )
    if any(k in clean for k in ["park arm", "park robot", "home position", "standby arm", "retract arm"]):
        return make_envelope(
            intent="park_arm",
            category="TOOL_HANDLING",
            transcript=text,
            slots={},
        )

    # 7. Log Event (MONITOR_DISPLAY category)
    if clean.startswith("log") or clean.startswith("record") or "milestone" in clean:
        m_log = re.search(
            r"(?:log\s+event|record\s+milestone|record\s+event|log\s+milestone|log|record)[:\s]+(.+?)(?:[,\s]+(?:note|comment)[:\s]+(.+?))?(?:\s+(?:now|please)|$)",
            text,
            re.IGNORECASE
        )
        if m_log:
            event_name = m_log.group(1).strip().rstrip(",")
            comment = m_log.group(2).strip() if m_log.group(2) else None
            if not event_name:
                event_name = "surgical milestone"
            slots = {"event_name": event_name}
            if comment:
                slots["comment"] = comment
            return make_envelope(
                intent="log_event",
                category="MONITOR_DISPLAY",
                transcript=text,
                slots=slots,
            )

    # 8. Fallback to pure NLU pipeline if available
    try:
        from src.nlu.pipeline import VoiceToIntentPipeline
        pipeline = VoiceToIntentPipeline(num_threads=1)
        return pipeline.process(text)
    except Exception:
        # Out-of-domain envelope
        return make_envelope(
            intent="unknown",
            category="UNKNOWN",
            transcript=text,
            confidence=0.40,
            status="OUT-OF-DOMAIN",
        )


class MonitorPromptSender:
    """Manages ZeroMQ PUSH connections to normal and priority orchestrator channels."""

    def __init__(self, nlu_addr: str = DEFAULT_NLU_ADDR, prio_addr: str = DEFAULT_PRIO_ADDR):
        self.context = zmq.Context()
        self.normal_push = self.context.socket(zmq.PUSH)
        self.normal_push.connect(nlu_addr)

        self.prio_push = self.context.socket(zmq.PUSH)
        self.prio_push.connect(prio_addr)
        print(f"[*] Connected to Monitor Service:")
        print(f"    - Normal Channel   : {nlu_addr}")
        print(f"    - Priority Channel : {prio_addr}")

    def send(self, payload: Dict[str, Any]) -> None:
        """Pushes command payload to the appropriate ZeroMQ channel."""
        raw_bytes = orjson.dumps(payload)
        if payload.get("priority") == "HIGH":
            self.prio_push.send(raw_bytes, zmq.NOBLOCK)
            channel = "HIGH-PRIORITY CHANNEL"
        else:
            self.normal_push.send(raw_bytes, zmq.NOBLOCK)
            channel = "NORMAL CHANNEL"

        print(f"\n[SENT via {channel}] -> ID: {payload['command_id']}")
        print(json.dumps(payload, indent=2))
        print(">> View dashboard update at: http://localhost:8000\n")

    def close(self):
        self.normal_push.close(linger=100)
        self.prio_push.close(linger=100)
        self.context.term()


# 20 Standardized Menu Presets Covering All 9 Supported Monitor Actions
MENU_OPTIONS = [
    # --- DICOM Viewport (MONITOR_DISPLAY) ---
    ("1", "Open CT Scan", "open ct scan"),
    ("2", "Open MRI Scan", "open mri scan"),
    ("3", "Open X-Ray View", "open xray"),
    ("4", "Close Active Scan", "close scan"),
    ("5", "Zoom In 2.0x (Default)", "zoom in 2x"),
    ("6", "Zoom In 3.5x (Deep Magnification)", "zoom in 3.5x"),
    ("7", "Zoom Out 2.0x (Reduce)", "zoom out 2x"),
    ("8", "Next Slice (+1)", "next slice"),
    ("9", "Advance 5 Slices (+5)", "advance 5 slices"),
    ("10", "Previous Slice (-1)", "previous slice"),
    ("11", "Jump to Slice 42", "goto slice 42"),
    # --- EMR Clinical Milestone Dictation (MONITOR_DISPLAY) ---
    ("12", "Log Milestone (Event Only)", "log event: first incision"),
    ("13", "Log Milestone (+ Clinical Note)", "record milestone: trocar insertion complete, note patient vitals stable"),
    # --- Robot Tool Handling & Gripper (TOOL_HANDLING) ---
    ("14", "Pick Scalpel", "hand me the scalpel"),
    ("15", "Pick Hemostat Clamp", "pass the hemostat"),
    ("16", "Pick Surgical Scissors", "give me the scissors"),
    ("17", "Pick Forceps", "pass the forceps"),
    ("18", "Park Robotic Arm", "park arm"),
    # --- Safety & Emergency Interlock (EMERGENCY - High Priority) ---
    ("19", "HALT EMERGENCY (Lock Brakes)", "emergency stop immediately"),
    ("20", "CLEAR EMERGENCY (+Z Evacuation)", "clear emergency"),
]


def run_preset(sender: MonitorPromptSender, preset_name: str) -> None:
    """Executes multi-step automated test sequences with timed pacing."""
    if preset_name == "routine":
        steps = [
            "open ct scan",
            "goto slice 45",
            "zoom in 2.5x",
            "hand me the scalpel",
            "log event: first incision completed, note vitals steady",
            "pass the hemostat",
            "park arm",
        ]
    elif preset_name == "emergency":
        steps = [
            "open mri scan",
            "hand me the scalpel",
            "emergency stop immediately",
            "clear emergency",
            "park arm",
        ]
    elif preset_name == "all":
        steps = [phrase for _, _, phrase in MENU_OPTIONS]
    else:
        print(f"Unknown preset '{preset_name}'. Supported: 'routine', 'emergency', 'all'")
        return

    print(f"\n[*] Executing '{preset_name}' test preset ({len(steps)} steps)...")
    for i, step in enumerate(steps, 1):
        print(f"\n--- [Step {i}/{len(steps)}] Sending: \"{step}\" ---")
        payload = parse_prompt(step)
        if payload:
            sender.send(payload)
        time.sleep(1.0)
    print(f"\n[+] Preset '{preset_name}' execution finished.")


def interactive_loop(sender: MonitorPromptSender) -> None:
    """Runs the interactive console prompt loop."""
    print("\n" + "=" * 70)
    print("      SCRUB-VOICE STANDARDIZED MONITOR PROMPT INTERFACE")
    print("=" * 70)
    print("  Make sure the monitor service is running:")
    print("  uv run python src/orchestrator/run_monitor_service.py")
    print("  Dashboard UI: http://localhost:8000")
    print("-" * 70)
    print("  Quick Menu Options:")
    for key, desc, phrase in MENU_OPTIONS:
        print(f"    [{key:>2}] {desc:<35} -> \"{phrase}\"")
    print("    [ p] Run 'routine' surgical workflow preset")
    print("    [ q] Quit")
    print("-" * 70)
    print("  Or type ANY custom prompt (e.g. 'zoom 3.5x', 'slice 24', 'pick scalpel')")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("monitor-prompt > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if user_input.lower() in ("p", "preset"):
            run_preset(sender, "routine")
            continue

        # Check menu numbers
        matched_phrase = user_input
        for key, desc, phrase in MENU_OPTIONS:
            if user_input == key:
                matched_phrase = phrase
                print(f"[*] Selected [{key}]: {desc} (\"{phrase}\")")
                break

        try:
            payload = parse_prompt(matched_phrase)
            if payload:
                sender.send(payload)
            else:
                print("[-] Could not parse prompt into a valid command schema envelope.")
        except Exception as e:
            print(f"[!] Error creating/sending command: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send standardized prompts and schema-compliant commands to the Intraoperative Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python test/test_monitor_prompt.py
  uv run python test/test_monitor_prompt.py "open ct scan"
  uv run python test/test_monitor_prompt.py --zoom-in 3.5
  uv run python test/test_monitor_prompt.py --zoom-out 2.0
  uv run python test/test_monitor_prompt.py --slice 45
  uv run python test/test_monitor_prompt.py --next-slice 5
  uv run python test/test_monitor_prompt.py --open-scan ct
  uv run python test/test_monitor_prompt.py --pick-tool scalpel
  uv run python test/test_monitor_prompt.py --log-event "trocar placed" --comment "vitals steady"
  uv run python test/test_monitor_prompt.py --preset routine
        """,
    )
    parser.add_argument("prompt", nargs="*", help="Optional free-text prompt to parse and send")
    parser.add_argument("--addr", default=DEFAULT_NLU_ADDR, help="Normal NLU ZeroMQ address")
    parser.add_argument("--priority-addr", default=DEFAULT_PRIO_ADDR, help="Priority NLU ZeroMQ address")

    # Direct Command Options
    cmd_group = parser.add_argument_group("Direct Command Arguments")
    cmd_group.add_argument("--open-scan", choices=["ct", "mri", "xray"], help="Open specified scan modality")
    cmd_group.add_argument("--close-scan", action="store_true", help="Close active scan")
    cmd_group.add_argument("--zoom-in", type=float, nargs="?", const=2.0, metavar="FACTOR", help="Zoom in (default: 2.0x, range: 1.1-10.0)")
    cmd_group.add_argument("--zoom-out", type=float, nargs="?", const=2.0, metavar="FACTOR", help="Zoom out (default: 2.0x, range: 1.1-10.0)")
    cmd_group.add_argument("--slice", type=int, metavar="INDEX", help="Direct slice jump (e.g. --slice 42)")
    cmd_group.add_argument("--next-slice", type=int, nargs="?", const=1, metavar="STEP", help="Advance slices (default step: 1)")
    cmd_group.add_argument("--prev-slice", type=int, nargs="?", const=1, metavar="STEP", help="Rewind slices (default step: 1)")
    cmd_group.add_argument("--pick-tool", choices=["scalpel", "scissors", "hemostat", "forceps"], help="Pick surgical tool")
    cmd_group.add_argument("--park-arm", action="store_true", help="Park robot arm to standby")
    cmd_group.add_argument("--halt", action="store_true", help="Trigger EMERGENCY HALT (Priority HIGH)")
    cmd_group.add_argument("--clear", action="store_true", help="Clear emergency state (+Z ascent)")
    cmd_group.add_argument("--log-event", type=str, metavar="NAME", help="Log surgical milestone event")
    cmd_group.add_argument("--comment", type=str, metavar="TEXT", help="Optional clinical comment for --log-event")
    cmd_group.add_argument("--preset", choices=["routine", "emergency", "all"], help="Execute automated sequence preset")

    args = parser.parse_args()
    sender = MonitorPromptSender(nlu_addr=args.addr, prio_addr=args.priority_addr)

    try:
        # 1. Preset sequences
        if args.preset:
            run_preset(sender, args.preset)
            return

        # 2. Direct flags
        payload = None
        if args.halt:
            payload = make_envelope("halt_emergency", "EMERGENCY", "emergency stop immediately", priority="HIGH")
        elif args.clear:
            payload = make_envelope("clear_emergency", "EMERGENCY", "clear emergency", priority="HIGH")
        elif args.open_scan:
            payload = make_envelope("open_scan", "MONITOR_DISPLAY", f"open {args.open_scan}", slots={"scan_type": args.open_scan})
        elif args.close_scan:
            payload = make_envelope("close_scan", "MONITOR_DISPLAY", "close scan", slots={})
        elif args.zoom_in is not None:
            factor = round(max(1.1, min(10.0, float(args.zoom_in))), 2)
            payload = make_envelope("zoom_display", "MONITOR_DISPLAY", f"zoom in {factor}x", slots={"direction": "IN", "factor": factor})
        elif args.zoom_out is not None:
            factor = round(max(1.1, min(10.0, float(args.zoom_out))), 2)
            payload = make_envelope("zoom_display", "MONITOR_DISPLAY", f"zoom out {factor}x", slots={"direction": "OUT", "factor": factor})
        elif args.slice is not None:
            idx = max(1, int(args.slice))
            payload = make_envelope("navigate_slice", "MONITOR_DISPLAY", f"goto slice {idx}", slots={"slice_index": idx})
        elif args.next_slice is not None:
            step = max(1, int(args.next_slice))
            payload = make_envelope("navigate_slice", "MONITOR_DISPLAY", f"advance {step} slices", slots={"direction": "NEXT", "step_count": step})
        elif args.prev_slice is not None:
            step = max(1, int(args.prev_slice))
            payload = make_envelope("navigate_slice", "MONITOR_DISPLAY", f"previous {step} slices", slots={"direction": "PREVIOUS", "step_count": step})
        elif args.pick_tool:
            payload = make_envelope("pick_tool", "TOOL_HANDLING", f"hand me the {args.pick_tool}", slots={"tool_identifier": args.pick_tool})
        elif args.park_arm:
            payload = make_envelope("park_arm", "TOOL_HANDLING", "park arm", slots={})
        elif args.log_event:
            event_name = args.log_event.strip()
            slots = {"event_name": event_name}
            transcript = f"log event: {event_name}"
            if args.comment:
                comment = args.comment.strip()
                slots["comment"] = comment
                transcript += f", note {comment}"
            payload = make_envelope("log_event", "MONITOR_DISPLAY", transcript, slots=slots)
        elif args.prompt:
            prompt_text = " ".join(args.prompt)
            payload = parse_prompt(prompt_text)

        if payload:
            sender.send(payload)
        elif not any([args.preset, args.halt, args.clear, args.open_scan, args.close_scan, args.zoom_in, args.zoom_out, args.slice, args.next_slice, args.prev_slice, args.pick_tool, args.park_arm, args.log_event, args.prompt]):
            interactive_loop(sender)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
