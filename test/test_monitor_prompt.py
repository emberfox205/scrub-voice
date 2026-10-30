#!/usr/bin/env python3
"""Interactive Intraoperative Monitor Prompt Interface.

Allows testers to input standardized voice commands (natural language or schema shorthands)
and immediately dispatches schema-compliant JSON command envelopes to the running
Orchestrator and Monitor Service via ZeroMQ.

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
from typing import Any, Dict, Optional, Tuple

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
    """Builds a validated 10-key schema payload."""
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
    """Parses natural text or shorthand into a schema command envelope."""
    clean = text.strip().lower()

    # 1. Emergency Commands (HIGH Priority)
    if any(k in clean for k in ["halt", "emergency stop", "freeze", "stop robot"]):
        return make_envelope(
            intent="halt_emergency",
            category="EMERGENCY",
            transcript=text,
            priority="HIGH",
        )
    if any(k in clean for k in ["clear emergency", "resume", "clear alarm"]):
        return make_envelope(
            intent="clear_emergency",
            category="EMERGENCY",
            transcript=text,
            priority="HIGH",
        )

    # 2. Open Scan
    m_open = re.search(r"open\s+(ct|mri|xray|x-ray)", clean)
    if m_open:
        scan_type = m_open.group(1).replace("-", "")
        return make_envelope(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"scan_type": scan_type},
        )
    if "open" in clean and "scan" in clean:
        return make_envelope(
            intent="open_scan",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"scan_type": "ct"},
        )

    # 3. Close Scan
    if any(k in clean for k in ["close scan", "close display", "hide scan"]):
        return make_envelope(
            intent="close_scan",
            category="MONITOR_DISPLAY",
            transcript=text,
        )

    # 4. Zoom Display
    m_zoom = re.search(r"zoom\s+(in|out)?\s*([0-9.]+)?x?", clean)
    if m_zoom or "zoom" in clean:
        direction = "OUT" if "out" in clean else "IN"
        factor = 2.0
        if m_zoom and m_zoom.group(2):
            try:
                factor = float(m_zoom.group(2))
            except ValueError:
                factor = 2.0
        return make_envelope(
            intent="zoom_display",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": direction, "factor": factor},
        )

    # 5. Navigate Slice
    m_slice_num = re.search(r"(?:slice|goto slice|go to slice|jump to slice)\s*([0-9]+)", clean)
    if m_slice_num:
        idx = int(m_slice_num.group(1))
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"slice_index": idx},
        )
    if any(k in clean for k in ["next slice", "advance slice", "forward"]):
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": "FORWARD", "step": 1},
        )
    if any(k in clean for k in ["prev slice", "previous slice", "back slice"]):
        return make_envelope(
            intent="navigate_slice",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"direction": "BACKWARD", "step": 1},
        )

    # 6. Tool Handling
    m_tool = re.search(r"(?:pick|give me|hand me|pass)\s+(?:the\s+)?(scalpel|scissors|hemostat|forceps)", clean)
    if m_tool:
        tool_name = m_tool.group(1)
        return make_envelope(
            intent="pick_tool",
            category="TOOL_HANDLING",
            transcript=text,
            slots={"tool_identifier": tool_name},
        )
    if any(k in clean for k in ["park arm", "park robot", "home position"]):
        return make_envelope(
            intent="park_arm",
            category="TOOL_HANDLING",
            transcript=text,
        )

    # 7. Log Event
    if clean.startswith("log") or "record milestone" in clean:
        note = text.replace("log event", "").replace("log", "").strip()
        if not note:
            note = "Surgical milestone recorded by operator"
        return make_envelope(
            intent="log_event",
            category="MONITOR_DISPLAY",
            transcript=text,
            slots={"event_type": "MILESTONE", "note": note},
        )

    # Fallback to pure NLU pipeline if installed
    try:
        from src.nlu.pipeline import VoiceToIntentPipeline
        pipeline = VoiceToIntentPipeline(num_threads=1)
        return pipeline.process(text)
    except Exception:
        # Generic out-of-domain envelope
        return make_envelope(
            intent="unknown",
            category="UNKNOWN",
            transcript=text,
            confidence=0.40,
            status="OUT-OF-DOMAIN",
        )


class MonitorPromptSender:
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


MENU_OPTIONS = [
    ("1", "Open CT Scan", "open ct scan"),
    ("2", "Open MRI Scan", "open mri scan"),
    ("3", "Navigate to Slice 45", "goto slice 45"),
    ("4", "Zoom In 2.5x", "zoom in 2.5x"),
    ("5", "Pick Scalpel", "hand me the scalpel"),
    ("6", "Park Robotic Arm", "park arm"),
    ("7", "Log Milestone Note", "log event trocar placed successfully"),
    ("8", "HALT EMERGENCY (Preemption)", "emergency stop immediately"),
    ("9", "Clear Emergency", "clear emergency"),
    ("0", "Close Scan", "close scan"),
]


def interactive_loop(sender: MonitorPromptSender) -> None:
    print("\n" + "=" * 65)
    print("      SCRUB-VOICE STANDARDIZED MONITOR PROMPT INTERFACE")
    print("=" * 65)
    print("  Make sure the monitor service is running:")
    print("  python src/orchestrator/run_monitor_service.py")
    print("  Dashboard UI: http://localhost:8000")
    print("-" * 65)
    print("  Quick Menu Options:")
    for key, desc, phrase in MENU_OPTIONS:
        print(f"    [{key}] {desc:<28} -> \"{phrase}\"")
    print("    [q] Quit")
    print("-" * 65)
    print("  Or type ANY custom prompt (e.g., 'zoom 3x', 'slice 12', 'pick hemostat')")
    print("=" * 65 + "\n")

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

        # Check menu numbers
        matched_phrase = user_input
        for key, desc, phrase in MENU_OPTIONS:
            if user_input == key:
                matched_phrase = phrase
                print(f"[*] Selected: {desc} (\"{phrase}\")")
                break

        try:
            payload = parse_prompt(matched_phrase)
            if payload:
                sender.send(payload)
            else:
                print("[-] Could not parse prompt into a valid command schema envelope.")
        except Exception as e:
            print(f"[!] Error creating/sending command: {e}")


def main():
    parser = argparse.ArgumentParser(description="Send standardized prompts to the Intraoperative Monitor")
    parser.add_argument("prompt", nargs="*", help="Optional single prompt to send in batch mode")
    parser.add_argument("--addr", default=DEFAULT_NLU_ADDR, help="Normal NLU ZeroMQ address")
    parser.add_argument("--priority-addr", default=DEFAULT_PRIO_ADDR, help="Priority NLU ZeroMQ address")

    args = parser.parse_args()
    sender = MonitorPromptSender(nlu_addr=args.addr, prio_addr=args.priority_addr)

    try:
        if args.prompt:
            prompt_text = " ".join(args.prompt)
            payload = parse_prompt(prompt_text)
            sender.send(payload)
        else:
            interactive_loop(sender)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
