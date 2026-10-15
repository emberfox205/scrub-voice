"""
Tool Handling Intent Handlers for Scrub-Voice Monitor.
Tracks currently held surgical tool and arm parking state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.monitor.handlers.registry import register_intent
from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.handlers.tool")


VALID_TOOLS = {"scalpel", "scissors", "hemostat", "forceps"}


@register_intent("pick_tool")
def handle_pick_tool(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """Updates held surgical tool identifier and gripper telemetry."""
    slots = command.get("slots", {})
    tool_id = slots.get("tool_identifier")
    if tool_id is not None:
        norm_tool = str(tool_id).strip().lower()
        if norm_tool in VALID_TOOLS:
            state.held_tool = norm_tool
            state.telemetry.gripper_status = f"HOLDING {norm_tool.upper()}"
            state.telemetry.status = "OPERATIONAL"
            logger.info("Held tool updated to: %s (gripper holding)", state.held_tool)
        else:
            logger.warning("Rejected invalid tool_identifier: %r (valid: %s)", tool_id, VALID_TOOLS)
    return state


@register_intent("park_arm")
def handle_park_arm(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """Clears held tool and transitions robot and gripper to standby."""
    state.held_tool = None
    state.telemetry.gripper_status = "STANDBY / OFFLINE"
    state.telemetry.status = "PARKED / STANDBY"
    logger.info("Arm parked; held tool cleared, gripper in standby.")
    return state

