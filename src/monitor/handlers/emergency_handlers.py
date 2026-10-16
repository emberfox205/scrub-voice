"""
Emergency Intent Handlers for Scrub-Voice Monitor.
Implements halt_emergency and clear_emergency state transitions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.monitor.handlers.registry import register_intent
from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.handlers.emergency")


@register_intent("halt_emergency")
def handle_halt_emergency(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """Activates surgical emergency halt interlock, locking brakes."""
    state.emergency_active = True
    state.telemetry.status = "EMERGENCY HALT (LOCKED)"
    logger.critical("EMERGENCY INTERLOCK ACTIVATED: Robot motion suspended and brakes locked!")
    return state


@register_intent("clear_emergency")
def handle_clear_emergency(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Executes emergency evacuation: arm steers clear of surgical field (+Z vertical ascent)
    to allow immediate human intervention.
    """
    state.emergency_active = False
    state.telemetry.status = "EVACUATING CAVITY (+Z ASCENT)"
    # Evacuate surgical cavity with high-speed vertical ascent (+Z)
    state.telemetry.z_mm = max(state.telemetry.z_mm, 250.0)
    logger.warning("EMERGENCY CLEAR: Arm steering clear of surgical area (+Z ascent) for human intervention.")
    return state
