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
    """Activates surgical emergency halt interlock flag (actuator telemetry remains decoupled)."""
    state.emergency_active = True
    logger.critical("EMERGENCY INTERLOCK ACTIVATED: Emergency halt engaged!")
    return state


@register_intent("clear_emergency")
def handle_clear_emergency(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """Clears surgical emergency halt interlock flag (actuator telemetry remains decoupled)."""
    state.emergency_active = False
    logger.warning("EMERGENCY CLEAR: Emergency interlock disengaged.")
    return state
