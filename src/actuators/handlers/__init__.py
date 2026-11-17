#!/usr/bin/env python3
"""
Actuator Intent Handlers Package.
Exports handler registry and imports handler modules to trigger registration.
"""

from src.actuators.handlers.registry import (
    ACTUATOR_HANDLERS,
    handle_actuator_command,
    register_actuator_intent,
)
from src.actuators.handlers import tool_handlers

__all__ = [
    "ACTUATOR_HANDLERS",
    "handle_actuator_command",
    "register_actuator_intent",
    "tool_handlers",
]
