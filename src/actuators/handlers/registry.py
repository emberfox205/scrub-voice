#!/usr/bin/env python3
"""
Modular Intent Handler Registry for Scrub-Voice Actuators.

Provides a decorator-based registration pattern for dispatching validated
NLU commands from the ZeroMQ bus to physical/simulated robot motion handlers,
with strict state machine safety gating.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.actuators.context import ActuatorContext, RobotState

logger = logging.getLogger("actuators.handlers")

HandlerFunc = Callable[[ActuatorContext, Dict[str, Any]], Any]
ACTUATOR_HANDLERS: Dict[str, HandlerFunc] = {}


def register_actuator_intent(intent_name: str) -> Callable[[HandlerFunc], HandlerFunc]:
    """Decorator to register a function as the actuator handler for an intent name."""
    def decorator(fn: HandlerFunc) -> HandlerFunc:
        if intent_name in ACTUATOR_HANDLERS:
            logger.warning("Overriding existing actuator handler for intent '%s'", intent_name)
        ACTUATOR_HANDLERS[intent_name] = fn
        return fn
    return decorator


def handle_actuator_command(context: ActuatorContext, command: Dict[str, Any]) -> Any:
    """
    Applies an incoming NLU command to the robot via registered actuator handlers.
    Enforces status validation, safety gating (IDLE/BUSY/FAULT), and error containment.
    """
    if not isinstance(command, dict):
        logger.warning("Rejected non-dict command payload: %s", type(command))
        return {"success": False, "rejected": True, "reason": "Malformed command payload"}

    intent = command.get("intent", "unknown")
    status = command.get("status", "EXECUTE")
    cmd_id = command.get("command_id", "cmd_unknown")

    # 1. Non-EXECUTE status guard
    if status != "EXECUTE":
        logger.info("Ignoring non-EXECUTE command [%s]: intent=%s, status=%s", cmd_id, intent, status)
        return {"success": False, "rejected": True, "reason": f"Status '{status}' is not EXECUTE"}

    # 2. Registered handler check
    if intent not in ACTUATOR_HANDLERS:
        logger.warning("No actuator handler registered for intent '%s'", intent)
        return {"success": False, "rejected": True, "reason": f"Unregistered intent '{intent}'"}

    # 3. State Machine Safety Gating
    can_run, reason = context.can_execute(intent)
    if not can_run:
        logger.warning("Safety gate rejected [%s] intent '%s': %s", cmd_id, intent, reason)
        return {"success": False, "rejected": True, "reason": reason}

    # 4. Handler Execution
    handler = ACTUATOR_HANDLERS[intent]
    try:
        logger.info("Dispatching [%s] intent '%s' to actuator handler...", cmd_id, intent)
        result = handler(context, command)
        return result
    except Exception as e:
        logger.error("Exception in actuator handler for intent '%s': %s", intent, e, exc_info=True)
        context.transition_to(RobotState.FAULT, error=str(e))
        return {"success": False, "error": str(e), "intent": intent}
