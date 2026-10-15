"""
Modular Intent Handler Registry for Scrub-Voice Monitor.
Provides a decorator-based registration pattern for dispatching validated NLU
commands into state updates.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict

from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.handlers")

HandlerFunc = Callable[[MonitorState, Dict[str, Any]], MonitorState]
HANDLERS: Dict[str, HandlerFunc] = {}


def register_intent(intent_name: str) -> Callable[[HandlerFunc], HandlerFunc]:
    """Decorator to register a function as the handler for an intent name."""
    def decorator(fn: HandlerFunc) -> HandlerFunc:
        if intent_name in HANDLERS:
            logger.warning("Overriding existing handler for intent '%s'", intent_name)
        HANDLERS[intent_name] = fn
        return fn
    return decorator


def handle_command(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Applies an incoming NLU command to the given state using registered handlers.
    Updates metadata (last_command, last_updated) and handles unhandled intents safely.
    Uses state's internal lock if available to ensure thread-safe mutation.
    """
    if not isinstance(command, dict):
        logger.warning("Rejected non-dict command payload: %s", type(command))
        return state

    lock = getattr(state, "_lock", None)
    if lock is not None:
        with lock:
            return _apply_command(state, command)
    return _apply_command(state, command)


def _apply_command(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    intent = command.get("intent", "unknown")
    status = command.get("status", "EXECUTE")

    # Only EXECUTE status commands modify subsystem state
    if status == "EXECUTE" and intent in HANDLERS:
        handler = HANDLERS[intent]
        try:
            state = handler(state, command)
        except Exception as e:
            logger.error("Error executing handler for intent '%s': %s", intent, e, exc_info=True)
    elif status != "EXECUTE":
        logger.info("Ignoring non-EXECUTE status command: intent=%s, status=%s", intent, status)
    else:
        logger.debug("No custom handler registered for intent '%s'", intent)

    state.last_command = command
    state.last_updated = time.time()
    return state
