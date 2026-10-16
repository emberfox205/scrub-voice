"""
Event and EMR Audit Logging Intent Handlers for Scrub-Voice Monitor.
Implements log_event to record intraoperative surgical milestones.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.monitor.handlers.registry import register_intent
from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.handlers.event")

MAX_MILESTONES = 50


@register_intent("log_event")
def handle_log_event(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Appends an intraoperative surgical milestone to recent_milestones.
    Validates event_name presence and limits storage size to MAX_MILESTONES.
    """
    slots = command.get("slots", {})
    raw_event = slots.get("event_name")
    if raw_event is None:
        logger.warning("Rejected log_event with missing event_name slot")
        return state

    event_name = str(raw_event).strip()
    if not event_name:
        logger.warning("Rejected log_event with empty event_name: %r", raw_event)
        return state

    raw_comment = slots.get("comment")
    comment = str(raw_comment).strip() if raw_comment is not None else ""
    cmd_id = str(command.get("command_id", "cmd_manual"))
    timestamp = float(command.get("timestamp", time.time()))

    milestone_record = {
        "event_name": event_name,
        "comment": comment,
        "timestamp": timestamp,
        "command_id": cmd_id,
    }

    state.recent_milestones.append(milestone_record)
    if len(state.recent_milestones) > MAX_MILESTONES:
        state.recent_milestones.pop(0)

    logger.info("Recorded surgical milestone [%s]: %s (notes: %s)", cmd_id, event_name, comment)
    return state
