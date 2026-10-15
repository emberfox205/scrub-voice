"""
Scan and Viewport Intent Handlers for Scrub-Voice Monitor.
Implements open_scan, close_scan, zoom_display, and navigate_slice with
strict parameter clamping and validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.monitor.handlers.registry import register_intent
from src.monitor.state import MonitorState

logger = logging.getLogger("monitor.handlers.scan")

VALID_SCAN_TYPES = {"ct", "mri", "xray"}
MIN_ZOOM = 1.0
MAX_ZOOM = 10.0
DEFAULT_ZOOM_FACTOR = 2.0


@register_intent("open_scan")
def handle_open_scan(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Opens diagnostic imaging scan (CT, MRI, or X-ray).
    Resets current slice to 1 and zoom factor to 1.0x.
    """
    slots = command.get("slots", {})
    raw_scan_type = slots.get("scan_type")
    if raw_scan_type is not None:
        scan_type = str(raw_scan_type).strip().lower()
        if scan_type in VALID_SCAN_TYPES:
            state.active_scan = scan_type
            state.current_slice = 1
            state.zoom_factor = 1.0
            logger.info("Opened scan modality: %s", scan_type)
        else:
            logger.warning("Rejected invalid scan_type: %r (valid: %s)", raw_scan_type, VALID_SCAN_TYPES)
    else:
        logger.warning("open_scan missing scan_type in slots")

    return state


@register_intent("close_scan")
def handle_close_scan(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Closes the active scan.
    If scan_type is specified in slots, closes only if it matches current active_scan.
    If scan_type is omitted or empty, closes whatever scan is active.
    """
    slots = command.get("slots", {})
    target_scan = slots.get("scan_type")

    if not target_scan or str(target_scan).strip().lower() == state.active_scan:
        state.reset_scan()
        logger.info("Closed active scan viewport.")
    else:
        logger.info(
            "Ignored close_scan for '%s' because active scan is '%s'",
            target_scan,
            state.active_scan,
        )

    return state


@register_intent("zoom_display")
def handle_zoom_display(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Magnifies or reduces viewport zoom factor.
    Clamps magnification factor between 1.0x and 10.0x.
    """
    slots = command.get("slots", {})
    direction = str(slots.get("direction", "IN")).upper()

    try:
        factor = float(slots.get("factor", DEFAULT_ZOOM_FACTOR))
        # Ensure factor is strictly positive
        if factor <= 0:
            factor = DEFAULT_ZOOM_FACTOR
    except (ValueError, TypeError):
        factor = DEFAULT_ZOOM_FACTOR

    if direction == "OUT":
        new_zoom = state.zoom_factor / factor
    else:  # Defaults to "IN"
        new_zoom = state.zoom_factor * factor

    state.zoom_factor = round(max(MIN_ZOOM, min(MAX_ZOOM, new_zoom)), 2)
    logger.info("Updated zoom_factor: %.2fx (direction=%s, factor=%.2f)", state.zoom_factor, direction, factor)
    return state


@register_intent("navigate_slice")
def handle_navigate_slice(state: MonitorState, command: Dict[str, Any]) -> MonitorState:
    """
    Navigates through the cross-sectional slice stack.
    Supports either direct slice_index or relative direction (NEXT/PREVIOUS) with step_count.
    Slice numbers are clamped to minimum 1.
    """
    slots = command.get("slots", {})

    if slots.get("slice_index") is not None:
        try:
            target_slice = int(slots["slice_index"])
            state.current_slice = max(1, target_slice)
            logger.info("Direct slice navigation to index: %d", state.current_slice)
        except (ValueError, TypeError):
            logger.warning("Invalid slice_index in slots: %r", slots["slice_index"])
    elif "direction" in slots and slots.get("direction") is not None:
        direction = str(slots["direction"]).upper()
        try:
            step = max(1, int(slots.get("step_count", 1)))
        except (ValueError, TypeError):
            step = 1

        if direction == "PREVIOUS":
            state.current_slice = max(1, state.current_slice - step)
        else:  # NEXT or default
            state.current_slice += step

        logger.info("Relative slice navigation: direction=%s, step=%d -> new_slice=%d", direction, step, state.current_slice)
    else:
        logger.warning("navigate_slice missing both slice_index and direction in slots: %s", slots)

    return state
