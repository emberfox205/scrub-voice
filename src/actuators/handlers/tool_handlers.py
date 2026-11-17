#!/usr/bin/env python3
"""
Tool Handling Intent Handlers for Scrub-Voice Actuators.

Executes physical/simulated robot tool retrieval and arm parking via
ToolRetrievalPipeline and persistent ActuatorContext HAL interfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.actuators.context import ActuatorContext, RobotState
from src.actuators.handlers.registry import register_actuator_intent
from src.actuators.pipeline import ToolRetrievalPipeline

logger = logging.getLogger("actuators.handlers.tool")

VALID_TOOLS = {"scalpel", "scissors", "hemostat", "forceps"}
TOOL_ALIASES = {
    "knife": "scalpel",
    "blade": "scalpel",
    "clamp": "hemostat",
    "tweezer": "forceps",
    "tweezers": "forceps",
    "pickup": "forceps",
}


@register_actuator_intent("pick_tool")
def handle_pick_tool(context: ActuatorContext, command: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes surgical tool pick and handover workflow.
    Validates tool slot, transitions to BUSY, invokes ToolRetrievalPipeline,
    and updates context state to IDLE on success or FAULT on failure.
    """
    slots = command.get("slots", {})
    raw_tool = slots.get("tool_identifier")
    if not raw_tool:
        logger.warning("Rejected pick_tool: missing 'tool_identifier' slot")
        return {
            "success": False,
            "rejected": True,
            "reason": "Missing required slot: tool_identifier",
        }

    norm_tool = str(raw_tool).strip().lower()
    tool = TOOL_ALIASES.get(norm_tool, norm_tool)

    if tool not in VALID_TOOLS:
        logger.warning("Rejected pick_tool: unknown tool '%s' (raw: '%s')", tool, raw_tool)
        return {
            "success": False,
            "rejected": True,
            "reason": f"Unknown tool '{norm_tool}'. Valid tools: {sorted(VALID_TOOLS)}",
        }

    logger.info("Starting pick_tool execution for '%s'...", tool)
    context.transition_to(RobotState.BUSY, task_name=f"pick_{tool}")

    # Determine execution options
    source = command.get("source", "sim" if context.sim else "realsense")
    sim_auto_pull = command.get("sim_auto_pull", True if context.sim else False)
    handover_timeout = float(command.get("handover_timeout", 60.0 if not context.sim else 5.0))
    speed = float(command.get("speed", 0.2))

    try:
        mission_res = ToolRetrievalPipeline.execute(
            target=tool,
            context=context,
            source=source,
            speed=speed,
            sim_auto_pull=sim_auto_pull,
            handover_timeout=handover_timeout,
        )

        if mission_res.success:
            context.held_tool = tool
            context.transition_to(RobotState.IDLE)
            logger.info("Successfully executed pick_tool for '%s' (%.2fs)", tool, mission_res.elapsed_time_s)
            return {
                "success": True,
                "intent": "pick_tool",
                "tool": tool,
                "phase": mission_res.phase,
                "message": mission_res.message,
                "elapsed_time_s": mission_res.elapsed_time_s,
            }
        else:
            context.transition_to(RobotState.FAULT, task_name=f"pick_{tool}", error=mission_res.message)
            logger.error("pick_tool failed for '%s' in phase '%s': %s", tool, mission_res.phase, mission_res.message)
            return {
                "success": False,
                "intent": "pick_tool",
                "tool": tool,
                "phase": mission_res.phase,
                "error": mission_res.message,
                "elapsed_time_s": mission_res.elapsed_time_s,
            }

    except Exception as e:
        context.transition_to(RobotState.FAULT, task_name=f"pick_{tool}", error=str(e))
        logger.error("Unexpected exception during pick_tool for '%s': %s", tool, e, exc_info=True)
        return {
            "success": False,
            "intent": "pick_tool",
            "tool": tool,
            "error": str(e),
        }


@register_actuator_intent("park_arm")
def handle_park_arm(context: ActuatorContext, command: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parks arm at calibrated home pose and closes gripper fingers.
    Allows recovery from FAULT state back to IDLE. Clears held_tool.
    """
    logger.info("Starting park_arm execution (recovery/home transit)...")
    context.transition_to(RobotState.BUSY, task_name="park_arm")

    speed = float(command.get("speed", 0.2))
    try:
        home_joints = context.waypoints["home_pose"]["joints_rad"]
        context.rtde_c.moveJ(home_joints, speed=speed, acceleration=0.25)
        context.gripper.close(force_n=20.0)

        context.held_tool = None
        context.transition_to(RobotState.IDLE)
        logger.info("Successfully parked arm at home pose; state restored to IDLE")
        return {
            "success": True,
            "intent": "park_arm",
            "message": "Arm parked at home pose; gripper closed.",
        }
    except Exception as e:
        context.transition_to(RobotState.FAULT, task_name="park_arm", error=str(e))
        logger.error("Unexpected exception during park_arm: %s", e, exc_info=True)
        return {
            "success": False,
            "intent": "park_arm",
            "error": str(e),
        }
