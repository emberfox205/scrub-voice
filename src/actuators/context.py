#!/usr/bin/env python3
"""
Actuator Context & Runtime State Management for Scrub-Voice.

Maintains persistent HAL interface handles (UR10e RTDE and OnRobot RG-2),
calibrated waypoint configurations, operational modes, and the runtime state
machine (IDLE, BUSY, FAULT) with thread-safe execution gating.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WAYPOINTS_PATH = Path(__file__).resolve().parent / "waypoints.json"

logger = logging.getLogger("actuators.context")


class RobotState(str, Enum):
    """Operational lifecycle states of the robot actuator subsystem."""
    IDLE = "IDLE"    # Connected, stationary at home pose, ready for commands
    BUSY = "BUSY"    # Actively executing a motion (pick_tool, park_arm)
    FAULT = "FAULT"  # Task failed (vision miss, grip unconfirmed, trajectory error)


class OperationalMode(str, Enum):
    """Functional operational modes for the multi-role UR10e arm."""
    TOOL_HANDLING = "TOOL_HANDLING"  # Active: Tray retrieval, RG-2 gripper, handover
    CAMERA_LIGHT = "CAMERA_LIGHT"    # Deferred: Endoscope wand holder, Cartesian jogging


class ActuatorContext:
    """
    Thread-safe context managing persistent robot connections, waypoints,
    and state transitions across successive commands.
    """

    def __init__(
        self,
        robot_ip: Optional[str] = None,
        gripper_ip: Optional[str] = None,
        sim: bool = False,
        headless: Optional[bool] = None,
        waypoints_path: Optional[Path] = None,
        rtde_c: Optional[Any] = None,
        rtde_r: Optional[Any] = None,
        gripper: Optional[Any] = None,
    ) -> None:
        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.sim = sim
        self.headless = headless
        self.waypoints_path = Path(waypoints_path or DEFAULT_WAYPOINTS_PATH)

        self._lock = threading.RLock()
        self.state: RobotState = RobotState.IDLE
        self.mode: OperationalMode = OperationalMode.TOOL_HANDLING
        self.current_task: Optional[str] = None
        self.held_tool: Optional[str] = None
        self.last_error: Optional[str] = None

        # Load waypoints
        self.waypoints = self._load_waypoints(self.waypoints_path)

        # Persistent HAL handles (lazy or injected)
        if rtde_c is not None and rtde_r is not None and gripper is not None:
            self.rtde_c = rtde_c
            self.rtde_r = rtde_r
            self.gripper = gripper
        else:
            self._init_hal()

        logger.info(
            "ActuatorContext initialized (Mode: %s, State: %s, Sim: %s)",
            self.mode.value, self.state.value, self.sim
        )

    def _load_waypoints(self, path: Path) -> Dict[str, Any]:
        """Loads and validates calibrated waypoints JSON."""
        if not path.exists():
            raise FileNotFoundError(f"Waypoints file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = ["home_pose", "tray_pose", "handover_pose"]
        for k in required:
            if k not in data or "joints_rad" not in data[k]:
                raise ValueError(f"Malformed waypoints file {path}: missing '{k}'")
        return data

    def _init_hal(self) -> None:
        """Initializes persistent UR10e RTDE and RG-2 gripper interfaces."""
        from src.actuators.hal import get_gripper, get_robot_interfaces

        logger.info("Initializing persistent HAL interfaces (sim=%s)...", self.sim)
        self.rtde_c, self.rtde_r = get_robot_interfaces(
            robot_ip=self.robot_ip,
            sim=self.sim,
            headless=self.headless,
        )
        self.gripper = get_gripper(
            gripper_ip=self.gripper_ip,
            sim=self.sim,
            headless=self.headless,
        )

    def can_execute(self, intent: str) -> Tuple[bool, Optional[str]]:
        """
        Validates whether current operational mode and state allow executing intent.
        Returns (is_allowed, rejection_reason).
        """
        with self._lock:
            # 1. Operational Mode Safety Gate
            if self.mode != OperationalMode.TOOL_HANDLING and intent in ("pick_tool", "park_arm"):
                return False, f"Robot is in {self.mode.value} mode; cannot execute {intent} without switching modes."

            # 2. Busy Check
            if self.state == RobotState.BUSY:
                return False, f"Robot is BUSY executing task '{self.current_task}'; ignoring command."

            # 3. Fault Check: only park_arm is permitted for recovery
            if self.state == RobotState.FAULT:
                if intent == "park_arm":
                    return True, None
                return False, f"Robot in FAULT state: {self.last_error}. Execute park_arm to reset."

            # 4. Idle Check
            if self.state == RobotState.IDLE:
                return True, None

            return False, f"Unknown or invalid robot state: {self.state.value}"

    def transition_to(
        self,
        new_state: RobotState,
        task_name: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Thread-safe state transition with logging."""
        with self._lock:
            old_state = self.state
            self.state = new_state
            if new_state == RobotState.BUSY:
                self.current_task = task_name
            elif new_state == RobotState.IDLE:
                self.current_task = None
                self.last_error = None
            elif new_state == RobotState.FAULT:
                self.current_task = None
                self.last_error = error or "Unspecified task failure"

            logger.info(
                "Robot state transition: %s -> %s (task: %s, error: %s)",
                old_state.value, new_state.value, self.current_task, self.last_error
            )

    def switch_mode(self, target_mode: OperationalMode) -> Tuple[bool, str]:
        """
        Safely switches operational mode if gripper is empty and arm is not busy.
        """
        with self._lock:
            if self.held_tool is not None:
                return False, f"Cannot switch mode: surgical tool '{self.held_tool}' held. Park arm first."
            if self.state == RobotState.BUSY:
                return False, f"Cannot switch mode: robot is BUSY executing '{self.current_task}'."

            old_mode = self.mode
            self.mode = target_mode
            logger.info("Operational mode changed: %s -> %s", old_mode.value, target_mode.value)
            return True, f"Mode switched to {target_mode.value}"

    def get_status(self) -> Dict[str, Any]:
        """Returns deep snapshot dictionary of internal actuator state."""
        with self._lock:
            return {
                "state": self.state.value,
                "mode": self.mode.value,
                "current_task": self.current_task,
                "held_tool": self.held_tool,
                "last_error": self.last_error,
                "is_busy": self.state == RobotState.BUSY,
                "timestamp": round(time.time(), 3),
            }

    def close(self) -> None:
        """Releases HAL handles cleanly on service shutdown."""
        with self._lock:
            logger.info("Closing ActuatorContext connections...")
            if self.sim:
                try:
                    from src.actuators.hal import reset_shared_sim
                    reset_shared_sim()
                except Exception as e:
                    logger.debug("reset_shared_sim on close: %s", e)
