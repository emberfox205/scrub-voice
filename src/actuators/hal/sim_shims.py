#!/usr/bin/env python3
"""
Hardware Abstraction Layer (HAL) Drop-In Simulation Shims for UR-10e and OnRobot RG-2.

Provides drop-in replacements for `ur_rtde.RTDEControlInterface`,
`ur_rtde.RTDEReceiveInterface`, and `RG2Gripper` that interface with
the in-memory `MujocoSimController` physics engine.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from src.actuators.sim_ws.mujoco_controller import MujocoSimController, rotmat_to_rotvec

# Thread-safe simulation singleton
_SHARED_SIM_LOCK = threading.Lock()
_SHARED_SIM: Optional[MujocoSimController] = None


def get_shared_sim(headless: Optional[bool] = None) -> MujocoSimController:
    """
    Returns the shared in-memory MujocoSimController singleton.
    Guarantees arm motion and gripper commands operate within the same physical scene.
    Automatically falls back to headless execution if no display is detected.
    """
    global _SHARED_SIM
    with _SHARED_SIM_LOCK:
        if _SHARED_SIM is None or not _SHARED_SIM.is_running():
            in_test = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            effective_headless = True if in_test else (not has_display if headless is None else headless)
            _SHARED_SIM = MujocoSimController(headless=effective_headless)
            if not effective_headless:
                _SHARED_SIM.start_viewer()
        return _SHARED_SIM


def reset_shared_sim() -> None:
    """Terminates and resets the shared simulation controller instance."""
    global _SHARED_SIM
    with _SHARED_SIM_LOCK:
        if _SHARED_SIM is not None:
            _SHARED_SIM.close_viewer()
            _SHARED_SIM = None


class SimRTDEControlInterface:
    """
    Drop-in simulation shim matching `rtde_control.RTDEControlInterface`.
    Delegates commands to the underlying MujocoSimController.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        sim_controller: Optional[MujocoSimController] = None,
        headless: Optional[bool] = None,
    ):
        self.ip = ip
        self._sim = sim_controller if sim_controller is not None else get_shared_sim(headless=headless)
        self._connected = True

    def isConnected(self) -> bool:
        """Returns True if the simulation connection is active."""
        return self._connected and self._sim.is_running()

    def moveJ(
        self,
        q: List[float],
        speed: float = 0.3,
        acceleration: float = 0.4,
        asynchronous: bool = False,
    ) -> bool:
        """Moves robot joints via smooth S-curve interpolation."""
        return self._sim.moveJ(q, speed=speed, acceleration=acceleration, wait=not asynchronous)

    def moveL(
        self,
        pose: List[float],
        speed: float = 0.05,
        acceleration: float = 0.1,
        asynchronous: bool = False,
    ) -> bool:
        """Moves TCP linearly in Cartesian space using differential IK."""
        return self._sim.moveL(pose, speed=speed, acceleration=acceleration, wait=not asynchronous)

    def poseTrans(self, p_from: List[float], p_offset: List[float]) -> List[float]:
        """Calculates spatial transform T_result = T_from * T_offset."""
        return self._sim.poseTrans(p_from, p_offset)

    def stopJ(self, a: float = 2.0) -> bool:
        """Stops joint motion safely and holds current position."""
        return self._sim.stop()

    def stopL(self, a: float = 2.0) -> bool:
        """Stops linear motion safely and holds current position."""
        return self._sim.stop()

    def stopScript(self) -> None:
        """Simulates stopping robot script execution."""
        self._sim.stop()

    def reuploadScript(self) -> bool:
        """Simulates re-uploading control script."""
        return True

    def teachMode(self) -> bool:
        """Simulates enabling teach/freedrive mode."""
        return True

    def endTeachMode(self) -> bool:
        """Simulates disabling teach/freedrive mode."""
        return True

    def zeroFtSensor(self) -> bool:
        """Simulates zeroing the force-torque sensor."""
        if hasattr(self._sim, "_simulated_tcp_force"):
            self._sim._simulated_tcp_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return True

    def speedJ(self, qd: List[float], acceleration: float = 0.5, time: float = 0.0) -> bool:
        """Simulates joint speed command."""
        return True

    def speedL(self, xd: List[float], acceleration: float = 0.5, time: float = 0.0) -> bool:
        """Simulates linear speed command."""
        return True

    def getInverseKinematics(
        self,
        x: List[float],
        qnear: Optional[List[float]] = None,
        max_position_error: float = 1e-3,
        max_orientation_error: float = 1e-3,
    ) -> List[float]:
        """Calculates inverse kinematics solution matching rtde_c contract."""
        return self._sim.get_inverse_kinematics(
            x, qnear=qnear, pos_tol=max_position_error, rot_tol=max_orientation_error
        )

    def getTargetWaypoint(self) -> List[float]:
        """Returns the current target waypoint TCP pose."""
        return self._sim.get_actual_tcp_pose()

    def isMoving(self) -> bool:
        """Returns True if the arm is currently executing a motion."""
        return hasattr(self._sim, "is_moving") and self._sim.is_moving()

    def getAsyncOperationProgress(self) -> int:
        """Returns 1 if idle, 0 if motion in progress."""
        if hasattr(self._sim, "is_moving") and self._sim.is_moving():
            return 0
        return 1

    def disconnect(self) -> None:
        """Simulates disconnecting control interface."""
        self._connected = False


class SimRTDEReceiveInterface:
    """
    Drop-in simulation shim matching `rtde_receive.RTDEReceiveInterface`.
    Queries live states from the underlying MujocoSimController.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        sim_controller: Optional[MujocoSimController] = None,
        headless: Optional[bool] = None,
    ):
        self.ip = ip
        self._sim = sim_controller if sim_controller is not None else get_shared_sim(headless=headless)
        self._connected = True

    def isConnected(self) -> bool:
        """Returns True if the simulation connection is active."""
        return self._connected and self._sim.is_running()

    def getActualQ(self) -> List[float]:
        """Returns actual joint positions in radians."""
        return self._sim.get_actual_q()

    def getActualTCPPose(self) -> List[float]:
        """Returns actual TCP pose [X, Y, Z, Rx, Ry, Rz]."""
        return self._sim.get_actual_tcp_pose()

    def getActualQd(self) -> List[float]:
        """Returns actual joint velocities in rad/s using dofadr."""
        if hasattr(self._sim.data, "qvel"):
            return [float(self._sim.data.qvel[idx]) for idx in self._sim.arm_dof_indices]
        return [0.0] * 6

    def getActualTCPForce(self) -> List[float]:
        """Returns simulated TCP forces [Fx, Fy, Fz, Mx, My, Mz]."""
        if hasattr(self._sim, "_simulated_tcp_force"):
            return list(self._sim._simulated_tcp_force)
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_simulated_tcp_force(self, force_vector: List[float]) -> None:
        """Sets simulated TCP force vector for test injection."""
        if hasattr(self._sim, "_simulated_tcp_force"):
            self._sim._simulated_tcp_force = list(force_vector)

    def getActualTCPSpeed(self) -> List[float]:
        """Returns simulated TCP speed vector [Vx, Vy, Vz, Wx, Wy, Wz]."""
        qd = self.getActualQd()
        j = self._sim._compute_tcp_jacobian(self.getActualQ())
        tcp_vel = j @ np.array(qd, dtype=float)
        return [float(v) for v in tcp_vel]

    def getTargetQ(self) -> List[float]:
        """Returns target joint positions in radians."""
        return [float(self._sim.data.ctrl[idx]) for idx in self._sim.arm_actuator_ids]

    def getTargetTCPPose(self) -> List[float]:
        """Returns target TCP pose."""
        pos, mat = self._sim._compute_fk(self.getTargetQ())
        rot_vec = rotmat_to_rotvec(mat)
        return [float(pos[0]), float(pos[1]), float(pos[2])] + [float(v) for v in rot_vec]

    def getRobotMode(self) -> int:
        """Returns 7 (RUNNING) matching physical UR robot state."""
        return 7

    def getSafetyMode(self) -> int:
        """Returns 1 (NORMAL) matching physical UR robot safety state."""
        return 1

    def isProtectiveStopped(self) -> bool:
        """Returns False in simulation."""
        return False

    def isEmergencyStopped(self) -> bool:
        """Returns False in simulation."""
        return False

    def disconnect(self) -> None:
        """Simulates disconnecting receive interface."""
        self._connected = False


class SimRG2Gripper:
    """
    Drop-in simulation shim matching `RG2Gripper` (Modbus TCP).
    Delegates commands to the underlying MujocoSimController.
    """

    MAX_WIDTH_MM = 110.0
    MIN_WIDTH_MM = 0.0
    MAX_FORCE_N = 40.0
    MIN_FORCE_N = 3.0

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 502,
        sim_controller: Optional[MujocoSimController] = None,
        headless: Optional[bool] = None,
    ):
        self.ip = f"{ip} (MuJoCo Simulation)"
        self.port = port
        self._sim = sim_controller if sim_controller is not None else get_shared_sim(headless=headless)

    def get_state(self) -> Dict[str, Any]:
        """Queries current gripper state."""
        return self._sim.get_gripper_state()

    def move(
        self,
        width_mm: float,
        force_n: float = 20.0,
        wait_for_idle: bool = True,
    ) -> Dict[str, Any]:
        """Moves gripper fingers to target width in mm with force limit."""
        return self._sim.move_gripper(width_mm, force_n=force_n, wait=wait_for_idle)

    def open(self, force_n: float = 20.0) -> Dict[str, Any]:
        """Opens gripper fully."""
        return self.move(self.MAX_WIDTH_MM, force_n=force_n)

    def close(self, force_n: float = 20.0) -> Dict[str, Any]:
        """Closes gripper fully."""
        return self.move(self.MIN_WIDTH_MM, force_n=force_n)

    @property
    def is_busy(self) -> bool:
        return bool(self.get_state()["busy"])

    @property
    def is_grip_detected(self) -> bool:
        return bool(self.get_state()["grip_detected"])

    @property
    def width_mm(self) -> float:
        return float(self.get_state()["width_mm"])
