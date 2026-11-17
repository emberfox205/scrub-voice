#!/usr/bin/env python3
"""
Hardware Abstraction Layer (HAL) for UR-10e Robot Arm and OnRobot RG-2 Gripper.

Provides unified factory functions `get_robot_interfaces` and `get_gripper`
that transparently switch between physical hardware (ur_rtde, Modbus TCP) and
high-fidelity MuJoCo simulation based on configuration or command-line flags.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from src.actuators.hal.sim_shims import (
    SimRG2Gripper,
    SimRTDEControlInterface,
    SimRTDEReceiveInterface,
    get_shared_sim,
    reset_shared_sim,
)

__all__ = [
    "get_robot_interfaces",
    "get_gripper",
    "get_shared_sim",
    "reset_shared_sim",
    "SimRTDEControlInterface",
    "SimRTDEReceiveInterface",
    "SimRG2Gripper",
]


def _is_sim_mode(explicit_sim: bool, target_ip: Optional[str]) -> bool:
    """Evaluates whether simulation mode is requested."""
    if explicit_sim:
        return True
    if target_ip and target_ip.strip().lower() in ("sim", "simulation", "mujoco"):
        return True
    env_backend = os.getenv("ROBOT_BACKEND", "").strip().lower()
    if env_backend in ("sim", "simulation", "mujoco"):
        return True
    return (
        os.getenv("ROBOT_SIM", "").strip().lower() in ("1", "true", "yes")
        or os.getenv("USE_SIM", "").strip().lower() in ("1", "true", "yes")
    )


def get_robot_interfaces(
    robot_ip: Optional[str] = None,
    sim: bool = False,
    headless: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """
    Factory function returning (RTDEControlInterface, RTDEReceiveInterface).

    Args:
        robot_ip: IP address of UR-10e robot or 'sim'.
        sim: True to force MuJoCo simulation backend.
        headless: Optional headless simulation override (if None, auto-detects display).

    Returns:
        (rtde_c, rtde_r) matching physical ur_rtde interface contracts.
    """
    if _is_sim_mode(sim, robot_ip):
        sim_ctrl = get_shared_sim(headless=headless)
        rtde_c = SimRTDEControlInterface(ip=robot_ip or "127.0.0.1", sim_controller=sim_ctrl, headless=headless)
        rtde_r = SimRTDEReceiveInterface(ip=robot_ip or "127.0.0.1", sim_controller=sim_ctrl, headless=headless)
        return rtde_c, rtde_r

    # Physical robot backend
    target_ip = robot_ip or os.getenv("ROBOT_IP", "192.168.1.105")
    try:
        import rtde_control
        import rtde_receive
    except ImportError as e:
        raise ImportError(
            f"Physical ur-rtde package not installed: {e}. "
            "Pass `sim=True` or set `ROBOT_BACKEND=sim` to run against MuJoCo simulation."
        ) from e

    rtde_c = rtde_control.RTDEControlInterface(target_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(target_ip)
    return rtde_c, rtde_r


def get_gripper(
    gripper_ip: Optional[str] = None,
    sim: bool = False,
    headless: Optional[bool] = None,
) -> Any:
    """
    Factory function returning RG-2 gripper controller interface.

    Args:
        gripper_ip: IP address of OnRobot Compute Box or 'sim'.
        sim: True to force MuJoCo simulation backend.
        headless: Optional headless simulation override (if None, auto-detects display).

    Returns:
        Gripper controller matching RG2Gripper contract.
    """
    if _is_sim_mode(sim, gripper_ip):
        sim_ctrl = get_shared_sim(headless=headless)
        return SimRG2Gripper(ip=gripper_ip or "127.0.0.1", sim_controller=sim_ctrl, headless=headless)

    # Physical gripper backend
    from src.actuators.sim_ws.rg2_control import RG2Gripper

    target_ip = gripper_ip or os.getenv("GRIPPER_IP", "192.168.1.101")
    return RG2Gripper(ip=target_ip)
