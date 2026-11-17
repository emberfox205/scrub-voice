#!/usr/bin/env python3
"""
Unit and Integration Tests for HAL Drop-In Simulation Shims and Factory.
Verifies SimRTDEControlInterface, SimRTDEReceiveInterface, SimRG2Gripper,
and singleton management.
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.hal import (
    SimRG2Gripper,
    SimRTDEControlInterface,
    SimRTDEReceiveInterface,
    get_gripper,
    get_robot_interfaces,
    reset_shared_sim,
)


@pytest.fixture(autouse=True)
def setup_and_teardown_sim():
    """Ensures a clean shared simulation instance for each test module."""
    yield
    reset_shared_sim()


def test_sim_rtde_control_interface():
    """Verifies SimRTDEControlInterface methods and contracts."""
    rtde_c = SimRTDEControlInterface()
    assert rtde_c.isConnected()

    # Spatial transform
    p_tcp = rtde_c._sim.get_actual_tcp_pose()
    p_offset = rtde_c.poseTrans(p_tcp, [0.0, 0.0, -0.05, 0.0, 0.0, 0.0])
    assert len(p_offset) == 6

    # Control lifecycle methods
    assert rtde_c.stopJ()
    assert rtde_c.stopL()
    assert rtde_c.reuploadScript()
    assert rtde_c.teachMode()
    assert rtde_c.endTeachMode()
    assert rtde_c.zeroFtSensor()
    assert rtde_c.speedJ([0.0] * 6)
    assert rtde_c.speedL([0.0] * 6)
    assert len(rtde_c.getTargetWaypoint()) == 6

    # Inverse kinematics
    ik_q = rtde_c.getInverseKinematics(p_tcp)
    assert len(ik_q) == 6

    rtde_c.stopScript()
    rtde_c.disconnect()
    assert not rtde_c.isConnected()


def test_sim_rtde_receive_interface():
    """Verifies SimRTDEReceiveInterface queries."""
    rtde_r = SimRTDEReceiveInterface()
    assert rtde_r.isConnected()

    q = rtde_r.getActualQ()
    assert len(q) == 6
    assert isinstance(q[0], float)

    tcp = rtde_r.getActualTCPPose()
    assert len(tcp) == 6
    assert isinstance(tcp[0], float)

    qd = rtde_r.getActualQd()
    assert len(qd) == 6

    tcp_force = rtde_r.getActualTCPForce()
    assert len(tcp_force) == 6

    tcp_speed = rtde_r.getActualTCPSpeed()
    assert len(tcp_speed) == 6

    target_q = rtde_r.getTargetQ()
    assert len(target_q) == 6

    target_tcp = rtde_r.getTargetTCPPose()
    assert len(target_tcp) == 6

    assert rtde_r.getRobotMode() == 7
    assert rtde_r.getSafetyMode() == 1
    assert not rtde_r.isProtectiveStopped()
    assert not rtde_r.isEmergencyStopped()

    rtde_r.disconnect()
    assert not rtde_r.isConnected()


def test_sim_rg2_gripper():
    """Verifies SimRG2Gripper actuation and state reporting."""
    gripper = SimRG2Gripper()

    # Move to 40 mm
    state = gripper.move(40.0, force_n=20.0)
    assert isinstance(state, dict)
    assert abs(state["width_mm"] - 40.0) < 5.0
    assert "status_raw" in state
    assert "busy" in state
    assert "grip_detected" in state

    # Property helpers
    assert isinstance(gripper.is_busy, bool)
    assert isinstance(gripper.is_grip_detected, bool)
    assert isinstance(gripper.width_mm, float)


def test_hal_singleton_sharing():
    """Verifies get_robot_interfaces and get_gripper share the exact same simulation instance."""
    rtde_c, rtde_r = get_robot_interfaces(sim=True)
    gripper = get_gripper(sim=True)

    assert isinstance(rtde_c, SimRTDEControlInterface)
    assert isinstance(rtde_r, SimRTDEReceiveInterface)
    assert isinstance(gripper, SimRG2Gripper)

    # Core assertion: Arm and Gripper share the same physics controller
    assert rtde_c._sim is rtde_r._sim
    assert rtde_c._sim is gripper._sim


def test_hal_env_var_sim_selection(monkeypatch):
    """Verifies setting ROBOT_BACKEND=sim automatically selects simulation backends."""
    monkeypatch.setenv("ROBOT_BACKEND", "sim")

    rtde_c, rtde_r = get_robot_interfaces()
    assert isinstance(rtde_c, SimRTDEControlInterface)
    assert isinstance(rtde_r, SimRTDEReceiveInterface)

    gripper = get_gripper()
    assert isinstance(gripper, SimRG2Gripper)


def test_physical_backend_import_error(monkeypatch):
    """Verifies informative error when physical ur-rtde is missing."""
    monkeypatch.setitem(sys.modules, "rtde_control", None)
    monkeypatch.setitem(sys.modules, "rtde_receive", None)
    with pytest.raises(ImportError, match="Physical ur-rtde package not installed"):
        get_robot_interfaces(robot_ip="192.168.1.105", sim=False)
