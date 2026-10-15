#!/usr/bin/env python3
"""
Unit and Integration Tests for MuJoCo UR-10e and RG-2 Gripper Simulation.
"""

from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.sim_ws.mujoco_controller import MujocoSimController


@pytest.fixture(scope="module")
def sim():
    """Initializes a headless simulation instance for testing."""
    controller = MujocoSimController(headless=True)
    yield controller
    controller.close_viewer()


def test_scene_loading(sim):
    """Verifies the MuJoCo model loads with expected degrees of freedom."""
    # 6 UR-10e arm joints + 6 RG-2 joints + 7 scalpel freejoint = 19
    assert sim.model.nq == 19
    assert sim.model.nu == 7  # 6 arm joints + 1 RG-2 gripper actuator
    assert len(sim.get_actual_q()) == 6


def test_joint_reading_and_forward_kinematics(sim):
    """Verifies joint angle retrieval and TCP pose computation."""
    q_rad = sim.get_actual_q()
    q_deg = sim.get_actual_q_deg()
    tcp = sim.get_actual_tcp_pose()

    assert len(q_rad) == 6
    assert len(q_deg) == 6
    assert len(tcp) == 6  # [X, Y, Z, Rx, Ry, Rz]
    # Check that TCP height Z is above the floor
    assert tcp[2] > 0.0


def test_gripper_actuation(sim):
    """Verifies RG-2 gripper open and close methods."""
    sim.open_gripper(duration=0.5)
    grip_q = float(sim.data.qpos[sim.gripper_joint_idx])
    assert grip_q > 0.3

    sim.close_gripper(duration=0.5)
    grip_q = float(sim.data.qpos[sim.gripper_joint_idx])
    assert grip_q < 0.1


def test_movej_and_pose_transitions(sim):
    """Verifies smooth arm joint motion to tray and back to home."""
    # Move to tray
    success = sim.move_to_tray(duration=1.2)
    assert success
    actual_q = sim.get_actual_q()
    for target, actual in zip(sim.tray_pose, actual_q):
        assert abs(target - actual) < 0.05

    # Move back to home
    success = sim.move_to_home(duration=1.2)
    assert success
    actual_q = sim.get_actual_q()
    for target, actual in zip(sim.home_pose, actual_q):
        assert abs(target - actual) < 0.05

