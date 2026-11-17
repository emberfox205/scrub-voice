#!/usr/bin/env python3
# ==============================================================================
# HELPER / RUN COMMANDS:
#   # 1. Run MuJoCo simulation tests via pytest:
#   pytest test/test_mujoco_sim.py -v
#
#   # 2. Or run via unittest / pytest:
#   python -m pytest test/test_mujoco_sim.py
# ==============================================================================
"""
Unit and Integration Tests for MuJoCo UR-10e and RG-2 Gripper Simulation.
Tests kinematics, trajectory generation (moveJ, moveL), spatial transforms (poseTrans),
millimeter-based RG-2 gripper control, and contact sensing.
"""

import sys
import time
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.sim_ws.mujoco_controller import MujocoSimController, pose_trans


@pytest.fixture(scope="module")
def sim():
    """Initializes a headless simulation instance for testing."""
    controller = MujocoSimController(headless=True)
    yield controller
    controller.close_viewer()


def test_scene_loading(sim):
    """Verifies the MuJoCo model loads with expected degrees of freedom."""
    # 6 UR-10e arm joints + 6 RG-2 joints + 7 scalpel freejoint + 7 scissors freejoint = 26
    assert sim.model.nq == 26
    assert sim.model.nv == 24
    assert sim.model.nu == 7  # 6 arm joints + 1 RG-2 gripper actuator
    assert len(sim.get_actual_q()) == 6


def test_wrist_camera_frame_capture(sim):
    """Verifies eye-in-hand simulated RealSense frame capture."""
    sim.moveJ(sim.tray_pose, duration=0.5)
    color, depth, intrinsics = sim.capture_camera_frame("wrist_realsense")

    assert color.shape == (480, 640, 3)
    assert color.dtype == np.uint8
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32

    # Verify optical intrinsics
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert abs(intrinsics.cx - 320.0) < 1.0
    assert abs(intrinsics.cy - 240.0) < 1.0
    assert intrinsics.fx > 300.0 and intrinsics.fy > 300.0

    # Verify depth values are reasonable for eye-in-hand tray view (between 0.04m and 1.5m)
    assert np.all(depth >= 0.04)
    assert np.mean(depth) < 1.0


def test_scissors_contact_and_grip_detection(sim):
    """Verifies physical contact and kinematic proximity detection on scissors shank."""
    if hasattr(sim, "reset_tools"):
        sim.reset_tools()
    sim.moveJ(sim.tray_pose, duration=0.5)
    sim.open_gripper(duration=0.2)
    state = sim.get_gripper_state()
    assert state["grip_detected"] is False

    # Move TCP directly above scissors shank
    tray_rot = sim.waypoints["tray_pose"]["tcp_pose"][3:6]
    contact_z = sim.waypoints.get("padded_contact_pose", {}).get("tcp_pose", [0, 0, 0.0151])[2]
    pick_tcp = [sim.scissors_pos[0], sim.scissors_pos[1], contact_z] + tray_rot
    sim.moveL(pick_tcp, speed=0.05, duration=0.5)

    # Close gripper on scissors
    close_state = sim.close_gripper(force_n=25.0, duration=0.2)
    assert close_state["grip_detected"] is True
    assert abs(close_state["width_mm"] - 14.0) <= 2.5
    sim.open_gripper(duration=0.2)
    sim.moveJ(sim.tray_pose, duration=0.3)


def test_scalpel_contact_and_grip_detection(sim):
    """Verifies physical contact and kinematic proximity detection on scalpel neck."""
    if hasattr(sim, "reset_tools"):
        sim.reset_tools()
    sim.moveJ(sim.tray_pose, duration=0.5)
    sim.open_gripper(duration=0.2)
    state = sim.get_gripper_state()
    assert state["grip_detected"] is False

    # Move TCP directly above scalpel neck
    tray_rot = sim.waypoints["tray_pose"]["tcp_pose"][3:6]
    contact_z = sim.waypoints.get("padded_contact_pose", {}).get("tcp_pose", [0, 0, 0.0151])[2]
    pick_tcp = [sim.scalpel_pos[0], sim.scalpel_pos[1], contact_z] + tray_rot
    sim.moveL(pick_tcp, speed=0.05, duration=0.5)

    # Close gripper on scalpel (scalpel neck width in authentic mesh is ~16.4 mm)
    close_state = sim.close_gripper(force_n=25.0, duration=0.2)
    assert close_state["grip_detected"] is True
    assert abs(close_state["width_mm"] - 16.4) <= 2.5
    sim.open_gripper(duration=0.2)
    sim.moveJ(sim.tray_pose, duration=0.3)


def test_simulated_camera_scissors_and_scalpel_vision(sim):
    """Verifies simulated camera frame contains separable scissors and scalpel features."""
    from src.vision.detector import ToolDetector
    from src.vision.grasp_estimator import GraspEstimator

    sim.open_gripper(duration=0.1)
    sim.moveJ(sim.tray_pose, duration=0.5)
    if hasattr(sim, "reset_tools"):
        sim.reset_tools()
        time.sleep(0.1)
    color, depth, intr = sim.capture_camera_frame("wrist_realsense")
    detector = ToolDetector(allow_fallback=True)

    # Detect scissors
    s_dets = detector.detect(color, target_classes=["scissors"])
    assert len(s_dets) == 1
    assert s_dets[0].label == "scissors"

    # Detect scalpel
    sc_dets = detector.detect(color, target_classes=["scalpel"])
    assert len(sc_dets) == 1
    assert sc_dets[0].label == "scalpel"

    tray_tcp = sim.waypoints["tray_pose"]["tcp_pose"]
    estimator = GraspEstimator(camera_base_tcp=tray_tcp)
    est_s = estimator.estimate(s_dets[0], intr, depth)
    assert est_s is not None
    # Pick pose near scissors shank
    assert abs(est_s.robot_pick_pose[0] - sim.scissors_pos[0]) < 0.045
    assert abs(est_s.robot_pick_pose[1] - sim.scissors_pos[1]) < 0.045

    est_sc = estimator.estimate(sc_dets[0], intr, depth)
    assert est_sc is not None
    # Pick pose near scalpel neck
    assert abs(est_sc.robot_pick_pose[0] - sim.scalpel_pos[0]) < 0.045
    assert abs(est_sc.robot_pick_pose[1] - sim.scalpel_pos[1]) < 0.045


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


def test_movej_dynamic_duration(sim):
    """Verifies moveJ calculates dynamic duration when omitted and accepts speed/acceleration."""
    start_q = sim.get_actual_q()
    target_q = [start_q[0] + 0.1, start_q[1], start_q[2], start_q[3], start_q[4], start_q[5]]

    # Should compute duration = max(0.5, 0.1 / 0.8) = 0.5s without TypeError
    success = sim.moveJ(target_q, speed=0.8, acceleration=0.8)
    assert success
    actual_q = sim.get_actual_q()
    assert abs(actual_q[0] - target_q[0]) < 0.02

    # Return
    sim.moveJ(start_q, speed=0.8, acceleration=0.8)


def test_posetrans_math(sim):
    """Verifies exact URScript / ur_rtde spatial transform math (T_result = T_from * T_offset)."""
    p_tcp = [-0.546, 0.0026, 0.1704, -0.3012, 3.1271, -0.0002]

    # 1. Identity transform offset
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p_same = sim.poseTrans(p_tcp, identity)
    for expected, actual in zip(p_tcp, p_same):
        assert abs(expected - actual) < 1e-3

    # 2. Pure translation offset
    offset = [0.05, -0.02, 0.10, 0.0, 0.0, 0.0]
    p_offset = sim.poseTrans(p_tcp, offset)
    assert len(p_offset) == 6
    # Rotation remains the same for pure translation
    assert abs(p_offset[3] - p_tcp[3]) < 1e-3
    assert abs(p_offset[4] - p_tcp[4]) < 1e-3

    # 3. Direct function call
    direct_res = pose_trans(p_tcp, offset)
    assert direct_res == p_offset


def test_movel_cartesian_trajectory(sim):
    """Verifies Cartesian linear motion (moveL) using numerical differential IK."""
    sim.moveJ(sim.tray_pose, duration=1.0)
    start_tcp = sim.get_actual_tcp_pose()

    # Move TCP +3 cm in X and -2 cm in Z
    target_offset = [0.03, 0.0, -0.02, 0.0, 0.0, 0.0]
    target_tcp = sim.poseTrans(start_tcp, target_offset)

    success = sim.moveL(target_tcp, speed=0.05, acceleration=0.1, duration=1.0)
    assert success

    end_tcp = sim.get_actual_tcp_pose()
    assert abs(end_tcp[0] - target_tcp[0]) < 0.015
    assert abs(end_tcp[2] - target_tcp[2]) < 0.015


def test_rg2_gripper_millimeter_mapping(sim):
    """Verifies millimeter-based RG-2 gripper control and get_gripper_state dictionary layout."""
    # Move to 60.0 mm
    state = sim.move_gripper(60.0, force_n=20.0, wait=True)
    assert isinstance(state, dict)
    assert "width_mm" in state
    assert "status_raw" in state
    assert "busy" in state
    assert "grip_detected" in state
    assert "safety_circuit_active" in state
    assert abs(state["width_mm"] - 60.0) < 5.0

    # Move to 20.0 mm
    state = sim.move_gripper(20.0, force_n=15.0, wait=True)
    assert abs(state["width_mm"] - 20.0) < 5.0

    # Move to neutral 50.0 mm
    state = sim.move_gripper(50.0, force_n=20.0, wait=True)
    assert abs(state["width_mm"] - 50.0) < 5.0


def test_rg2_contact_detection(sim):
    """Verifies contact checking in MuJoCo contacts for grip_detected."""
    state = sim.get_gripper_state()
    # Initially no object grasped
    assert not state["grip_detected"]

    # Test contact override trigger
    sim._simulated_contact_override = True
    assert sim._check_grip_detected() is True
    state_grip = sim.get_gripper_state()
    assert state_grip["grip_detected"] is True
    assert bool(state_grip["status_raw"] & 2) is True

    # Restore
    sim._simulated_contact_override = False
    assert sim._check_grip_detected() is False


def test_input_validation(sim):
    """Verifies that vector length errors are rejected without crashing the background thread."""
    with pytest.raises(ValueError, match="target_q must have 6 elements"):
        sim.moveJ([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Joint positions must have 6 elements"):
        sim.set_joint_positions([0.0] * 5)

    with pytest.raises(ValueError, match="target_tcp must have 6 elements"):
        sim.moveL([0.0, 0.0, 0.0])

    # Ensure background simulation thread remains running and healthy
    assert sim.is_running()
    q = sim.get_actual_q()
    assert len(q) == 6


def test_stop_motion(sim):
    """Verifies that stop() immediately halts active motion and holds position."""
    import threading
    import time

    start_q = sim.get_actual_q()
    target_q = [start_q[0] + 0.4, start_q[1], start_q[2], start_q[3], start_q[4], start_q[5]]

    t = threading.Thread(target=lambda: sim.moveJ(target_q, speed=0.1))
    t.start()
    time.sleep(0.3)

    sim.stop()
    time.sleep(0.1)
    q_stopped = sim.get_actual_q()
    time.sleep(0.5)
    q_later = sim.get_actual_q()

    # The arm should have stayed at q_stopped without continuing towards target_q
    assert abs(q_later[0] - q_stopped[0]) < 0.01
    assert abs(q_later[0] - target_q[0]) > 0.05


def test_movel_unreachable(sim):
    """Verifies that moveL returns False when target TCP is outside the reachable workspace."""
    # 5 meters out of reach
    unreachable_tcp = [5.0, 0.0, 0.0, 0.0, 3.14, 0.0]
    success = sim.moveL(unreachable_tcp, duration=0.2)
    assert success is False


def test_get_inverse_kinematics(sim):
    """Verifies inverse kinematics solver on reachable and unreachable TCPs."""
    current_tcp = sim.get_actual_tcp_pose()
    ik_q = sim.get_inverse_kinematics(current_tcp)
    assert len(ik_q) == 6

    # Verify FK of ik_q matches current_tcp position
    fk_p, _ = sim._compute_fk(np.array(ik_q))
    for expected, actual in zip(current_tcp[:3], fk_p):
        assert abs(expected - actual) < 0.01

    # Unreachable target
    unreachable_tcp = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
    assert sim.get_inverse_kinematics(unreachable_tcp) == []


def test_asynchronous_motion(sim):
    """Verifies that moveJ with wait=False returns immediately without blocking."""
    import time
    start_q = sim.get_actual_q()
    target_q = [start_q[0] + 0.1, start_q[1], start_q[2], start_q[3], start_q[4], start_q[5]]

    t0 = time.time()
    success = sim.moveJ(target_q, speed=0.1, wait=False)
    elapsed = time.time() - t0

    assert success is True
    assert elapsed < 0.05  # Returned immediately

    # Wait for completion
    time.sleep(1.5)
    sim.stop()
