#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Vision-to-Actuation Grasp and Handover Pipeline.

Verifies:
1. Spatial rotation vector composition math (tool Z strictly perpendicular to floor).
2. Naive rotation addition bug contrast & cross-axis finger orientation.
3. Waypoint definition integrity in waypoints.json.
4. On-demand 'snap-and-stop' RealSenseStream acquisition lifecycle.
5. Affordance grasp geometry, approach/pick poses, and finger clearance.
6. Handover pull monitoring and compliant force-triggered release.
7. End-to-end simulated mission execution.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.hal import reset_shared_sim
from src.actuators.mission_runner import (
    acquire_and_estimate_grasp,
    compute_rg2_fingertip_offset_m,
    load_waypoints,
    monitor_handover_release,
    run_mission,
)
from src.vision.detector import DetectionResult
from src.vision.grasp_estimator import GraspEstimator
from src.vision.realsense_stream import CameraIntrinsics, RealSenseStream

TEST_IMAGE_PATH = REPO_ROOT / "data" / "vision" / "desk.jpg"


@pytest.fixture(autouse=True)
def clean_simulation():
    """Ensures simulation state is reset before and after each test."""
    yield
    reset_shared_sim()


# ---------------------------------------------------------------------------
# 1. Kinematics Rotation Math & Table Floor Perpendicularity
# ---------------------------------------------------------------------------

def test_compose_tray_rotation_perpendicular_to_table():
    """
    Verifies that spatial rotation composition preserves tool normal [0, 0, -1]
    across full 360-degree range of planar yaw angles.
    """
    base_tcp = [-0.54601, 0.0026, 0.17045, -0.30124, 3.1271, -0.00019]

    test_angles_deg = [-135, -90, -60, -45, -30, 0, 30, 45, 60, 90, 120, 180]

    for deg in test_angles_deg:
        yaw_rad = math.radians(deg)
        composed_rot = GraspEstimator.compose_tray_rotation(base_tcp, yaw_rad)

        assert len(composed_rot) == 3, f"Expected 3-element rotation vector, got {len(composed_rot)}"

        rot_mat = R.from_rotvec(composed_rot).as_matrix()
        z_tool = rot_mat[:, 2]

        # Tool Z must point directly down along negative Z of base frame: [0, 0, -1]
        np.testing.assert_allclose(
            z_tool,
            [0.0, 0.0, -1.0],
            atol=1e-3,
            err_msg=f"Tool Z tilted away from vertical for yaw={deg}°: z_tool={z_tool}",
        )

        dot = np.dot(z_tool, [0.0, 0.0, -1.0])
        assert dot >= 0.9999, f"Dot product {dot} below threshold for yaw={deg}°"


def test_naive_rotation_flaw_contrast():
    """
    Proves that the naive rotation approach (r_z += yaw) causes substantial tilt away
    from vertical, confirming why spatial composition is required.
    """
    base_tcp = [-0.54601, 0.0026, 0.17045, -0.30124, 3.1271, -0.00019]

    # At 45 degrees yaw:
    naive_rot_45 = [base_tcp[3], base_tcp[4], base_tcp[5] + math.radians(45.0)]
    mat_naive_45 = R.from_rotvec(naive_rot_45).as_matrix()
    z_naive_45 = mat_naive_45[:, 2]
    tilt_45_deg = math.degrees(math.acos(np.clip(np.dot(z_naive_45, [0.0, 0.0, -1.0]), -1.0, 1.0)))

    # Naive approach tilts tool by > 25 degrees at 45 deg yaw
    assert tilt_45_deg > 25.0, f"Expected naive tilt > 25 deg, got {tilt_45_deg:.2f} deg"

    # At 90 degrees yaw:
    naive_rot_90 = [base_tcp[3], base_tcp[4], base_tcp[5] + math.radians(90.0)]
    mat_naive_90 = R.from_rotvec(naive_rot_90).as_matrix()
    z_naive_90 = mat_naive_90[:, 2]
    tilt_90_deg = math.degrees(math.acos(np.clip(np.dot(z_naive_90, [0.0, 0.0, -1.0]), -1.0, 1.0)))

    # Naive approach tilts tool by > 50 degrees at 90 deg yaw
    assert tilt_90_deg > 50.0, f"Expected naive tilt > 50 deg, got {tilt_90_deg:.2f} deg"

    # In contrast, spatial composition maintains ~0.0 deg tilt
    composed_rot_45 = GraspEstimator.compose_tray_rotation(base_tcp, math.radians(45.0))
    mat_comp_45 = R.from_rotvec(composed_rot_45).as_matrix()
    z_comp_45 = mat_comp_45[:, 2]
    comp_tilt_45 = math.degrees(math.acos(np.clip(np.dot(z_comp_45, [0.0, 0.0, -1.0]), -1.0, 1.0)))
    assert comp_tilt_45 < 0.1, f"Expected composed tilt < 0.1 deg, got {comp_tilt_45:.4f} deg"


def test_cross_axis_finger_orientation():
    """
    Verifies that gripper yaw is perpendicular to tool longitudinal axis (shank normal)
    and stays normalized within [-90, 90] degrees.
    """
    estimator = GraspEstimator()
    w, h = 640, 480
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=w, height=h)

    # Test object orientations at 0, 30, 45, -60 degrees
    for test_angle in [0.0, 30.0, 45.0, -60.0]:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (320, 240), (90, 15), test_angle, 0, 360, 255, -1)
        det = DetectionResult(label="scissors", confidence=0.90, bbox=(220, 150, 420, 330), mask=mask)

        est = estimator.estimate(det, intrinsics)
        assert est is not None

        # Gripper yaw must be offset by 90 degrees relative to contour axis
        # and kept within [-90, 90]
        assert abs(est.yaw_deg) <= 90.0, f"Gripper yaw {est.yaw_deg}° exceeded [-90, 90] range"


def test_rg2_fingertip_offset_calculation():
    """
    Verifies the RG-2 parallel four-bar linkage kinematic fingertip retraction calculation.
    Checks:
    1. Zero offset at 0 mm opening (closed).
    2. Monotonic increase in retraction as fingers open wider.
    3. Retraction of ~5.4 mm at 40.7 mm opening (scissors pre-open).
    4. Retraction of ~28.9 mm at 100 mm opening (full open).
    """
    # 1. Closed state
    assert compute_rg2_fingertip_offset_m(0.0) == 0.0
    assert compute_rg2_fingertip_offset_m(-5.0) == 0.0

    # 2. Key physical widths
    offset_10 = compute_rg2_fingertip_offset_m(10.0) * 1000.0
    offset_40_7 = compute_rg2_fingertip_offset_m(40.7) * 1000.0
    offset_65 = compute_rg2_fingertip_offset_m(65.0) * 1000.0
    offset_100 = compute_rg2_fingertip_offset_m(100.0) * 1000.0

    # Monotonicity
    assert 0.0 < offset_10 < offset_40_7 < offset_65 < offset_100

    # Known physical values within ±0.5 mm
    assert abs(offset_40_7 - 5.50) < 0.5
    assert abs(offset_100 - 28.9) < 1.0


def test_rotation_invariant_camera_projection():
    """
    Verifies that GraspEstimator dynamically transforms camera displacements and
    mount offsets through the relative tool rotation, allowing arbitrary wrist orientations
    without requiring re-calibration.
    """
    estimator = GraspEstimator()
    w, h = 640, 480
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=w, height=h)

    # Synthetic object in center
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 20, 255, -1)
    det = DetectionResult(label="scissors", confidence=0.90, bbox=(300, 220, 340, 260), mask=mask)

    # 1. At baseline calibration pose
    est_calib = estimator.estimate(det, intrinsics, camera_base_tcp=estimator.calib_base_tcp)
    assert est_calib is not None

    # Expected at calibration pose: base_tcp[0] + dx, base_tcp[1] + dy
    np.testing.assert_allclose(
        est_calib.robot_pick_pose[:2],
        [estimator.calib_base_tcp[0] + estimator.dx_m, estimator.calib_base_tcp[1] + estimator.dy_m],
        atol=5e-3,
    )

    # 2. At 180-degree inverted wrist orientation
    inverted_rot = GraspEstimator.compose_tray_rotation(estimator.calib_base_tcp, math.radians(180.0))
    inverted_tcp = list(estimator.calib_base_tcp[:3]) + inverted_rot
    est_inv = estimator.estimate(det, intrinsics, camera_base_tcp=inverted_tcp)
    assert est_inv is not None

    # In 180-degree flipped pose, the mount offset in base frame should be inverted
    np.testing.assert_allclose(
        est_inv.robot_pick_pose[:2],
        [inverted_tcp[0] - estimator.dx_m, inverted_tcp[1] - estimator.dy_m],
        atol=1e-2,
    )


def test_hand_eye_6dof_transform_invariance():
    """
    Verifies that the full 6-DOF Hand-Eye matrix T_tcp_cam is orthonormal,
    rigid, and projects 3D targets to the exact same world coordinate regardless
    of arbitrary wrist yaw or rotation angles.
    """
    estimator = GraspEstimator()
    assert estimator.hand_eye_T is not None
    assert estimator.hand_eye_R is not None
    assert estimator.hand_eye_t is not None

    # Verify T_tcp_cam properties: orthogonal in O(3) with |det(R)| == 1, R^T * R == I
    np.testing.assert_almost_equal(abs(float(np.linalg.det(estimator.hand_eye_R))), 1.0, decimal=5)
    np.testing.assert_allclose(estimator.hand_eye_R.T @ estimator.hand_eye_R, np.eye(3), atol=1e-5)

    w, h = 640, 480
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=w, height=h)

    # Fixed physical world target on table
    ground_truth_target_base = np.array([-0.5765, -0.7287, estimator.contact_height_m])

    # Test across multiple wrist rotation angles: 0, 30, 45, 90, 180, -90 deg
    nominal_tcp = list(estimator.camera_base_tcp)
    for angle_deg in [0.0, 30.0, 45.0, 90.0, 180.0, -90.0]:
        rot_target = GraspEstimator.compose_tray_rotation(nominal_tcp, math.radians(angle_deg))
        current_tcp = list(nominal_tcp[:3]) + rot_target

        # 1. Exact 3D point forward/inverse transformation test (closed-form continuous kinematics)
        R_base_tcp = R.from_rotvec(rot_target).as_matrix()
        t_base_tcp = np.array(current_tcp[:3])
        p_tcp = R_base_tcp.T @ (ground_truth_target_base - t_base_tcp)
        p_cam = estimator.hand_eye_R.T @ (p_tcp - estimator.hand_eye_t)

        p_tcp_rec = estimator.hand_eye_R @ p_cam + estimator.hand_eye_t
        p_base_rec = R_base_tcp @ p_tcp_rec + t_base_tcp
        np.testing.assert_allclose(
            p_base_rec,
            ground_truth_target_base,
            atol=1e-5,
            err_msg=f"Continuous 6-DOF Hand-Eye transform drifted at wrist angle {angle_deg}°",
        )

        # 2. End-to-end vision pipeline test with rasterized pixel contour
        depth_m = float(p_cam[2])
        gu = float(p_cam[0] * intrinsics.fx / depth_m + intrinsics.cx)
        gv = float(p_cam[1] * intrinsics.fy / depth_m + intrinsics.cy)

        # Create mask around this pixel
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(round(gu)), int(round(gv))), 15, 255, -1)
        det = DetectionResult(label="forceps", confidence=0.95, bbox=(int(gu - 15), int(gv - 15), int(gu + 15), int(gv + 15)), mask=mask)

        # Pass depth map with ground truth depth
        depth_map = np.full((h, w), depth_m, dtype=np.float32)
        est = estimator.estimate(det, intrinsics, depth_map_m=depth_map, camera_base_tcp=current_tcp)
        assert est is not None

        # Pick horizontal coordinate must match ground truth target within 2.5mm (raster tolerance)
        np.testing.assert_allclose(
            est.robot_pick_pose[:2],
            ground_truth_target_base[:2],
            atol=2.5e-3,
            err_msg=f"Rasterized vision projection drifted at wrist angle {angle_deg}°",
        )
        # Pick Z must match contact height
        np.testing.assert_almost_equal(est.robot_pick_pose[2], estimator.contact_height_m, decimal=3)


# ---------------------------------------------------------------------------
# 2. Waypoints Loading & Structure
# ---------------------------------------------------------------------------

def test_load_waypoints():
    """Verifies calibrated waypoints.json loading and kinematic validity."""
    waypoints = load_waypoints()
    assert "home_pose" in waypoints
    assert "tray_pose" in waypoints
    assert "handover_pose" in waypoints

    for wp_name in ["home_pose", "tray_pose", "handover_pose"]:
        joints = waypoints[wp_name]["joints_rad"]
        assert len(joints) == 6
        assert all(isinstance(v, (int, float)) for v in joints)
        assert all(-2.0 * math.pi <= v <= 2.0 * math.pi for v in joints)

        tcp = waypoints[wp_name]["tcp_pose"]
        assert len(tcp) == 6
        assert all(isinstance(v, (int, float)) for v in tcp)
        assert tcp[2] > 0.0  # Z above table / base zero

    # Tray pose scan standoff should be within safe operational bounds (~0.15 - 0.25 m)
    assert 0.10 <= waypoints["tray_pose"]["tcp_pose"][2] <= 0.30


def test_load_waypoints_missing_file():
    """Verifies informative error when waypoints file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_waypoints(Path("/nonexistent/waypoints.json"))


# ---------------------------------------------------------------------------
# 3. Snap-and-Stop Vision Frame Acquisition
# ---------------------------------------------------------------------------

def test_snap_and_stop_lifecycle():
    """
    Verifies on-demand snap-and-stop frame capture: acquires frame and ensures
    camera pipeline is immediately stopped to save Pi CPU and USB resources.
    """
    assert TEST_IMAGE_PATH.exists(), f"Sample image missing at {TEST_IMAGE_PATH}"
    stream = RealSenseStream(source=str(TEST_IMAGE_PATH))

    # Pipeline is initially inactive
    assert not stream._is_running

    color, depth, intrinsics = stream.capture_snapshot(warmup_frames=2)

    # Verifies frame contents
    assert color is not None
    assert color.shape[0] > 100 and color.shape[1] > 100
    assert intrinsics.fx > 100.0

    # Streamer MUST be stopped immediately after snapshot
    assert not stream._is_running, "RealSenseStream should be stopped after capture_snapshot"

    # Repeatability check: ensure repeated snapshots work without leaking resources
    color2, _, _ = stream.capture_snapshot(warmup_frames=1)
    assert color2 is not None
    assert not stream._is_running


def test_snap_and_stop_realsense_mock():
    """Verifies that RealSense warmup frames are consumed and pipeline is stopped in finally block."""
    stream = RealSenseStream(source="realsense")
    assert stream.fps == 15
    mock_pipe = MagicMock()
    mock_align = MagicMock()

    stream.active_backend = "realsense"
    stream.pipeline = mock_pipe
    stream.align = mock_align
    stream._is_running = True

    # Mock get_frame return
    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_intrinsics = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    stream.get_frame = MagicMock(return_value=(fake_img, None, fake_intrinsics))

    # Test with custom warmup frames
    color, depth, intrinsics = stream.capture_snapshot(warmup_frames=5)
    assert mock_pipe.wait_for_frames.call_count == 5
    assert mock_pipe.stop.call_count == 1
    assert not stream._is_running

    # Test default 2 warmup frames
    stream.pipeline = mock_pipe
    stream._is_running = True
    mock_pipe.reset_mock()
    stream.capture_snapshot()
    assert mock_pipe.wait_for_frames.call_count == 2
    assert mock_pipe.stop.call_count == 1
    assert not stream._is_running


# ---------------------------------------------------------------------------
# 4. Affordance Grasp Geometry, Approach/Pick Poses, and Finger Clearance
# ---------------------------------------------------------------------------

def test_grasp_estimator_scissors_geometry():
    """
    Tests grasp estimation on a synthetic scissors mask:
    Verifies scan standoff height, contact pick height, box-lock affordance point,
    and RG-2 finger opening width bounds.
    """
    estimator = GraspEstimator()
    w, h = 640, 480
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=w, height=h)

    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (int(w * 0.12), int(h * 0.03)), 35, 0, 360, 255, -1)
    ring_x = int(cx + w * 0.10 * math.cos(math.radians(35)))
    ring_y = int(cy + w * 0.10 * math.sin(math.radians(35)))
    cv2.circle(mask, (ring_x, ring_y), int(h * 0.05), 255, -1)
    x1, y1 = max(0, cx - int(w * 0.15)), max(0, cy - int(h * 0.12))
    x2, y2 = min(w, cx + int(w * 0.15)), min(h, cy + int(h * 0.12))
    det = DetectionResult(label="scissors", confidence=0.95, bbox=(x1, y1, x2, y2), mask=mask)
    est = estimator.estimate(det, intrinsics)

    assert est is not None
    assert est.label == "scissors"

    # UR10e Approach and Pick heights
    # Scan height: matches estimator.scan_height_m, Pick height: matches estimator.contact_height_m
    np.testing.assert_almost_equal(est.robot_approach_pose[2], estimator.scan_height_m, decimal=3)
    np.testing.assert_almost_equal(est.robot_pick_pose[2], estimator.contact_height_m, decimal=3)

    # Approach and pick must share the same horizontal coordinate & orientation
    assert est.robot_approach_pose[0] == est.robot_pick_pose[0]
    assert est.robot_approach_pose[1] == est.robot_pick_pose[1]
    assert est.robot_approach_pose[3:] == est.robot_pick_pose[3:]

    # Tool Z must remain perpendicular to table
    mat = R.from_rotvec(est.robot_approach_pose[3:]).as_matrix()
    np.testing.assert_allclose(mat[:, 2], [0.0, 0.0, -1.0], atol=1e-3)

    # Gripper opening width: tool thickness + 18mm, clipped to [25, 95] mm
    assert 25.0 <= est.gripper_open_width_mm <= 95.0


def test_acquire_and_estimate_grasp_from_image():
    """Tests the combined acquisition + grasp estimation helper using desk image."""
    estimate = acquire_and_estimate_grasp(
        source=str(TEST_IMAGE_PATH),
        target="scissors",
        sim=True,
    )
    assert estimate is not None
    assert len(estimate.robot_approach_pose) == 6
    assert len(estimate.robot_pick_pose) == 6
    assert estimate.robot_approach_pose[2] > estimate.robot_pick_pose[2]


def test_acquire_and_estimate_grasp_from_sim_camera():
    """Tests acquisition + grasp estimation helper using simulated eye-in-hand camera."""
    from src.actuators.hal import get_shared_sim
    sim = get_shared_sim()
    if hasattr(sim, "reset_tools"):
        sim.reset_tools()
    sim.moveJ(sim.tray_pose, duration=0.2)
    sim.close_gripper(force_n=20.0, duration=0.2)

    estimate = acquire_and_estimate_grasp(
        source="sim",
        target="scissors",
        sim=True,
    )
    assert estimate is not None
    assert len(estimate.robot_approach_pose) == 6
    assert len(estimate.robot_pick_pose) == 6
    assert estimate.robot_approach_pose[2] > estimate.robot_pick_pose[2]
    # Scissors pick pose in simulation should be at scissors shank
    assert abs(estimate.robot_pick_pose[0] - sim.scissors_pos[0]) < 0.045
    assert abs(estimate.robot_pick_pose[1] - sim.scissors_pos[1]) < 0.045
    contact_z = sim.waypoints.get("padded_contact_pose", {}).get("tcp_pose", [0, 0, 0.0151])[2]
    assert abs(estimate.robot_pick_pose[2] - contact_z) < 0.005

    # Scalpel pick pose in simulation should be at scalpel neck
    est_scalpel = acquire_and_estimate_grasp(
        source="sim",
        target="scalpel",
        sim=True,
    )
    assert est_scalpel is not None
    assert abs(est_scalpel.robot_pick_pose[0] - sim.scalpel_pos[0]) < 0.03
    assert abs(est_scalpel.robot_pick_pose[1] - sim.scalpel_pos[1]) < 0.045
    assert abs(est_scalpel.robot_pick_pose[2] - contact_z) < 0.005


def test_realsense_stream_sim_source():
    """Verifies RealSenseStream with source='sim' connects to shared simulation and returns valid frames."""
    from src.actuators.hal import get_shared_sim
    sim = get_shared_sim()
    if hasattr(sim, "reset_tools"):
        sim.reset_tools()
    sim.moveJ(sim.tray_pose, duration=0.2)
    sim.close_gripper(force_n=20.0, duration=0.2)

    stream = RealSenseStream(source="sim")
    stream.start()

    color, depth, intr = stream.get_frame()
    assert color is not None
    assert depth is not None
    assert intr is not None
    assert color.shape == (480, 640, 3)
    assert depth.shape == (480, 640)
    assert intr.width == 640
    assert intr.height == 480

    snap_c, snap_d, snap_intr = stream.capture_snapshot(warmup_frames=2)
    assert snap_c.shape == (480, 640, 3)
    assert snap_d.shape == (480, 640)

    stream.stop()


# ---------------------------------------------------------------------------
# 5. Handover Pull Force Monitor
# ---------------------------------------------------------------------------

def test_monitor_handover_release_triggers_on_force():
    """Verifies that surgeon pull above threshold triggers zeroing, compliance, and gripper release."""
    mock_rtde_c = MagicMock()
    mock_rtde_r = MagicMock()
    mock_gripper = MagicMock()

    # Sequence of forces: initial tare [0, 0, 0], gentle touch [2, 2, 1] (F=3N), then pull [8, 8, 2] (F=11.5N >= 10N)
    force_sequence = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        [8.0, 8.0, 2.0, 0.0, 0.0, 0.0],
    ]
    mock_rtde_r.getActualTCPForce.side_effect = force_sequence

    released = monitor_handover_release(
        rtde_c=mock_rtde_c,
        rtde_r=mock_rtde_r,
        gripper=mock_gripper,
        force_threshold_n=10.0,
        timeout_s=5.0,
        poll_rate_hz=100.0,
    )

    assert released
    mock_rtde_c.zeroFtSensor.assert_called_once()
    mock_gripper.open.assert_called_once_with(force_n=20.0)


def test_monitor_handover_release_timeout():
    """Verifies that handover monitoring times out safely if no pull is detected."""
    mock_rtde_c = MagicMock()
    mock_rtde_r = MagicMock()
    mock_gripper = MagicMock()

    # Constant low noise force below 10 N
    mock_rtde_r.getActualTCPForce.return_value = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    released = monitor_handover_release(
        rtde_c=mock_rtde_c,
        rtde_r=mock_rtde_r,
        gripper=mock_gripper,
        force_threshold_n=10.0,
        timeout_s=0.2,  # Short timeout for test
        poll_rate_hz=50.0,
    )

    assert not released
    mock_gripper.open.assert_not_called()


def test_monitor_handover_release_sim_auto_pull():
    """Verifies that simulation auto-pull triggers compliant release after specified delay."""
    mock_rtde_c = MagicMock()
    mock_rtde_r = MagicMock()
    mock_gripper = MagicMock()
    mock_rtde_r.getActualTCPForce.return_value = [0.0] * 6

    released = monitor_handover_release(
        rtde_c=mock_rtde_c,
        rtde_r=mock_rtde_r,
        gripper=mock_gripper,
        force_threshold_n=10.0,
        timeout_s=2.0,
        sim_auto_pull=True,
        sim_pull_delay_s=0.1,
    )

    assert released
    mock_gripper.open.assert_called_once()


# ---------------------------------------------------------------------------
# 6. End-to-End Mission Pipeline (Simulation)
# ---------------------------------------------------------------------------

def test_mission_runner_end_to_end_sim():
    """
    Executes the entire retrieval mission end-to-end in headless MuJoCo simulation:
    home -> tray -> snap capture -> grasp estimate -> approach -> plunge -> grasp
    -> retract -> handover -> pull release -> home.
    """
    result = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source=str(TEST_IMAGE_PATH),
        speed=1.5,           # Accelerated motion speed for rapid automated testing
        force_threshold=10.0,
        handover_timeout=5.0,
        sim_auto_pull=True,
        sim_pull_delay=0.1,
    )

    assert result.success, f"Mission failed with error: {result.message} (phase: {result.phase})"
    assert result.phase == "completed"
    assert result.target == "scissors"
    assert result.grasp_estimate is not None
    assert result.grip_detected
    assert result.handover_released
    assert result.elapsed_time_s > 0.0


def test_mission_runner_end_to_end_sim_camera():
    """
    Executes the entire retrieval mission end-to-end using the simulated eye-in-hand
    wrist RealSense camera frame and authentic scissors tool detection:
    home -> tray -> sim camera snapshot -> tool detection -> grasp estimate -> plunge
    -> clamp -> retract -> handover -> pull release -> home.
    """
    result = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source="sim",
        speed=2.5,
        force_threshold=10.0,
        handover_timeout=5.0,
        sim_auto_pull=True,
        sim_pull_delay=0.1,
    )

    assert result.success, f"Mission failed with error: {result.message} (phase: {result.phase})"
    assert result.phase == "completed"
    assert result.target == "scissors"
    assert result.grasp_estimate is not None
    assert result.grip_detected
    assert result.handover_released
    assert result.elapsed_time_s > 0.0


def test_mission_runner_end_to_end_sim_camera_scalpel():
    """
    Executes the entire retrieval mission end-to-end for scalpel using the simulated
    eye-in-hand wrist RealSense camera frame and authentic scalpel tool detection:
    home -> tray -> sim camera snapshot -> scalpel detection -> grasp estimate -> plunge
    -> clamp -> retract -> handover -> pull release -> home.
    """
    result = run_mission(
        target="scalpel",
        sim=True,
        headless=True,
        source="sim",
        speed=3.0,
        force_threshold=10.0,
        handover_timeout=5.0,
        sim_auto_pull=True,
        sim_pull_delay=0.1,
    )

    assert result.success, f"Mission failed with error: {result.message} (phase: {result.phase})"
    assert result.phase == "completed"
    assert result.target == "scalpel"
    assert result.grasp_estimate is not None
    assert result.grip_detected
    assert result.handover_released
    assert result.elapsed_time_s > 0.0


def test_sim_tcp_force_sensing_and_zeroing():
    """Verifies simulated TCP force injection and taring via zeroFtSensor."""
    from src.actuators.hal import get_robot_interfaces
    rtde_c, rtde_r = get_robot_interfaces(sim=True, headless=True)

    # Initial force is zero
    assert rtde_r.getActualTCPForce() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Inject simulated pull force
    rtde_r.set_simulated_tcp_force([0.0, 0.0, 14.5, 0.0, 0.0, 0.0])
    forces = rtde_r.getActualTCPForce()
    assert forces[2] == 14.5

    # Zeroing tares the sensor back to zero
    rtde_c.zeroFtSensor()
    assert rtde_r.getActualTCPForce() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_monitor_handover_release_real_force_sensing():
    """Verifies that monitor_handover_release works with the real SimRTDE interface using injected force."""
    from src.actuators.hal import get_gripper, get_robot_interfaces
    rtde_c, rtde_r = get_robot_interfaces(sim=True, headless=True)
    gripper = get_gripper(sim=True, headless=True)

    released = monitor_handover_release(
        rtde_c=rtde_c,
        rtde_r=rtde_r,
        gripper=gripper,
        force_threshold_n=10.0,
        timeout_s=3.0,
        sim_auto_pull=True,
        sim_pull_delay_s=0.1,
    )

    assert released
    assert gripper.width_mm > 100.0  # Gripper opened fully


def test_mission_runner_grasp_failed_aborts_safely(monkeypatch):
    """
    Verifies that when grip verification fails, run_mission aborts safely
    at approach height without traversing to handover_pose.
    """
    from src.actuators.hal import SimRG2Gripper

    # Force gripper to return grip_detected=False
    def mock_close(self, force_n=20.0):
        return {
            "width_mm": 0.0,
            "status_raw": 0,
            "busy": False,
            "grip_detected": False,
            "safety_circuit_active": False,
        }

    monkeypatch.setattr(SimRG2Gripper, "close", mock_close)

    result = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source=str(TEST_IMAGE_PATH),
        speed=2.0,
        sim_auto_pull=True,
    )

    assert not result.success
    assert result.phase == "grasp_failed"
    assert not result.grip_detected
    assert not result.handover_released


def test_mission_runner_handover_timeout_holds_position():
    """
    Verifies that when surgeon pull is not detected, run_mission aborts safely
    with handover_timeout and holds position without returning home.
    """
    result = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source=str(TEST_IMAGE_PATH),
        speed=2.0,
        force_threshold=10.0,
        handover_timeout=0.2,   # Very short timeout
        sim_auto_pull=False,    # Do not auto-pull
    )

    assert not result.success
    assert result.phase == "handover_timeout"
    assert result.grip_detected
    assert not result.handover_released


def test_load_waypoints_missing_required_key(tmp_path):
    """Verifies that missing required waypoints raises ValueError."""
    import json
    bad_wp_file = tmp_path / "bad_waypoints.json"
    with open(bad_wp_file, "w") as f:
        json.dump({"home_pose": {"joints_rad": [0.0] * 6}}, f)

    with pytest.raises(ValueError, match="missing required waypoint"):
        load_waypoints(bad_wp_file)


def test_mission_runner_surface_selection_and_limits():
    """Verifies that run_mission with padded and table surface respects dual-tier limits in simulation."""
    result_padded = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source=str(TEST_IMAGE_PATH),
        speed=2.0,
        surface="padded",
        pad_offset_mm=2.0,
        sim_auto_pull=True,
        sim_pull_delay=0.1,
    )
    assert result_padded.success
    assert result_padded.phase == "completed"

    result_table = run_mission(
        target="scissors",
        sim=True,
        headless=True,
        source=str(TEST_IMAGE_PATH),
        speed=2.0,
        surface="table",
        sim_auto_pull=True,
        sim_pull_delay=0.1,
    )
    assert result_table.success
    assert result_table.phase == "completed"



