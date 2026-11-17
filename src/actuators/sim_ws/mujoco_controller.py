#!/usr/bin/env python3
"""
MuJoCo Simulation Controller for Universal Robots UR-10e and OnRobot RG-2 Gripper.

Provides high-fidelity physics simulation with authentic CAD models,
wall-clock synchronized trajectory execution (no teleportation),
Cartesian linear interpolation (moveL), spatial transformations (poseTrans),
millimeter-based RG-2 gripper control with contact sensing,
and live state monitoring via a dedicated single-threaded simulation loop.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from src.vision.realsense_stream import CameraIntrinsics

# Check if native MuJoCo C-bindings are installed
try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco = None

# Default paths dynamically resolved relative to this module
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
DEFAULT_SCENE_XML = CURRENT_DIR.parent / "mujoco" / "models" / "scene.xml"
WAYPOINTS_FILE = CURRENT_DIR.parent / "waypoints.json"


# ---------------------------------------------------------------------------
# Spatial Transformation & Kinematic Math Utilities
# ---------------------------------------------------------------------------

def rotvec_to_rotmat(r: np.ndarray) -> np.ndarray:
    """Converts a 3D axis-angle rotation vector (radians) to a 3x3 rotation matrix."""
    return R.from_rotvec(r).as_matrix()


def rotmat_to_rotvec(rot_mat: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a 3D axis-angle rotation vector (radians)."""
    return R.from_matrix(rot_mat).as_rotvec()


def pose_trans(p_from: List[float], p_offset: List[float]) -> List[float]:
    """
    Computes the spatial transformation T_result = T_from * T_offset,
    matching the exact specification of URScript / ur_rtde `poseTrans`.
    Both p_from and p_offset are [X, Y, Z, Rx, Ry, Rz] (p_offset can also be [X, Y, Z]).
    """
    if len(p_from) != 6:
        raise ValueError(f"p_from must have 6 elements, got {len(p_from)}")

    p1 = np.array(p_from[:3], dtype=float)
    rot1 = R.from_rotvec(p_from[3:6])

    p_off = list(p_offset)
    if len(p_off) == 3:
        p_off = p_off + [0.0, 0.0, 0.0]
    elif len(p_off) != 6:
        raise ValueError(f"p_offset must have 3 or 6 elements, got {len(p_offset)}")

    p2 = np.array(p_off[:3], dtype=float)
    rot2 = R.from_rotvec(p_off[3:6])

    p_res = p1 + rot1.apply(p2)
    rot_res = rot1 * rot2
    r_res = rot_res.as_rotvec()

    return [float(x) for x in np.concatenate([p_res, r_res])]


def _quat_to_mat(q: Tuple[float, float, float, float] | List[float]) -> np.ndarray:
    """Converts MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _analytical_hand_link_pose(q: np.ndarray | List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes forward kinematics (pos, rotmat) of hand_link in UR-10e base frame
    matching scene.xml definitions.
    """
    def rotz(th: float) -> np.ndarray:
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def roty(th: float) -> np.ndarray:
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    # Base: rotated 180 deg around Z
    t_mat = np.eye(4)
    t_mat[:3, :3] = np.diag([-1.0, -1.0, 1.0])

    # Shoulder link: pos [0, 0, 0.181], joint shoulder_pan_joint around Z
    t_shoulder = np.eye(4)
    t_shoulder[:3, 3] = [0.0, 0.0, 0.181]
    t_shoulder[:3, :3] = rotz(float(q[0]))
    t_mat = t_mat @ t_shoulder

    # Upper arm: pos [0, 0.176, 0], quat [1, 0, 1, 0] (90 deg about Y), joint shoulder_lift_joint around Y
    r_u = _quat_to_mat([1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0), 0.0])
    t_upper = np.eye(4)
    t_upper[:3, 3] = [0.0, 0.176, 0.0]
    t_upper[:3, :3] = r_u @ roty(float(q[1]))
    t_mat = t_mat @ t_upper

    # Forearm: pos [0, -0.137, 0.613], joint elbow_joint around Y
    t_forearm = np.eye(4)
    t_forearm[:3, 3] = [0.0, -0.137, 0.613]
    t_forearm[:3, :3] = roty(float(q[2]))
    t_mat = t_mat @ t_forearm

    # Wrist 1: pos [0, 0, 0.571], quat [1, 0, 1, 0] (90 deg about Y), joint wrist_1_joint around Y
    r_w1 = _quat_to_mat([1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0), 0.0])
    t_w1 = np.eye(4)
    t_w1[:3, 3] = [0.0, 0.0, 0.571]
    t_w1[:3, :3] = r_w1 @ roty(float(q[3]))
    t_mat = t_mat @ t_w1

    # Wrist 2: pos [0, 0.135, 0], joint wrist_2_joint around Z
    t_w2 = np.eye(4)
    t_w2[:3, 3] = [0.0, 0.135, 0.0]
    t_w2[:3, :3] = rotz(float(q[4]))
    t_mat = t_mat @ t_w2

    # Wrist 3: pos [0, 0, 0.12], joint wrist_3_joint around Y
    t_w3 = np.eye(4)
    t_w3[:3, 3] = [0.0, 0.0, 0.12]
    t_w3[:3, :3] = roty(float(q[5]))
    t_mat = t_mat @ t_w3

    # Hand link: pos [0, 0.1168, 0] + [0.007, 0, 0] in adapter frame = [0.0, 0.1238, 0.0] in wrist_3
    # Bolted to PickNik RealSense adapter mounted to UR-10e tool flange (flange at y = 0.1168)
    r_hand = _quat_to_mat([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    t_hand = np.eye(4)
    t_hand[:3, 3] = [0.0, 0.1238, 0.0]
    t_hand[:3, :3] = r_hand
    t_mat = t_mat @ t_hand

    return t_mat[:3, 3], t_mat[:3, :3]


def _analytical_ur10e_rg2_fk(q: np.ndarray | List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward kinematics (TCP pos, TCP rotmat) along the UR-10e + RG-2 tree
    matching scene.xml definitions.
    """
    p_hand, r_hand = _analytical_hand_link_pose(q)
    t_hand = np.eye(4)
    t_hand[:3, 3] = p_hand
    t_hand[:3, :3] = r_hand

    # TCP site: calibrated at RG-2 fingertips
    # pos [0.20, -0.00195, 0.00451], quat [0.499596, -0.499753, 0.498662, -0.501983]
    t_tcp = np.eye(4)
    t_tcp[:3, 3] = [0.20, -0.00195, 0.00451]
    t_tcp[:3, :3] = np.array([
        [-0.00131,  0.00319,  0.99999],
        [-0.99999, -0.00355, -0.00130],
        [ 0.00354, -0.99999,  0.00319]
    ])
    t_mat = t_hand @ t_tcp

    return t_mat[:3, 3], t_mat[:3, :3]


def _numerical_ur10e_rg2_jacobian(q: np.ndarray | List[float]) -> np.ndarray:
    """Computes the 6x6 TCP site Jacobian for arm joints via central differences."""
    eps = 1e-6
    jacp = np.zeros((3, 6))
    jacr = np.zeros((3, 6))
    for i in range(6):
        qp = np.array(q, dtype=float)
        qp[i] += eps
        qm = np.array(q, dtype=float)
        qm[i] -= eps
        pp, rp = _analytical_ur10e_rg2_fk(qp)
        pm, rm = _analytical_ur10e_rg2_fk(qm)
        jacp[:, i] = (pp - pm) / (2.0 * eps)
        r_err = rp @ rm.T
        jacr[:, i] = R.from_matrix(r_err).as_rotvec() / (2.0 * eps)
    return np.vstack([jacp, jacr])


# ---------------------------------------------------------------------------
# Fallback MuJoCo Data Structures for Environments without compiled bindings
# ---------------------------------------------------------------------------

class _FallbackMjJoint:
    def __init__(self, qposadr: int, dofadr: Optional[int] = None):
        self.qposadr = [qposadr]
        self.dofadr = [dofadr if dofadr is not None else qposadr]


class _FallbackMjModel:
    """Mock model holding identical indices and structure as scene.xml."""
    def __init__(self):
        self.nq = 26
        self.nv = 24
        self.nu = 7
        self.opt = type("Opt", (), {"timestep": 0.002})()
        self._joints = {
            "shoulder_pan_joint": _FallbackMjJoint(0, 0),
            "shoulder_lift_joint": _FallbackMjJoint(1, 1),
            "elbow_joint": _FallbackMjJoint(2, 2),
            "wrist_1_joint": _FallbackMjJoint(3, 3),
            "wrist_2_joint": _FallbackMjJoint(4, 4),
            "wrist_3_joint": _FallbackMjJoint(5, 5),
            "gripper_joint": _FallbackMjJoint(6, 6),
            "scalpel_joint": _FallbackMjJoint(12, 12),
            "scissors_joint": _FallbackMjJoint(19, 18),
        }
        self.actuator_forcerange = np.tile([-40.0, 40.0], (7, 1))
        self._geom_names = {
            "hand_col": 0, "l_moment_col": 1, "l_finger_col": 2, "l_truss_col": 3,
            "r_moment_col": 4, "r_finger_col": 5, "r_truss_col": 6,
            "scalpel_handle": 10, "scalpel_blade": 11, "floor": 12,
            "scissors_shank": 20, "scissors_blade_col": 21, "scissors_handle_col": 22,
            "scissors_blade_left": 23, "scissors_blade_right": 24,
            "scissors_handle_left": 25, "scissors_handle_right": 26,
            "scissors_vis": 27, "scissors_blades_vis": 28, "scissors_handles_vis": 29,
        }

    def joint(self, name: str) -> _FallbackMjJoint:
        return self._joints[name]


class _FallbackMjContact:
    def __init__(self, geom1: int, geom2: int):
        self.geom1 = geom1
        self.geom2 = geom2


class _FallbackMjData:
    """Mock data holding qpos, qvel, ctrl, contacts, and site transforms."""
    def __init__(self, model: _FallbackMjModel):
        self.model = model
        self.qpos = np.zeros(model.nq, dtype=float)
        self.qvel = np.zeros(model.nv, dtype=float)
        self.ctrl = np.zeros(model.nu, dtype=float)
        self.site_xpos = np.zeros((10, 3), dtype=float)
        self.site_xmat = np.zeros((10, 9), dtype=float)
        self.ncon = 0
        self.contact: List[_FallbackMjContact] = []


class _FallbackMujocoModule:
    """Mock mujoco module providing mj_name2id, mj_forward, mj_step, and viewer."""
    class mjtObj:
        mjOBJ_JOINT = 1
        mjOBJ_ACTUATOR = 2
        mjOBJ_SITE = 3
        mjOBJ_GEOM = 4
        mjOBJ_CAMERA = 7

    @staticmethod
    def mj_name2id(model: Any, obj_type: int, name: str) -> int:
        actuators = {
            "shoulder_pan": 0, "shoulder_lift": 1, "elbow": 2,
            "wrist_1": 3, "wrist_2": 4, "wrist_3": 5, "rg2_actuator": 6
        }
        sites = {"rg2_tcp": 0}
        cameras = {"wrist_realsense": 0, "overview_cam": 1}
        geoms = {
            "hand_col": 0, "l_moment_col": 1, "l_finger_col": 2, "l_truss_col": 3,
            "r_moment_col": 4, "r_finger_col": 5, "r_truss_col": 6,
            "scalpel_handle": 10, "scalpel_blade": 11, "floor": 12,
            "scissors_shank": 20, "scissors_blade_col": 21, "scissors_handle_col": 22,
            "scissors_blade_left": 23, "scissors_blade_right": 24,
            "scissors_handle_left": 25, "scissors_handle_right": 26,
            "scissors_vis": 27, "scissors_blades_vis": 28, "scissors_handles_vis": 29,
        }
        if obj_type == _FallbackMujocoModule.mjtObj.mjOBJ_ACTUATOR:
            return actuators.get(name, -1)
        elif obj_type == _FallbackMujocoModule.mjtObj.mjOBJ_SITE:
            return sites.get(name, -1)
        elif obj_type == _FallbackMujocoModule.mjtObj.mjOBJ_GEOM:
            return geoms.get(name, -1)
        elif obj_type == _FallbackMujocoModule.mjtObj.mjOBJ_CAMERA:
            return cameras.get(name, -1)
        return 0

    @staticmethod
    def mj_forward(model: Any, data: Any):
        pos, mat = _analytical_ur10e_rg2_fk(data.qpos[:6])
        data.site_xpos[0] = pos
        data.site_xmat[0] = mat.flatten()

    @staticmethod
    def mj_fwdPosition(model: Any, data: Any):
        pos, mat = _analytical_ur10e_rg2_fk(data.qpos[:6])
        data.site_xpos[0] = pos
        data.site_xmat[0] = mat.flatten()

    @staticmethod
    def mj_step(model: Any, data: Any):
        # Update arm joints towards ctrl with smooth PD step
        alpha = 0.15
        for i in range(6):
            diff = data.ctrl[i] - data.qpos[i]
            data.qvel[i] = diff / 0.016
            data.qpos[i] += alpha * diff

        # Update gripper joint towards ctrl with obstacle collision stopping
        grip_diff = data.ctrl[6] - data.qpos[6]
        new_qpos = data.qpos[6] + alpha * grip_diff

        # Physical tool obstacle contact in tray:
        # If TCP is at pick height near scissors shank (14mm diameter) or scalpel neck (12mm diameter),
        # the rigid tool prevents gripper fingers from penetrating through it.
        tcp_pos = data.site_xpos[0]
        s_pos = getattr(model, "scissors_pos", [-0.630, -0.7287, 0.0219])
        sc_pos = getattr(model, "scalpel_pos", [-0.520, -0.7287, 0.0155])
        dist_s = math.sqrt((tcp_pos[0] - s_pos[0]) ** 2 + (tcp_pos[1] - s_pos[1]) ** 2)
        dist_sc = math.sqrt((tcp_pos[0] - sc_pos[0]) ** 2 + (tcp_pos[1] - sc_pos[1]) ** 2)
        dist_z = abs(tcp_pos[2] - 0.0151)

        min_allowed_q = -0.45  # Closed 0.0 mm
        if dist_z < 0.040:
            if dist_s < 0.060 and dist_s <= dist_sc:
                min_allowed_q = -0.45 + (14.0 / 110.0) * 1.15  # 14.0 mm shank
            elif dist_sc < 0.060 and dist_sc < dist_s:
                min_allowed_q = -0.45 + (12.0 / 110.0) * 1.15  # 12.0 mm neck

        if new_qpos < min_allowed_q and data.qpos[6] >= min_allowed_q - 0.01:
            new_qpos = min_allowed_q
            data.qvel[6] = 0.0
        else:
            data.qvel[6] = grip_diff / 0.016

        data.qpos[6] = new_qpos

        _FallbackMujocoModule.mj_forward(model, data)

    @staticmethod
    def mj_jacSite(model: Any, data: Any, jacp: np.ndarray, jacr: np.ndarray, site_id: int):
        j = _numerical_ur10e_rg2_jacobian(data.qpos[:6])
        jacp[:, :6] = j[:3, :]
        jacr[:, :6] = j[3:, :]

    class viewer:
        @staticmethod
        def launch_passive(model: Any, data: Any):
            raise RuntimeError("GUI viewer unavailable in headless fallback mode")


# ---------------------------------------------------------------------------
# MujocoSimController Main Class
# ---------------------------------------------------------------------------

class MujocoSimController:
    """
    Simulation controller interfacing with MuJoCo for UR-10e arm and RG-2 gripper.
    Runs a single dedicated physics and viewer thread to guarantee thread safety
    and prevent concurrent mjData stack collisions.
    """

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    ARM_ACTUATOR_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    FINGER_GEOM_NAMES = [
        "hand_col",
        "l_moment_col",
        "l_finger_col",
        "l_truss_col",
        "r_moment_col",
        "r_finger_col",
        "r_truss_col",
    ]

    OBJECT_GEOM_NAMES = [
        "scalpel_handle",
        "scalpel_blade",
        "scissors_shank",
        "scissors_blade_col",
        "scissors_handle_col",
        "scissors_blade_left",
        "scissors_blade_right",
        "scissors_handle_left",
        "scissors_handle_right",
    ]

    def __init__(
        self,
        scene_xml: Optional[Path | str] = None,
        headless: Optional[bool] = None,
    ):
        global mujoco
        self.xml_path = Path(scene_xml) if scene_xml else DEFAULT_SCENE_XML

        # Auto-detect display if headless is omitted
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        in_test = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
        if headless is None:
            self.headless = True if in_test else (not has_display)
        else:
            self.headless = headless

        self.viewer = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Load MuJoCo Model and Data (or kinematic fallback)
        if HAS_MUJOCO:
            if not self.xml_path.exists():
                raise FileNotFoundError(f"MuJoCo scene XML not found at {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.data = mujoco.MjData(self.model)
            self._ik_data = mujoco.MjData(self.model)
            self._mj = mujoco
        else:
            self._mj = _FallbackMujocoModule
            self.model = _FallbackMjModel()
            self.data = _FallbackMjData(self.model)
            self._ik_data = _FallbackMjData(self.model)

        # Target object and tray positions (dynamically synced with scene.xml)
        self.trolley_pos = [-0.5765, -0.7287, 0.0]
        self.scissors_pos = [-0.630, -0.7287, 0.028]
        self.scalpel_pos = [-0.520, -0.7287, 0.028]

        if self.xml_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(self.xml_path))
                for body in tree.findall(".//body"):
                    name = body.get("name")
                    pos_str = body.get("pos")
                    if pos_str:
                        coords = [float(v) for v in pos_str.split()]
                        if name == "instrument_trolley" and len(coords) == 3:
                            self.trolley_pos = coords
                        elif name == "scissors" and len(coords) == 3:
                            self.scissors_pos = coords
                        elif name == "scalpel" and len(coords) == 3:
                            self.scalpel_pos = coords
            except Exception:
                pass

        if hasattr(self.model, "__dict__"):
            self.model.scissors_pos = self.scissors_pos
            self.model.scalpel_pos = self.scalpel_pos

        # Cache joint & actuator IDs
        self.arm_qpos_indices = [
            self.model.joint(name).qposadr[0] for name in self.ARM_JOINT_NAMES
        ]
        self.arm_dof_indices = [
            self.model.joint(name).dofadr[0] if hasattr(self.model.joint(name), "dofadr")
            else self.model.joint(name).qposadr[0]
            for name in self.ARM_JOINT_NAMES
        ]
        self.arm_actuator_ids = [
            self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ARM_ACTUATOR_NAMES
        ]
        self.gripper_actuator_id = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, "rg2_actuator"
        )
        self.gripper_joint_idx = self.model.joint("gripper_joint").qposadr[0]
        self.gripper_dof_idx = (
            self.model.joint("gripper_joint").dofadr[0]
            if hasattr(self.model.joint("gripper_joint"), "dofadr")
            else self.gripper_joint_idx
        )

        # TCP Site
        self.tcp_site_id = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_SITE, "rg2_tcp"
        )

        # Cache geom IDs for contact detection
        self.finger_geom_ids = set()
        for name in self.FINGER_GEOM_NAMES:
            gid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.finger_geom_ids.add(gid)

        self.object_geom_ids = set()
        for name in self.OBJECT_GEOM_NAMES:
            gid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.object_geom_ids.add(gid)

        self._geom_names = {
            "hand_col": 0, "l_moment_col": 1, "l_finger_col": 2, "l_truss_col": 3,
            "r_moment_col": 4, "r_finger_col": 5, "r_truss_col": 6,
            "scalpel_handle": 10, "scalpel_blade": 11, "floor": 12,
            "scissors_shank": 20, "scissors_blade_col": 21, "scissors_handle_col": 22,
            "scissors_blade_left": 23, "scissors_blade_right": 24,
            "scissors_handle_left": 25, "scissors_handle_right": 26,
            "scissors_vis": 27, "scissors_blades_vis": 28, "scissors_handles_vis": 29,
        }

        # Load waypoints
        self.waypoints = self._load_waypoints()

        # Predefined key poses (with robust structure checking)
        default_home = [-0.0388, -1.5986, 1.5708, -1.571, -1.5708, 0.0]
        default_tray = [-0.3361, -1.7783, 2.1912, -1.9841, -1.5741, -2.0954]

        home_wp = self.waypoints.get("home_pose", {})
        self.home_pose = (
            home_wp["joints_rad"] if isinstance(home_wp, dict) and "joints_rad" in home_wp
            else default_home
        )

        tray_wp = self.waypoints.get("tray_pose", {})
        self.tray_pose = (
            tray_wp["joints_rad"] if isinstance(tray_wp, dict) and "joints_rad" in tray_wp
            else default_tray
        )

        # Trajectory & actuator targets
        self._traj_lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._active_traj: Optional[Dict[str, Any]] = None
        self._target_gripper: float = 0.65  # Default open ~100mm

        # Mock contact injection for testing
        self._simulated_contact_override = False
        self._simulated_tcp_force: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Initialize to Home Pose and open gripper
        self.set_joint_positions(self.home_pose)
        self.data.ctrl[self.gripper_actuator_id] = self._target_gripper
        self._mj.mj_forward(self.model, self.data)

        # Automatically start background physics loop if headless
        if self.headless:
            self.start()

    def _load_waypoints(self) -> Dict[str, Any]:
        """Loads repository waypoints from waypoints.json."""
        if WAYPOINTS_FILE.exists():
            try:
                with open(WAYPOINTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def start_viewer(self):
        """Launches the passive interactive 3D viewer window and starts the loop."""
        in_test = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
        if in_test:
            self.headless = True
            self.start()
            return

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            if not self.headless:
                print("[MuJoCo] Headless mode: No graphical display detected. Falling back to headless simulation.")
                self.headless = True
        elif not self.headless and self.viewer is None:
            try:
                if HAS_MUJOCO and hasattr(mujoco, "viewer"):
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    print("[MuJoCo] Interactive 3D viewer launched successfully.")
                else:
                    self.headless = True
            except Exception as e:
                print(f"[MuJoCo] Notice: Could not launch GUI viewer ({e}). Falling back to headless.")
                self.headless = True
        self.start()

    def start(self):
        """Starts the dedicated simulation loop thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def close_viewer(self):
        """Closes the simulation thread and viewer."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def is_running(self) -> bool:
        """Returns whether the simulation is actively running."""
        if self.viewer is not None and not self.viewer.is_running():
            self._running = False
        return self._running

    def is_moving(self) -> bool:
        """Returns True if an active trajectory is currently executing."""
        with self._traj_lock:
            return self._active_traj is not None

    def _run_loop(self):
        """
        Dedicated simulation thread:
        Steps physics and synchronizes viewer at 60 FPS under viewer.lock().
        """
        frame_dt = 0.016  # ~60 FPS
        steps_per_frame = max(1, int(round(frame_dt / self.model.opt.timestep)))

        while self._running:
            if self.viewer is not None and not self.viewer.is_running():
                self._running = False
                break

            t_start = time.time()

            # 1. Update arm controls based on active trajectory
            with self._traj_lock:
                if self._active_traj is not None:
                    now = time.time()
                    elapsed = now - self._active_traj["start_time"]
                    dur = self._active_traj["duration"]

                    if self._active_traj["type"] == "joint":
                        start_q = self._active_traj["start_q"]
                        target_q = self._active_traj["target_q"]

                        if elapsed >= dur:
                            for i, act_id in enumerate(self.arm_actuator_ids):
                                self.data.ctrl[act_id] = target_q[i]
                            self._active_traj["done_event"].set()
                            self._active_traj = None
                        else:
                            tau = min(1.0, elapsed / dur)
                            s = 0.5 * (1.0 - math.cos(math.pi * tau))
                            for i, act_id in enumerate(self.arm_actuator_ids):
                                self.data.ctrl[act_id] = start_q[i] + s * (target_q[i] - start_q[i])

                    elif self._active_traj["type"] == "waypoints":
                        if elapsed >= dur:
                            final_q = self._active_traj["q_points"][-1]
                            for i, act_id in enumerate(self.arm_actuator_ids):
                                self.data.ctrl[act_id] = float(final_q[i])
                            self._active_traj["done_event"].set()
                            self._active_traj = None
                        else:
                            times = self._active_traj["times"]
                            q_pts = self._active_traj["q_points"]
                            idx = max(0, min(len(times) - 2, int(np.searchsorted(times, elapsed) - 1)))
                            t0, t1 = times[idx], times[idx + 1]
                            dt = max(1e-6, t1 - t0)
                            alpha = max(0.0, min(1.0, (elapsed - t0) / dt))
                            q_interp = q_pts[idx] + alpha * (q_pts[idx + 1] - q_pts[idx])
                            for i, act_id in enumerate(self.arm_actuator_ids):
                                self.data.ctrl[act_id] = float(q_interp[i])

                # Gripper control target
                self.data.ctrl[self.gripper_actuator_id] = self._target_gripper

            # 2. Step physics safely with gravity feedforward compensation
            if HAS_MUJOCO and not isinstance(self.model, _FallbackMjModel):
                self.data.qfrc_applied[self.arm_dof_indices] = self.data.qfrc_bias[self.arm_dof_indices]

            if self.viewer is not None:
                with self.viewer.lock():
                    with self._physics_lock:
                        for _ in range(steps_per_frame):
                            self._mj.mj_step(self.model, self.data)
                self.viewer.sync()
            else:
                with self._physics_lock:
                    for _ in range(steps_per_frame):
                        self._mj.mj_step(self.model, self.data)

            # Update simulated F/T sensor forces from physical finger contacts
            if HAS_MUJOCO and hasattr(self.data, "ncon") and hasattr(self.data, "contact"):
                total_fz = 0.0
                c_force = np.zeros(6, dtype=np.float64)
                for c_i in range(min(int(self.data.ncon), len(self.data.contact))):
                    try:
                        con = self.data.contact[c_i]
                        g1, g2 = int(con.geom1), int(con.geom2)
                        if g1 in self.finger_geom_ids or g2 in self.finger_geom_ids:
                            self._mj.mj_contactForce(self.model, self.data, c_i, c_force)
                            total_fz += abs(c_force[0])
                    except Exception:
                        break
                if total_fz > 0.01:
                    self._simulated_tcp_force = [0.0, 0.0, float(total_fz), 0.0, 0.0, 0.0]
                elif not self._simulated_contact_override and self._simulated_tcp_force[2] < 5.0:
                    self._simulated_tcp_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # 3. Maintain ~60 FPS pacing
            elapsed_frame = time.time() - t_start
            sleep_time = frame_dt - elapsed_frame
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def set_joint_positions(self, q: List[float]):
        """Sets arm joint positions directly and updates controls."""
        if len(q) != len(self.arm_qpos_indices):
            raise ValueError(
                f"Joint positions must have {len(self.arm_qpos_indices)} elements, got {len(q)}"
            )

        def _set_q():
            for i, idx in enumerate(self.arm_qpos_indices):
                self.data.qpos[idx] = float(q[i])
                self.data.ctrl[self.arm_actuator_ids[i]] = float(q[i])
            self._mj.mj_forward(self.model, self.data)

        if self.viewer is not None:
            with self.viewer.lock():
                with self._physics_lock:
                    _set_q()
        else:
            with self._physics_lock:
                _set_q()

    def reset_tools(self):
        """Resets simulated surgical tools in the tray to their initial resting poses and zeroes velocity."""
        if hasattr(self, "model") and hasattr(self, "data") and HAS_MUJOCO:
            def _reset():
                s_jnt = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, "scissors_joint")
                if s_jnt >= 0:
                    s_adr = self.model.jnt_qposadr[s_jnt]
                    self.data.qpos[s_adr : s_adr + 3] = self.scissors_pos
                    self.data.qpos[s_adr + 3 : s_adr + 7] = [1.0, 0.0, 0.0, 0.0]
                    dof_adr = self.model.jnt_dofadr[s_jnt]
                    self.data.qvel[dof_adr : dof_adr + 6] = 0.0

                sc_jnt = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, "scalpel_joint")
                if sc_jnt >= 0:
                    sc_adr = self.model.jnt_qposadr[sc_jnt]
                    self.data.qpos[sc_adr : sc_adr + 3] = self.scalpel_pos
                    self.data.qpos[sc_adr + 3 : sc_adr + 7] = [1.0, 0.0, 0.0, 0.0]
                    dof_adr = self.model.jnt_dofadr[sc_jnt]
                    self.data.qvel[dof_adr : dof_adr + 6] = 0.0

                self._mj.mj_forward(self.model, self.data)

            if self.viewer is not None:
                with self.viewer.lock():
                    with self._physics_lock:
                        _reset()
            else:
                with self._physics_lock:
                    _reset()

    def set_gripper_angle(self, angle_rad: float):
        """Sets RG-2 gripper control target in radians (-0.45 closed to 0.70 open)."""
        angle_clamped = max(-0.45, min(0.70, float(angle_rad)))
        with self._traj_lock:
            self._target_gripper = angle_clamped

    def get_actual_q(self) -> List[float]:
        """Returns current arm joint angles in radians."""
        if self.viewer is not None:
            with self.viewer.lock():
                with self._physics_lock:
                    return [float(self.data.qpos[idx]) for idx in self.arm_qpos_indices]
        with self._physics_lock:
            return [float(self.data.qpos[idx]) for idx in self.arm_qpos_indices]

    def get_actual_q_deg(self) -> List[float]:
        """Returns current arm joint angles in degrees."""
        return [round(math.degrees(q), 2) for q in self.get_actual_q()]

    def get_actual_tcp_pose(self) -> List[float]:
        """
        Computes forward kinematics of the RG-2 TCP site.
        Returns: [X, Y, Z, Rx, Ry, Rz] in meters and axis-angle radians.
        """
        if self.viewer is not None:
            with self.viewer.lock():
                pos = self.data.site_xpos[self.tcp_site_id].copy()
                mat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        else:
            pos = self.data.site_xpos[self.tcp_site_id].copy()
            mat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()

        rot_vec = rotmat_to_rotvec(mat)

        return [float(pos[0]), float(pos[1]), float(pos[2])] + [
            float(v) for v in rot_vec
        ]

    def _compute_fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes forward kinematics of the TCP site for arm joint angles q using dedicated ik_data."""
        if HAS_MUJOCO and not isinstance(self.model, _FallbackMjModel):
            for i, idx in enumerate(self.arm_qpos_indices):
                self._ik_data.qpos[idx] = float(q[i])
            self._mj.mj_fwdPosition(self.model, self._ik_data)
            pos = self._ik_data.site_xpos[self.tcp_site_id].copy()
            mat = self._ik_data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
            return pos, mat
        else:
            return _analytical_ur10e_rg2_fk(q)

    def _compute_tcp_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Computes the 6x6 TCP site Jacobian for arm joints using dedicated ik_data."""
        if HAS_MUJOCO and not isinstance(self.model, _FallbackMjModel):
            for i, idx in enumerate(self.arm_qpos_indices):
                self._ik_data.qpos[idx] = float(q[i])
            self._mj.mj_fwdPosition(self.model, self._ik_data)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            self._mj.mj_jacSite(self.model, self._ik_data, jacp, jacr, self.tcp_site_id)
            return np.vstack([jacp[:, self.arm_dof_indices], jacr[:, self.arm_dof_indices]])
        else:
            return _numerical_ur10e_rg2_jacobian(q)

    # -----------------------------------------------------------------------
    # Arm Trajectory Execution (moveJ, moveL, poseTrans)
    # -----------------------------------------------------------------------

    def stop(self) -> bool:
        """Immediately halts any active trajectory and holds current position."""
        with self._traj_lock:
            if self._active_traj is not None:
                self._active_traj["done_event"].set()
                self._active_traj = None
        current_q = self.get_actual_q()
        for i, act_id in enumerate(self.arm_actuator_ids):
            self.data.ctrl[act_id] = float(current_q[i])
        return True

    def get_inverse_kinematics(
        self,
        target_tcp: List[float],
        qnear: Optional[List[float]] = None,
        max_steps: int = 50,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-3,
    ) -> List[float]:
        """
        Solves numerical inverse kinematics for target_tcp [X, Y, Z, Rx, Ry, Rz].
        Returns 6-DoF joint list if converged, or empty list [] if unreachable.
        """
        if len(target_tcp) != 6:
            raise ValueError(f"target_tcp must have 6 elements, got {len(target_tcp)}")

        p_des = np.array(target_tcp[:3], dtype=float)
        r_des = rotvec_to_rotmat(np.array(target_tcp[3:6], dtype=float))

        q = np.array(qnear if qnear is not None else self.get_actual_q(), dtype=float)
        damping = 0.01

        for _ in range(max_steps):
            curr_p, curr_r = self._compute_fk(q)
            err_p = p_des - curr_p
            err_r = rotmat_to_rotvec(r_des @ curr_r.T)
            err_6d = np.concatenate([err_p, err_r])

            if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < rot_tol:
                return [float(val) for val in q]

            jacobian = self._compute_tcp_jacobian(q)
            reg = damping**2 * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + reg, err_6d)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 0.2:
                dq = dq * (0.2 / dq_norm)
            q = q + dq

        curr_p, curr_r = self._compute_fk(q)
        if np.linalg.norm(p_des - curr_p) < 0.01 and np.linalg.norm(rotmat_to_rotvec(r_des @ curr_r.T)) < 0.05:
            return [float(val) for val in q]
        return []

    def moveJ(
        self,
        target_q: List[float],
        speed: float = 0.3,
        acceleration: float = 0.4,
        duration: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        Moves the 6-DoF arm in joint space using a smooth, wall-clock synchronized S-curve.
        Eliminates teleportation by executing smoothly via the dedicated simulation thread.
        Calculates duration dynamically if omitted: max(0.5, max_joint_delta / speed).
        """
        if len(target_q) != len(self.arm_qpos_indices):
            raise ValueError(
                f"target_q must have {len(self.arm_qpos_indices)} elements, got {len(target_q)}"
            )

        if not self._running:
            self.start()

        target_q = [float(x) for x in target_q]
        current_q = self.get_actual_q()

        max_diff = max(abs(t - c) for t, c in zip(target_q, current_q))
        if max_diff < 1e-4:
            return True

        if duration is None:
            eff_speed = max(0.01, float(speed))
            duration = max(0.5, max_diff / eff_speed)
        else:
            duration = max(0.05, float(duration))

        done_event = threading.Event()
        with self._traj_lock:
            if self._active_traj is not None:
                self._active_traj["done_event"].set()
            self._active_traj = {
                "type": "joint",
                "start_q": current_q,
                "target_q": target_q,
                "start_time": time.time(),
                "duration": duration,
                "done_event": done_event,
            }

        if wait:
            done_event.wait(timeout=duration + 3.0)
            time.sleep(0.1)  # Settle time
        return True

    def moveL(
        self,
        target_tcp: List[float],
        speed: float = 0.05,
        acceleration: float = 0.1,
        duration: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        Moves the TCP linearly in Cartesian space using numerical Jacobian / differential IK.
        Achieves a smooth straight-line linear trajectory.
        Calculates duration dynamically if omitted: max(0.5, dist / speed).
        """
        if len(target_tcp) != 6:
            raise ValueError(
                f"target_tcp must have 6 elements [x, y, z, rx, ry, rz], got {len(target_tcp)}"
            )

        if not self._running:
            self.start()

        current_tcp = self.get_actual_tcp_pose()
        p0 = np.array(current_tcp[:3], dtype=float)
        p1 = np.array(target_tcp[:3], dtype=float)
        dist = float(np.linalg.norm(p1 - p0))

        r0 = np.array(current_tcp[3:6], dtype=float)
        r1 = np.array(target_tcp[3:6], dtype=float)
        r0_mat = rotvec_to_rotmat(r0)
        r1_mat = rotvec_to_rotmat(r1)
        r_diff_mat = r1_mat @ r0_mat.T
        r_diff = R.from_matrix(r_diff_mat).as_rotvec()
        rot_dist = float(np.linalg.norm(r_diff))

        if dist < 1e-4 and rot_dist < 1e-4:
            return True

        if duration is None:
            eff_speed = max(0.005, float(speed))
            t_trans = dist / eff_speed
            t_rot = rot_dist / max(0.05, eff_speed * 2.0)
            duration = max(0.5, t_trans, t_rot)
        else:
            duration = max(0.05, float(duration))

        # Setup spherical linear interpolation (SLERP) for smooth orientation without branch-cut jumps
        key_times = [0.0, 1.0]
        key_rots = R.from_matrix([r0_mat, r1_mat])
        slerp = Slerp(key_times, key_rots)

        # Discretize Cartesian straight line into waypoints
        num_steps = max(20, int(duration * 40))
        q = np.array(self.get_actual_q(), dtype=float)
        q_points = [q.copy()]
        t_points = [0.0]

        for k in range(1, num_steps + 1):
            tau = k / num_steps
            s = 0.5 * (1.0 - math.cos(math.pi * tau))  # S-curve parameter
            p_des = p0 + s * (p1 - p0)
            r_des = slerp([s]).as_matrix()[0]

            # DLS differential IK solver
            for _ in range(12):
                curr_p, curr_r = self._compute_fk(q)
                err_p = p_des - curr_p
                err_r = R.from_matrix(r_des @ curr_r.T).as_rotvec()
                err_6d = np.concatenate([err_p, err_r])
                if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-4:
                    break
                jacobian = self._compute_tcp_jacobian(q)
                damping = 0.01
                reg = damping**2 * np.eye(6)
                dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + reg, err_6d)
                dq_norm = float(np.linalg.norm(dq))
                if dq_norm > 0.15:
                    dq = dq * (0.15 / dq_norm)
                q = q + dq

            q_points.append(q.copy())
            t_points.append(tau * duration)

        # Check reachability of the final calculated waypoint
        final_p, final_r = self._compute_fk(q)
        final_err_p = float(np.linalg.norm(p1 - final_p))
        final_err_r = float(np.linalg.norm(R.from_matrix(r1_mat @ final_r.T).as_rotvec()))
        if final_err_p > 0.05 or final_err_r > 0.20:
            print(
                f"[WARN] moveL: Target TCP unreachable "
                f"(pos err: {final_err_p:.4f} m, rot err: {final_err_r:.4f} rad)"
            )
            return False

        done_event = threading.Event()
        with self._traj_lock:
            if self._active_traj is not None:
                self._active_traj["done_event"].set()
            self._active_traj = {
                "type": "waypoints",
                "times": t_points,
                "q_points": q_points,
                "start_time": time.time(),
                "duration": duration,
                "done_event": done_event,
            }

        if wait:
            done_event.wait(timeout=duration + 3.0)
            time.sleep(0.1)
        return True

    def poseTrans(self, p_from: List[float], p_offset: List[float]) -> List[float]:
        """
        Exact spatial transform matching URScript / ur_rtde `poseTrans(p_from, p_offset)`.
        Computes T_result = T_from * T_offset.
        """
        return pose_trans(p_from, p_offset)

    def move_to_tray(self, duration: float = 2.5) -> bool:
        """Smoothly moves UR-10e arm to the surgical tray pose."""
        print("[Motion] Moving arm smoothly to Tray Pose...")
        return self.moveJ(self.tray_pose, duration=duration)

    def move_to_home(self, duration: float = 2.5) -> bool:
        """Smoothly returns UR-10e arm to the home standby pose."""
        print("[Motion] Returning arm smoothly to Home Pose...")
        return self.moveJ(self.home_pose, duration=duration)

    # -----------------------------------------------------------------------
    # RG-2 Gripper Control & State Sensing
    # -----------------------------------------------------------------------

    def _check_grip_detected(self) -> bool:
        """Checks MuJoCo contacts between finger geoms and object geoms."""
        if self._simulated_contact_override:
            return True

        current_q = float(self.data.qpos[self.gripper_joint_idx])
        width_mm = (current_q - (-0.45)) / 1.15 * 110.0

        with self._physics_lock:
            if hasattr(self.data, "ncon") and hasattr(self.data, "contact"):
                ncon = int(self.data.ncon)
                for i in range(min(ncon, len(self.data.contact))):
                    try:
                        con = self.data.contact[i]
                        g1, g2 = int(con.geom1), int(con.geom2)
                        if (g1 in self.finger_geom_ids and g2 in self.object_geom_ids) or \
                           (g2 in self.finger_geom_ids and g1 in self.object_geom_ids):
                            # The actual tool clamping occurs when the gripper closes to the tool diameter
                            # Scissors shank is 14.0 mm, scalpel neck is 12.0 mm
                            if 1.0 <= width_mm <= 16.5:
                                return True
                    except (IndexError, AttributeError):
                        break

        # Kinematic proximity check for simulation:
        # Detect physical clamp when gripper fingers close on scissors shank or scalpel
        tcp = self.get_actual_tcp_pose()
        dist_scissors = math.sqrt((tcp[0] - self.scissors_pos[0]) ** 2 + (tcp[1] - self.scissors_pos[1]) ** 2)
        dist_scalpel = math.sqrt((tcp[0] - self.scalpel_pos[0]) ** 2 + (tcp[1] - self.scalpel_pos[1]) ** 2)
        contact_z = self.waypoints.get("padded_contact_pose", {}).get("tcp_pose", [0, 0, 0.0151])[2]
        dist_z = abs(tcp[2] - contact_z)

        if dist_z < 0.040:
            if dist_scissors < 0.060 and dist_scissors <= dist_scalpel and (1.0 <= width_mm <= 16.5):
                return True
            if dist_scalpel < 0.060 and dist_scalpel < dist_scissors and (1.0 <= width_mm <= 14.5):
                return True

        return False

    def get_gripper_state(self) -> dict:
        """
        Returns current RG-2 gripper state matching physical OnRobot RG2 register layout.
        Returns:
            {
                "width_mm": float,
                "status_raw": int,
                "busy": bool,
                "grip_detected": bool,
                "safety_circuit_active": False,
            }
        """
        current_q = float(self.data.qpos[self.gripper_joint_idx])
        # Joint -0.45 rad is 0.0 mm, +0.70 rad is 110.0 mm
        width_mm = (current_q - (-0.45)) / 1.15 * 110.0
        width_mm = max(0.0, min(110.0, width_mm))

        grip_detected = self._check_grip_detected()

        vel = 0.0
        if hasattr(self.data, "qvel") and len(self.data.qvel) > self.gripper_dof_idx:
            vel = abs(float(self.data.qvel[self.gripper_dof_idx]))

        busy = (vel > 0.01 or abs(current_q - self._target_gripper) > 0.02) and not grip_detected
        status_raw = (1 if busy else 0) | (2 if grip_detected else 0)

        return {
            "width_mm": round(width_mm, 1),
            "status_raw": status_raw,
            "busy": busy,
            "grip_detected": grip_detected,
            "safety_circuit_active": False,
        }

    def move_gripper(
        self,
        width_mm: float,
        force_n: float = 20.0,
        wait: bool = True,
    ) -> dict:
        """
        Commands RG-2 gripper to target width in mm (0.0 to 110.0) with force limit in N.
        Maps width linearly to joint range [-0.45, 0.70] rad.
        """
        if not self._running:
            self.start()

        width_clamped = max(0.0, min(110.0, float(width_mm)))
        force_clamped = max(3.0, min(40.0, float(force_n)))

        # Linear mapping: 0.0mm -> -0.45 rad, 110.0mm -> +0.70 rad
        target_q = -0.45 + (width_clamped / 110.0) * 1.15
        self.set_gripper_angle(target_q)

        # Update actuator force range if supported
        if hasattr(self.model, "actuator_forcerange"):
            self.model.actuator_forcerange[self.gripper_actuator_id] = [-force_clamped, force_clamped]

        if wait:
            start_t = time.time()
            while time.time() - start_t < 2.0:
                current_q = float(self.data.qpos[self.gripper_joint_idx])
                closing = target_q < current_q
                if abs(current_q - target_q) < 0.02 or (closing and self._check_grip_detected()):
                    if closing and self._check_grip_detected():
                        self.set_gripper_angle(current_q)
                    break
                time.sleep(0.05)

        return self.get_gripper_state()

    def open_gripper(self, force_n: float = 20.0, duration: float = 1.0) -> dict:
        """Smoothly opens the OnRobot RG-2 gripper (110 mm)."""
        if not self._running:
            self.start()
        print("[Gripper] Opening RG-2 gripper...")
        state = self.move_gripper(110.0, force_n=force_n, wait=True)
        if duration > 0.0:
            time.sleep(min(0.2, duration))
        return state

    def close_gripper(self, force_n: float = 20.0, duration: float = 1.0) -> dict:
        """Smoothly closes the OnRobot RG-2 gripper (0 mm)."""
        if not self._running:
            self.start()
        print("[Gripper] Closing RG-2 gripper...")
        state = self.move_gripper(0.0, force_n=force_n, wait=True)
        if duration > 0.0:
            time.sleep(min(0.2, duration))
        return state

    # -----------------------------------------------------------------------
    # Camera Capture & 3D Vision Rendering
    # -----------------------------------------------------------------------

    def capture_camera_frame(
        self,
        camera_name: str = "wrist_realsense",
        width: int = 640,
        height: int = 480,
    ) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """
        Captures aligned color and depth frames from the specified simulated camera.
        If native MuJoCo and OpenGL context is available, uses mujoco.Renderer.
        Otherwise, uses an integrated optical pinhole perspective projection rasterizer.

        Returns:
            Tuple of:
                - color_bgr (np.ndarray): BGR uint8 image [H, W, 3].
                - depth_m (np.ndarray): Depth map in meters [H, W] float32.
                - intrinsics (CameraIntrinsics): Calibrated optical pinhole parameters.
        """
        if HAS_MUJOCO and not isinstance(self.model, _FallbackMjModel):
            try:
                if not hasattr(self, "_renderer") or self._renderer is None:
                    self._renderer = mujoco.Renderer(self.model, height=height, width=width)
                elif self._renderer.height != height or self._renderer.width != width:
                    self._renderer = mujoco.Renderer(self.model, height=height, width=width)

                if self.viewer is not None:
                    with self.viewer.lock():
                        self._renderer.update_scene(self.data, camera=camera_name)
                else:
                    self._renderer.update_scene(self.data, camera=camera_name)

                rgb = self._renderer.render()
                color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                self._renderer.enable_depth_rendering()
                depth_m = self._renderer.render().astype(np.float32)
                self._renderer.disable_depth_rendering()

                fovy_deg = 58.0
                cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, camera_name)
                if cid >= 0 and hasattr(self.model, "cam_fovy"):
                    fovy_deg = float(self.model.cam_fovy[cid])

                fovy_rad = math.radians(fovy_deg)
                fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
                fx = fy
                intrinsics = CameraIntrinsics(
                    fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height
                )
                return color_bgr, depth_m, intrinsics
            except Exception:
                pass

        return self._render_pinhole_perspective_frame(camera_name, width, height)

    def _render_pinhole_perspective_frame(
        self,
        camera_name: str = "wrist_realsense",
        width: int = 640,
        height: int = 480,
    ) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """
        Integrated optical pinhole perspective projection rasterizer.
        Generates realistic aligned color (BGR) and depth (float32 meters) maps
        matching the physical RealSense D435 camera mounted on hand_link.
        """
        fovy_deg = 58.0
        fovy_rad = math.radians(fovy_deg)
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
        )

        color_bgr = np.full((height, width, 3), (45, 45, 50), dtype=np.uint8)
        depth_m = np.full((height, width), 0.50, dtype=np.float32)

        # Retrieve TCP pose from forward kinematics
        p_tcp = self.get_actual_tcp_pose()[:3]

        tray_wp = self.waypoints.get("tray_pose", {})
        tray_tcp = (
            tray_wp["tcp_pose"]
            if isinstance(tray_wp, dict) and "tcp_pose" in tray_wp
            else [-0.5338, -0.745, 0.1906]
        )

        # When arm is at tray_pose (or small delta from tray_pose), track delta
        tray_joint_diff = np.linalg.norm(np.array(self.get_actual_q()) - np.array(self.tray_pose))
        if tray_joint_diff < 0.05:
            delta_x = 0.0
            delta_y = 0.0
            delta_z = 0.0
        else:
            delta_x = p_tcp[0] - tray_tcp[0]
            delta_y = p_tcp[1] - tray_tcp[1]
            delta_z = p_tcp[2] - tray_tcp[2]

        cam_center_x = self.trolley_pos[0] + delta_x
        cam_center_y = self.trolley_pos[1] + delta_y
        cam_height = tray_tcp[2] + delta_z

        def project_pt(x_w: float, y_w: float, z_w: float) -> Tuple[int, int, float]:
            z_cam = max(0.04, cam_height - z_w)
            x_cam = x_w - cam_center_x
            y_cam = y_w - cam_center_y
            u = int(round(cx + (x_cam * fx) / z_cam))
            v = int(round(cy + (y_cam * fy) / z_cam))
            return u, v, z_cam

        # 1. Trolley table top (center trolley_pos, size 0.40 x 0.32)
        u_t1, v_t1, d_t = project_pt(self.trolley_pos[0] - 0.20, self.trolley_pos[1] - 0.16, 0.011)
        u_t2, v_t2, _ = project_pt(self.trolley_pos[0] + 0.20, self.trolley_pos[1] + 0.16, 0.011)
        tx1, tx2 = max(0, min(u_t1, u_t2)), min(width, max(u_t1, u_t2))
        ty1, ty2 = max(0, min(v_t1, v_t2)), min(height, max(v_t1, v_t2))
        if tx2 > tx1 and ty2 > ty1:
            color_bgr[ty1:ty2, tx1:tx2] = (145, 145, 150)
            depth_m[ty1:ty2, tx1:tx2] = float(d_t)

        # 2. Tray on trolley (center trolley_pos, size 0.32 x 0.24, floor 0.022)
        u_tr1, v_tr1, d_tr = project_pt(self.trolley_pos[0] - 0.16, self.trolley_pos[1] - 0.12, 0.022)
        u_tr2, v_tr2, _ = project_pt(self.trolley_pos[0] + 0.16, self.trolley_pos[1] + 0.12, 0.022)
        trx1, trx2 = max(0, min(u_tr1, u_tr2)), min(width, max(u_tr1, u_tr2))
        try1, try2 = max(0, min(v_tr1, v_tr2)), min(height, max(v_tr1, v_tr2))
        if trx2 > trx1 and try2 > try1:
            # Sterile surgical drape blue (BGR: ~#042850 -> (80, 40, 4))
            color_bgr[try1:try2, trx1:trx2] = (80, 40, 4)
            cv2.rectangle(color_bgr, (trx1, try1), (trx2, try2), (60, 30, 3), 3)
            depth_m[try1:try2, trx1:trx2] = float(d_tr)

        # 3. Scalpel resting alongside in tray
        # Authentic stainless steel surgical scalpel (blade at +y/+v, ribbed handle at -y/-v)
        u_sc, v_sc, d_sc = project_pt(self.scalpel_pos[0], self.scalpel_pos[1], self.scalpel_pos[2])
        if 0 <= u_sc < width and 0 <= v_sc < height:
            # Steel handle (-y, -v)
            cv2.line(color_bgr, (u_sc, v_sc - 5), (u_sc, v_sc - 65), (210, 210, 215), 10)
            # Steel blade (+y, +v)
            cv2.line(color_bgr, (u_sc, v_sc), (u_sc, v_sc + 35), (235, 235, 240), 4)
            r_box = 18
            sy1, sy2 = max(0, v_sc - 70), min(height, v_sc + 40)
            sx1, sx2 = max(0, u_sc - r_box), min(width, u_sc + r_box)
            depth_m[sy1:sy2, sx1:sx2] = np.minimum(depth_m[sy1:sy2, sx1:sx2], float(d_sc))

        # 4. Kitchen Scissors in tray
        # Shank is at (su, sv). Blades are at +y (+v, steel). Handles are at -y (-v, red).
        su, sv, d_s = project_pt(self.scissors_pos[0], self.scissors_pos[1], self.scissors_pos[2])
        if 0 <= su < width and 0 <= sv < height:
            # Red handle loops pointing towards -v (-y in world)
            cv2.ellipse(color_bgr, (su - 14, sv - 60), (10, 24), 10, 0, 360, (25, 25, 210), -1)
            cv2.ellipse(color_bgr, (su + 14, sv - 60), (10, 24), -10, 0, 360, (25, 25, 210), -1)
            # Inner loop cutouts (sterile blue tray color)
            cv2.ellipse(color_bgr, (su - 14, sv - 60), (5, 15), 10, 0, 360, (80, 40, 4), -1)
            cv2.ellipse(color_bgr, (su + 14, sv - 60), (5, 15), -10, 0, 360, (80, 40, 4), -1)

            # Central shank and neck connections (pivot box-lock cylinder, r=7mm)
            cv2.line(color_bgr, (su - 8, sv - 35), (su, sv), (215, 215, 220), 8)
            cv2.line(color_bgr, (su + 8, sv - 35), (su, sv), (215, 215, 220), 8)
            cv2.circle(color_bgr, (su, sv), 8, (215, 215, 220), -1)
            cv2.circle(color_bgr, (su, sv), 3, (160, 160, 165), -1)

            # Stainless steel blades pointing towards +v (+y in world)
            cv2.line(color_bgr, (su - 3, sv), (su - 4, sv + 45), (225, 225, 230), 4)
            cv2.line(color_bgr, (su + 3, sv), (su + 4, sv + 45), (225, 225, 230), 4)

            # Depth map for scissors region
            sy1, sy2 = max(0, sv - 88), min(height, sv + 52)
            sx1, sx2 = max(0, su - 28), min(width, su + 28)
            depth_m[sy1:sy2, sx1:sx2] = np.minimum(depth_m[sy1:sy2, sx1:sx2], float(d_s))

        return color_bgr, depth_m, intrinsics
