#!/usr/bin/env python3
"""
MuJoCo Simulation Controller for Universal Robots UR-10e and OnRobot RG-2 Gripper.

Provides high-fidelity physics simulation with authentic CAD models,
wall-clock synchronized trajectory execution (no teleportation),
and live state monitoring via a dedicated single-threaded simulation loop.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError("MuJoCo is not installed in the active environment. Please run: pip install mujoco")

# Default paths dynamically resolved relative to this module
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
DEFAULT_SCENE_XML = CURRENT_DIR.parent / "mujoco" / "models" / "scene.xml"
WAYPOINTS_FILE = CURRENT_DIR.parent / "waypoints.json"


class MujocoSimController:
    """
    Simulation controller interfacing with MuJoCo for both UR-10e arm and RG-2 gripper.
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

    def __init__(
        self,
        scene_xml: Optional[Path | str] = None,
        headless: bool = False,
    ):
        self.xml_path = Path(scene_xml) if scene_xml else DEFAULT_SCENE_XML
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo scene XML not found at {self.xml_path}")

        self.headless = headless
        self.viewer = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Load MuJoCo Model and Data
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Cache joint & actuator IDs
        self.arm_qpos_indices = [
            self.model.joint(name).qposadr[0] for name in self.ARM_JOINT_NAMES
        ]
        self.arm_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.ARM_ACTUATOR_NAMES
        ]
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg2_actuator"
        )
        self.gripper_joint_idx = self.model.joint("gripper_joint").qposadr[0]

        # TCP Site
        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "rg2_tcp"
        )

        # Load waypoints
        self.waypoints = self._load_waypoints()

        # Predefined key poses
        self.home_pose = self.waypoints.get(
            "home_pose",
            {"joints_rad": [-0.0388, -1.5986, 1.5708, -1.571, -1.5708, 0.0]},
        )["joints_rad"]

        self.tray_pose = self.waypoints.get(
            "tray_pose",
            {"joints_rad": [-0.4026, -1.6247, 2.2264, -2.1839, -1.5986, -2.0028]},
        )["joints_rad"]

        # Trajectory & actuator targets
        self._traj_lock = threading.Lock()
        self._active_traj: Optional[Dict[str, Any]] = None
        self._target_gripper: float = 0.5

        # Initialize to Home Pose and open gripper
        self.set_joint_positions(self.home_pose)
        self.data.ctrl[self.gripper_actuator_id] = self._target_gripper
        mujoco.mj_forward(self.model, self.data)

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
        if not self.headless and self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print("[MuJoCo] Interactive 3D viewer launched successfully.")
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

    def _run_loop(self):
        """
        Dedicated simulation thread:
        Steps physics and synchronizes the viewer at 60 FPS under viewer.lock().
        Guarantees that no concurrent threads ever access mjData or viewer.sync(),
        completely eliminating 'attempting to copy mjData while stack is in use'.
        """
        frame_dt = 0.016  # ~60 FPS
        # 8 steps of 0.002s = 0.016s of physics per 0.016s of real wall-clock time
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

                # Gripper control target
                self.data.ctrl[self.gripper_actuator_id] = self._target_gripper

            # 2. Step physics and sync viewer safely
            if self.viewer is not None:
                with self.viewer.lock():
                    for _ in range(steps_per_frame):
                        mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
            else:
                for _ in range(steps_per_frame):
                    mujoco.mj_step(self.model, self.data)

            # 3. Maintain ~60 FPS pacing
            elapsed_frame = time.time() - t_start
            sleep_time = frame_dt - elapsed_frame
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def set_joint_positions(self, q: List[float]):
        """Sets arm joint positions directly and updates controls."""
        if self.viewer is not None:
            with self.viewer.lock():
                for i, idx in enumerate(self.arm_qpos_indices):
                    self.data.qpos[idx] = float(q[i])
                    self.data.ctrl[self.arm_actuator_ids[i]] = float(q[i])
                mujoco.mj_forward(self.model, self.data)
        else:
            for i, idx in enumerate(self.arm_qpos_indices):
                self.data.qpos[idx] = float(q[i])
                self.data.ctrl[self.arm_actuator_ids[i]] = float(q[i])
            mujoco.mj_forward(self.model, self.data)

    def set_gripper_angle(self, angle_rad: float):
        """Sets RG-2 gripper control target in radians (-0.45 closed to 0.7 open)."""
        angle_clamped = max(-0.45, min(0.75, float(angle_rad)))
        with self._traj_lock:
            self._target_gripper = angle_clamped

    def get_actual_q(self) -> List[float]:
        """Returns current arm joint angles in radians."""
        if self.viewer is not None:
            with self.viewer.lock():
                return [float(self.data.qpos[idx]) for idx in self.arm_qpos_indices]
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

        trace = np.trace(mat)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = math.acos(cos_theta)
        if abs(theta) < 1e-6:
            rot_vec = [0.0, 0.0, 0.0]
        else:
            rx = (mat[2, 1] - mat[1, 2]) / (2.0 * math.sin(theta))
            ry = (mat[0, 2] - mat[2, 0]) / (2.0 * math.sin(theta))
            rz = (mat[1, 0] - mat[0, 1]) / (2.0 * math.sin(theta))
            rot_vec = [rx * theta, ry * theta, rz * theta]

        return [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)] + [
            round(float(v), 4) for v in rot_vec
        ]

    def moveJ(
        self,
        target_q: List[float],
        duration: float = 2.5,
    ) -> bool:
        """
        Moves the 6-DoF arm in joint space using a smooth, wall-clock synchronized S-curve.
        Eliminates teleportation by executing smoothly via the dedicated simulation thread.
        """
        if not self._running:
            self.start()

        target_q = [float(x) for x in target_q]
        current_q = self.get_actual_q()

        max_diff = max(abs(t - c) for t, c in zip(target_q, current_q))
        if max_diff < 1e-4:
            return True

        done_event = threading.Event()
        with self._traj_lock:
            self._active_traj = {
                "start_q": current_q,
                "target_q": target_q,
                "start_time": time.time(),
                "duration": max(0.05, float(duration)),
                "done_event": done_event,
            }

        # Wait until trajectory interpolation completes
        done_event.wait(timeout=duration + 3.0)
        time.sleep(0.4)  # Settle time
        return True

    def move_to_tray(self, duration: float = 2.5) -> bool:
        """Smoothly moves UR-10e arm to the surgical tray pose."""
        print("[Motion] Moving arm smoothly to Tray Pose...")
        return self.moveJ(self.tray_pose, duration=duration)

    def move_to_home(self, duration: float = 2.5) -> bool:
        """Smoothly returns UR-10e arm to the home standby pose."""
        print("[Motion] Returning arm smoothly to Home Pose...")
        return self.moveJ(self.home_pose, duration=duration)

    def open_gripper(self, duration: float = 1.0) -> None:
        """Smoothly opens the OnRobot RG-2 gripper."""
        if not self._running:
            self.start()
        print("[Gripper] Opening RG-2 gripper...")
        self.set_gripper_angle(0.65)
        time.sleep(duration)

    def close_gripper(self, duration: float = 1.0) -> None:
        """Smoothly closes the OnRobot RG-2 gripper."""
        if not self._running:
            self.start()
        print("[Gripper] Closing RG-2 gripper...")
        self.set_gripper_angle(-0.35)
        time.sleep(duration)
