#!/usr/bin/env python3
"""
Oriented Grasp Pose Estimator for Universal Robots UR10e & OnRobot RG-2.

Extracts principal geometric axes via PCA and Minimum Area Bounding Boxes,
calculates affordance-safe shank grip points, deprojects pixels to 3D camera
space, and maps coordinates to UR10e base space using camera_extrinsics.json.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.vision.detector import DetectionResult
from src.vision.realsense_stream import CameraIntrinsics

logger = logging.getLogger("vision.grasp_estimator")

DEFAULT_EXTRINSICS_PATH = (
    Path(__file__).resolve().parent.parent / "actuators" / "camera_extrinsics.json"
)


@dataclass
class GraspEstimate:
    """Comprehensive grasp pose and kinematic target definition."""
    label: str
    confidence: float

    # 2D Image Space
    pixel_grasp: Tuple[float, float]         # (u, v) grasp target
    pixel_tip: Tuple[float, float]           # (u, v) working tip
    pixel_tail: Tuple[float, float]          # (u, v) handle base
    yaw_deg: float                           # In-plane orientation [-180, 180] deg
    tool_length_px: float                    # Measured length in pixels
    tool_width_px: float                     # Measured thickness across minor axis

    # 3D Camera Frame (Meters)
    camera_coords_m: Tuple[float, float, float]  # (X_cam, Y_cam, Z_cam)

    # UR10e Cartesian Base Space [X, Y, Z, Rx, Ry, Rz]
    robot_approach_pose: List[float]         # Scan / approach height
    robot_pick_pose: List[float]             # Table contact pick height

    # Gripper Actuation
    gripper_open_width_mm: float             # Recommended pre-grasp finger opening


class GraspEstimator:
    """
    Transforms 2D object masks into executable UR10e pick poses.
    """

    def __init__(
        self,
        extrinsics_path: Optional[Path] = None,
        camera_base_tcp: Optional[List[float]] = None,
    ) -> None:
        self.extrinsics_path = Path(extrinsics_path or DEFAULT_EXTRINSICS_PATH)
        self.dx_m = -0.04309
        self.dy_m = 0.01528
        self.dz_m = -0.14484
        self.scan_height_m = 0.17045
        self.contact_height_m = 0.0151
        self.camera_base_tcp = [-0.54601, 0.0026, 0.17045, -0.30124, 3.1271, -0.00019]
        self.calib_base_tcp = list(self.camera_base_tcp)

        # 6-DOF Hand-Eye Extrinsics (Eye-in-Hand: Tool Flange/TCP to Camera)
        self.hand_eye_R: Optional[np.ndarray] = None
        self.hand_eye_t: Optional[np.ndarray] = None
        self.hand_eye_T: Optional[np.ndarray] = None

        self._load_extrinsics()

        if camera_base_tcp is not None:
            self.set_camera_base_tcp(camera_base_tcp)
        else:
            wp_path = self.extrinsics_path.parent / "waypoints.json"
            if wp_path.exists():
                try:
                    with open(wp_path, "r", encoding="utf-8") as f:
                        wps = json.load(f)
                    if "tray_pose" in wps and "tcp_pose" in wps["tray_pose"]:
                        self.set_camera_base_tcp(wps["tray_pose"]["tcp_pose"])
                except Exception:
                    pass

    def set_camera_base_tcp(self, tcp_pose: List[float]) -> None:
        """Sets active camera reference TCP pose and dynamically updates pick heights."""
        self.camera_base_tcp = [float(v) for v in tcp_pose]
        self.scan_height_m = float(tcp_pose[2])

    def _load_extrinsics(self) -> None:
        if not self.extrinsics_path.exists():
            logger.warning("Extrinsics config not found at %s. Using default offsets.", self.extrinsics_path)
            return

        try:
            with open(self.extrinsics_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            trans = data.get("translation_offset_m", {})
            self.dx_m = float(trans.get("dx", self.dx_m))
            self.dy_m = float(trans.get("dy", self.dy_m))
            self.dz_m = float(trans.get("dz", self.dz_m))

            notes = data.get("notes", {})
            self.scan_height_m = float(notes.get("calibrated_scan_height_tcp_z_m", self.scan_height_m))
            self.contact_height_m = float(notes.get("table_contact_height_tcp_z_m", self.contact_height_m))

            poses = data.get("calibration_poses", {})
            if "camera_tcp" in poses:
                self.calib_base_tcp = [float(v) for v in poses["camera_tcp"]]
                self.camera_base_tcp = list(self.calib_base_tcp)

            # 6-DOF Hand-Eye Transformation Matrix (TCP to Camera)
            he_data = data.get("hand_eye_transform_tcp_to_cam", {})
            if "matrix_4x4" in he_data:
                self.hand_eye_T = np.array(he_data["matrix_4x4"], dtype=np.float64)
                self.hand_eye_R = self.hand_eye_T[:3, :3]
                self.hand_eye_t = self.hand_eye_T[:3, 3]
            elif "rotation_matrix" in he_data and "translation_m" in he_data:
                self.hand_eye_R = np.array(he_data["rotation_matrix"], dtype=np.float64)
                self.hand_eye_t = np.array(he_data["translation_m"], dtype=np.float64)
                self.hand_eye_T = np.eye(4, dtype=np.float64)
                self.hand_eye_T[:3, :3] = self.hand_eye_R
                self.hand_eye_T[:3, 3] = self.hand_eye_t
            else:
                # Default fallback derived from calibration poses
                r_calib = R.from_rotvec(self.calib_base_tcp[3:6]).as_matrix()
                r_cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
                self.hand_eye_R = r_calib.T @ r_cam
                d = float(self.calib_base_tcp[2]) - float(self.contact_height_m)
                p_tcp_dot = r_calib.T @ (np.array([-0.5891, 0.01788, 0.02561]) - np.array(self.calib_base_tcp[:3]))
                self.hand_eye_t = p_tcp_dot - self.hand_eye_R @ np.array([0.0, 0.0, d])
                self.hand_eye_t[2] = 0.0
                self.hand_eye_T = np.eye(4, dtype=np.float64)
                self.hand_eye_T[:3, :3] = self.hand_eye_R
                self.hand_eye_T[:3, 3] = self.hand_eye_t

            logger.info("Loaded camera extrinsics: ΔX=%.1fmm, ΔY=%.1fmm, ΔZ=%.1fmm (6-DOF Hand-Eye active)",
                        self.dx_m * 1000, self.dy_m * 1000, self.dz_m * 1000)
        except Exception as e:
            logger.error("Failed to parse %s: %s", self.extrinsics_path, e)

    @staticmethod
    def compose_tray_rotation(base_tcp: List[float], yaw_rad: float) -> List[float]:
        """
        Composes the camera/tray TCP orientation with a planar yaw rotation around tool Z.
        Ensures TCP tool Z remains strictly normal to the table floor [0, 0, -1] without tilting.
        """
        rot_base = R.from_rotvec(base_tcp[3:6])
        rot_yaw = R.from_rotvec([0.0, 0.0, float(yaw_rad)])
        rot_target = (rot_base * rot_yaw).as_rotvec()
        return [float(v) for v in rot_target]

    def estimate(
        self,
        detection: DetectionResult,
        intrinsics: CameraIntrinsics,
        depth_map_m: Optional[np.ndarray] = None,
        camera_base_tcp: Optional[List[float]] = None,
    ) -> Optional[GraspEstimate]:
        """
        Calculates grasp parameters for a single detection result.
        """
        if detection.mask is None:
            # Generate rectangular mask from bbox if no segmentation mask
            x1, y1, x2, y2 = detection.bbox
            mask = np.zeros((intrinsics.height, intrinsics.width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
        else:
            mask = detection.mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take largest continuous contour
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 50:
            return None

        # 1. Minimum Area Rectangle for dimensions
        rect = cv2.minAreaRect(cnt)
        (center_u, center_v), (rect_w, rect_h), _ = rect
        tool_length = max(rect_w, rect_h)
        tool_width = max(1.0, min(rect_w, rect_h))

        # 2. Principal Component Analysis (PCA) for orientation vector
        pts = cnt.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)

        center_pca = mean[0]                 # [cx, cy]
        major_axis = eigenvectors[0]         # [vx, vy] (lengthwise direction)
        minor_axis = eigenvectors[1]         # [nx, ny] (transverse direction)

        # Normalize direction
        norm = np.linalg.norm(major_axis)
        if norm > 1e-6:
            major_axis /= norm
        else:
            major_axis = np.array([1.0, 0.0], dtype=np.float32)

        # In-plane yaw angle in degrees of tool longitudinal axis
        tool_axis_deg = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

        # Gripper fingers close ACROSS the shank (transverse / normal to major axis)
        # Offset by 90 degrees and normalize within [-90, 90] to minimize wrist rotation
        gripper_yaw_deg = tool_axis_deg - 90.0 if tool_axis_deg > 0.0 else tool_axis_deg + 90.0

        # 3. Project points onto major axis to find the two extreme ends
        projections = np.dot(pts - center_pca, major_axis)
        min_p, max_p = np.min(projections), np.max(projections)

        end_a = center_pca + min_p * major_axis
        end_b = center_pca + max_p * major_axis

        # 4. Determine tip vs tail (handle)
        # End with smaller local cross-section/area is considered the working tip
        sub_dist = (max_p - min_p) * 0.15
        pts_near_a = pts[projections < (min_p + sub_dist)]
        pts_near_b = pts[projections > (max_p - sub_dist)]

        spread_a = np.std(np.dot(pts_near_a - end_a, minor_axis)) if len(pts_near_a) > 2 else 1.0
        spread_b = np.std(np.dot(pts_near_b - end_b, minor_axis)) if len(pts_near_b) > 2 else 1.0

        if abs(spread_a - spread_b) > 4.0:
            tip_pt, tail_pt = (end_a, end_b) if spread_a <= spread_b else (end_b, end_a)
        else:
            tip_pt, tail_pt = (end_a, end_b) if end_a[1] <= end_b[1] else (end_b, end_a)

        # 5. Affordance Shank Offset
        # For ringed instruments (scissors, hemostats, clamps): box-lock / shank pivot is ~38% from tip
        # (accounting for the 1:2 blade-to-shank/ring length proportion of surgical instruments)
        # For tweezers/forceps: center grip is ~50%
        # For scalpels/knives: shank / neck grip is ~45%
        label_lower = detection.label.lower()
        if "scissor" in label_lower or "hemostat" in label_lower or "clamp" in label_lower:
            ratio = 0.38
        elif "forceps" in label_lower or "tweezer" in label_lower or "pickup" in label_lower:
            ratio = 0.50
        elif "pen" in label_lower or "brush" in label_lower:
            ratio = 0.45
        elif "knife" in label_lower or "scalpel" in label_lower:
            ratio = 0.50
        else:
            ratio = 0.40

        grasp_pt = tip_pt + ratio * (tail_pt - tip_pt)
        gu, gv = float(grasp_pt[0]), float(grasp_pt[1])

        # 6. Deproject to 3D Camera Coordinates
        depth_m = self._query_depth(depth_map_m, int(gu), int(gv))
        x_cam, y_cam, z_cam = intrinsics.deproject_pixel_to_point(gu, gv, depth_m)

        # 7. Coordinate Transformation: Camera Frame -> UR10e Robot Base Frame
        # When arm is at calibrated tray_pose, camera optical axis points down towards the tray
        # Optical center corresponds to camera_base_tcp [X_base, Y_base]
        # In camera frame: +X is image right, +Y is image down
        # At overhead tray scan pose, camera frame maps to base frame with translation offsets
        ref_tcp = camera_base_tcp if camera_base_tcp is not None else self.camera_base_tcp
        scan_z = float(ref_tcp[2]) if camera_base_tcp is not None else self.scan_height_m
        pick_z = float(self.contact_height_m)

        # 6-DOF Hand-Eye coordinate projection:
        # P_tool = R_tcp_cam * P_cam + t_tcp_cam
        # P_base = R_base_tcp * P_tool + t_base_tcp
        # Ensures full rotation, height, and translation invariance across arbitrary tray poses.
        if self.hand_eye_R is not None and self.hand_eye_t is not None and depth_map_m is not None:
            p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
            p_tcp = self.hand_eye_R @ p_cam + self.hand_eye_t
            r_base_tcp = R.from_rotvec(ref_tcp[3:6]).as_matrix()
            t_base_tcp = np.array(ref_tcp[:3], dtype=np.float64)
            p_base = r_base_tcp @ p_tcp + t_base_tcp
            x_base_target = float(p_base[0])
            y_base_target = float(p_base[1])
        else:
            # Fallback rotation-aware coordinate projection:
            # Maps camera displacement and mount offset through the relative tool orientation
            r_calib = R.from_rotvec(self.calib_base_tcp[3:6])
            r_current = R.from_rotvec(ref_tcp[3:6])
            r_rel = r_current * r_calib.inv()

            v_calib = np.array([x_cam + self.dx_m, y_cam + self.dy_m, 0.0], dtype=np.float64)
            v_base = r_rel.apply(v_calib)

            x_base_target = ref_tcp[0] + float(v_base[0])
            y_base_target = ref_tcp[1] + float(v_base[1])

        # Spatial rotation composition keeping tool Z normal to table floor [0, 0, -1]
        target_rot = self.compose_tray_rotation(ref_tcp, math.radians(gripper_yaw_deg))

        approach_pose = [
            round(x_base_target, 4),
            round(y_base_target, 4),
            round(scan_z, 4),
            round(float(target_rot[0]), 4),
            round(float(target_rot[1]), 4),
            round(float(target_rot[2]), 4),
        ]

        pick_pose = [
            round(x_base_target, 4),
            round(y_base_target, 4),
            round(pick_z, 4),
            round(float(target_rot[0]), 4),
            round(float(target_rot[1]), 4),
            round(float(target_rot[2]), 4),
        ]

        # 8. RG-2 Finger Opening Width (mm)
        # Convert pixel width to estimated mm via pinhole ratio at scan height
        px_to_mm = (depth_m / intrinsics.fx) * 1000.0
        tool_width_mm = tool_width * px_to_mm
        # Pre-open fingers with 18mm clearance (capped between 25mm and 95mm for RG-2)
        gripper_open_mm = float(np.clip(tool_width_mm + 18.0, 25.0, 95.0))

        return GraspEstimate(
            label=detection.label,
            confidence=detection.confidence,
            pixel_grasp=(round(gu, 1), round(gv, 1)),
            pixel_tip=(round(float(tip_pt[0]), 1), round(float(tip_pt[1]), 1)),
            pixel_tail=(round(float(tail_pt[0]), 1), round(float(tail_pt[1]), 1)),
            yaw_deg=round(gripper_yaw_deg, 1),
            tool_length_px=round(float(tool_length), 1),
            tool_width_px=round(float(tool_width), 1),
            camera_coords_m=(round(x_cam, 4), round(y_cam, 4), round(z_cam, 4)),
            robot_approach_pose=approach_pose,
            robot_pick_pose=pick_pose,
            gripper_open_width_mm=round(gripper_open_mm, 1),
        )

    def estimate_all(
        self,
        detections: List[DetectionResult],
        intrinsics: CameraIntrinsics,
        depth_map_m: Optional[np.ndarray] = None,
        camera_base_tcp: Optional[List[float]] = None,
    ) -> List[GraspEstimate]:
        """Calculates grasp parameters for all detections."""
        estimates: List[GraspEstimate] = []
        for det in detections:
            est = self.estimate(det, intrinsics, depth_map_m, camera_base_tcp=camera_base_tcp)
            if est is not None:
                estimates.append(est)
        return estimates

    def _query_depth(
        self,
        depth_map_m: Optional[np.ndarray],
        u: int,
        v: int,
        window_size: int = 5,
    ) -> float:
        """Queries depth with local median filtering, falling back to calibrated tray height."""
        if depth_map_m is None:
            return self.scan_height_m

        h, w = depth_map_m.shape[:2]
        half = window_size // 2
        u1, u2 = max(0, u - half), min(w, u + half + 1)
        v1, v2 = max(0, v - half), min(h, v + half + 1)

        patch = depth_map_m[v1:v2, u1:u2]
        valid_depths = patch[(patch > 0.05) & (patch < 1.5) & (~np.isnan(patch))]

        if len(valid_depths) > 0:
            return float(np.median(valid_depths))

        return self.scan_height_m
