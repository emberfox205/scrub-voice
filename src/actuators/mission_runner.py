#!/usr/bin/env python3
"""
Automated Vision-to-Actuation Grasp and Handover Pipeline for UR10e & OnRobot RG-2.

Orchestrates the complete surgical retrieval workflow:
1. Transit from home_pose to tray_pose via joint interpolation (moveJ).
2. On-demand snap-and-stop vision capture via RealSense RGB-D (stops camera to save Pi CPU).
3. YOLOv8-seg tool detection and PCA-based affordance grasp estimation.
4. Pre-opening of RG-2 gripper fingers at standoff height (approach_pose).
5. Guarded linear plunge (moveL) to pick_pose (capped at 3 cm/s).
6. Clamping of object with 25 N force and grip confirmation.
7. Retraction straight up (moveL) and transit to handover_pose via moveJ.
8. Compliant handover release via high-frequency wrist F/T sensor monitoring (F_pull >= 10 N).
9. Safe dwell (0.5s) and return to home_pose.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dynamic root resolution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.context import ActuatorContext
from src.actuators.pipeline import (
    DEFAULT_TOOL_GRASP_FORCES,
    DEFAULT_TOOL_NOMINAL_SHANK_WIDTH_MM,
    MissionResult,
    ToolRetrievalPipeline,
    acquire_and_estimate_grasp,
    compute_rg2_fingertip_offset_m,
    monitor_handover_release,
)
from src.vision.detector import DetectionResult
from src.vision.grasp_estimator import GraspEstimate, GraspEstimator
from src.vision.realsense_stream import RealSenseStream

DEFAULT_WAYPOINTS_PATH = Path(__file__).resolve().parent / "waypoints.json"
DEFAULT_EXTRINSICS_PATH = Path(__file__).resolve().parent / "camera_extrinsics.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MissionRunner] %(message)s",
)
logger = logging.getLogger("actuators.mission_runner")

# Recommended surgical tool clamping force limits in Newtons (OnRobot RG-2: 3-40 N)
DEFAULT_TOOL_GRASP_FORCES: Dict[str, float] = {
    "scalpel": 12.0,
    "knife": 12.0,
    "forceps": 15.0,
    "tweezer": 15.0,
    "pickup": 15.0,
    "scissors": 20.0,
    "hemostat": 20.0,
    "clamp": 20.0,
}

# Nominal surgical tool clamped shank / handle thickness in mm (for micro-lift re-extension calculation)
DEFAULT_TOOL_NOMINAL_SHANK_WIDTH_MM: Dict[str, float] = {
    "scalpel": 5.0,
    "knife": 5.0,
    "forceps": 8.0,
    "tweezer": 8.0,
    "pickup": 8.0,
    "scissors": 14.0,
    "hemostat": 14.0,
    "clamp": 14.0,
}


def compute_rg2_fingertip_offset_m(width_mm: float) -> float:
    """
    Computes the vertical retraction offset (in meters) of the OnRobot RG-2 fingertips
    toward the wrist caused by its parallel four-bar linkage arc as the opening width increases.

    At width=0 mm (closed), fingers are at maximum downward extension (offset = 0.0 m).
    As width increases, fingers travel along a circular arc, shortening tool Z.
    Fitted polynomial: dz(w) = 0.002652 * w^2 + 0.027301 * w (in mm).
    """
    w = max(0.0, float(width_mm))
    dz_mm = 0.002652 * (w ** 2) + 0.027301 * w
    return dz_mm / 1000.0


@dataclass
class MissionResult:
    """Summary of the executed retrieval and handover mission."""
    success: bool
    target: str
    phase: str
    message: str
    grasp_estimate: Optional[GraspEstimate] = None
    grip_detected: bool = False
    handover_released: bool = False
    elapsed_time_s: float = 0.0


def load_waypoints(waypoints_path: Optional[Path] = None, sim: bool = False) -> Dict[str, Any]:
    """Loads and validates calibrated waypoint definitions from JSON."""
    path = Path(waypoints_path or DEFAULT_WAYPOINTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Waypoints configuration not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["home_pose", "tray_pose", "handover_pose"]
    for k in required_keys:
        if k not in data or "joints_rad" not in data[k]:
            raise ValueError(f"Malformed waypoints file {path}: missing required waypoint '{k}'")

    return data


def acquire_and_estimate_grasp(
    source: str = "realsense",
    target: str = "scissors",
    conf: float = 0.15,
    model: Optional[str] = None,
    sim: bool = False,
    extrinsics_path: Optional[Path] = None,
    fps: int = 15,
    exposure: int = 66000,
    gain: int = 32,
    camera_base_tcp: Optional[List[float]] = None,
) -> GraspEstimate:
    """
    Captures an on-demand snapshot via RealSenseStream and computes the 3D grasp pose.
    """
    eff_source = source
    if sim and (not source or source.lower() in ("realsense", "auto")):
        eff_source = "sim"

    logger.info("Acquiring on-demand vision snapshot (source: %s)...", eff_source)
    stream = RealSenseStream(
        source=eff_source,
        fps=fps,
        exposure=exposure,
        gain=gain,
    )
    color_img, depth_img_m, intrinsics = stream.capture_snapshot()
    logger.info("Snapshot acquired (%dx%d, depth: %s). Stream stopped.",
                intrinsics.width, intrinsics.height, depth_img_m is not None)

    estimator = GraspEstimator(
        extrinsics_path=extrinsics_path,
        camera_base_tcp=camera_base_tcp,
    )

    detections: List[DetectionResult] = []
    try:
        from src.vision.detector import ToolDetector
        detector = ToolDetector(model_name=model, conf_threshold=conf, allow_fallback=True)
        detections = detector.detect(color_img, target_classes=[target])
    except Exception as e:
        logger.warning("ToolDetector inference error: %s", e)

    if not detections:
        raise RuntimeError(f"Target object '{target}' not detected in camera snapshot.")

    best_det = detections[0]
    logger.info("Identified '%s' (conf: %.2f, bbox: %s)", best_det.label, best_det.confidence, best_det.bbox)

    estimate = estimator.estimate(
        best_det,
        intrinsics,
        depth_img_m,
        camera_base_tcp=camera_base_tcp,
    )
    if estimate is None:
        raise RuntimeError(f"Failed to calculate valid grasp pose for '{target}'")

    logger.info("Grasp Estimate: Approach=%s | Pick=%s | Gripper θ=%.1f° | RG-2 Opening=%.1f mm",
                estimate.robot_approach_pose[:3],
                estimate.robot_pick_pose[:3],
                estimate.yaw_deg,
                estimate.gripper_open_width_mm)
    return estimate


def monitor_handover_release(
    rtde_c: Any,
    rtde_r: Any,
    gripper: Any,
    force_threshold_n: float = 10.0,
    timeout_s: float = 60.0,
    poll_rate_hz: float = 500.0,
    sim_auto_pull: bool = False,
    sim_pull_delay_s: float = 2.0,
) -> bool:
    """
    Stabilizes at handover_pose, tares the wrist F/T sensor baseline, and monitors
    for surgeon pull force (F_pull >= force_threshold_n) at high frequency.
    """
    logger.info("Stabilizing arm at handover pose (0.2s)...")
    time.sleep(0.2)

    logger.info("Zeroing wrist F/T sensor via rtde_c.zeroFtSensor()...")
    rtde_c.zeroFtSensor()

    logger.info("Monitoring TCP pull force (threshold: %.1f N, timeout: %.1fs)...",
                force_threshold_n, timeout_s)

    start_t = time.time()
    poll_interval = 1.0 / max(10.0, poll_rate_hz)
    pull_injected = False

    while time.time() - start_t < timeout_s:
        # Automated simulated surgeon pull
        if sim_auto_pull and not pull_injected and (time.time() - start_t >= sim_pull_delay_s):
            logger.info("Triggering simulated surgeon pull (%.1f N >= %.1f N)...",
                        force_threshold_n + 2.5, force_threshold_n)
            pull_injected = True
            is_sim_shim = type(rtde_r).__name__ == "SimRTDEReceiveInterface"
            if is_sim_shim and hasattr(rtde_r, "_sim") and hasattr(rtde_r._sim, "_simulated_tcp_force"):
                rtde_r._sim._simulated_tcp_force = [0.0, 0.0, force_threshold_n + 2.5, 0.0, 0.0, 0.0]
            elif is_sim_shim and hasattr(rtde_r, "set_simulated_tcp_force"):
                rtde_r.set_simulated_tcp_force([0.0, 0.0, force_threshold_n + 2.5, 0.0, 0.0, 0.0])
            else:
                gripper.open(force_n=20.0)
                time.sleep(0.5)
                return True

        # Query live 6-axis F/T sensor from UR10e wrist flange
        forces = rtde_r.getActualTCPForce()
        if forces and len(forces) >= 3:
            fx, fy, fz = float(forces[0]), float(forces[1]), float(forces[2])
            f_pull = math.sqrt(fx * fx + fy * fy + fz * fz)

            if f_pull >= force_threshold_n:
                logger.info("Surgeon pull detected! Force: %.2f N (Fx=%.2f, Fy=%.2f, Fz=%.2f)",
                            f_pull, fx, fy, fz)
                gripper.open(force_n=20.0)
                time.sleep(0.5)
                return True

        time.sleep(poll_interval)

    logger.warning("Handover safety timeout reached (%.1fs) without surgeon pull.", timeout_s)
    return False


def run_mission(
    target: str = "scissors",
    sim: bool = False,
    headless: Optional[bool] = None,
    robot_ip: Optional[str] = None,
    gripper_ip: Optional[str] = None,
    source: str = "realsense",
    speed: float = 0.2,
    force_threshold: float = 10.0,
    handover_timeout: float = 60.0,
    sim_auto_pull: bool = False,
    sim_pull_delay: float = 2.0,
    conf: float = 0.15,
    model: Optional[str] = None,
    fps: int = 15,
    exposure: int = 66000,
    gain: int = 32,
    waypoints_path: Optional[Path] = None,
    extrinsics_path: Optional[Path] = None,
    touch_force: float = 2.0,
    touch_speed: float = 0.010,
    lift_mm: float = 1.2,
    grasp_force: Optional[float] = None,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
    surface: str = "padded",
    pad_offset_mm: float = 0.0,
) -> MissionResult:
    """
    Executes the end-to-end vision-to-actuation grasp and handover pipeline.
    """
    context = ActuatorContext(
        robot_ip=robot_ip,
        gripper_ip=gripper_ip,
        sim=sim,
        headless=headless,
        waypoints_path=waypoints_path,
    )
    return ToolRetrievalPipeline.execute(
        target=target,
        context=context,
        source=source,
        speed=speed,
        force_threshold=force_threshold,
        handover_timeout=handover_timeout,
        sim_auto_pull=sim_auto_pull,
        sim_pull_delay=sim_pull_delay,
        conf=conf,
        model=model,
        fps=fps,
        exposure=exposure,
        gain=gain,
        extrinsics_path=extrinsics_path,
        touch_force=touch_force,
        touch_speed=touch_speed,
        lift_mm=lift_mm,
        grasp_force=grasp_force,
        offset_x_mm=offset_x_mm,
        offset_y_mm=offset_y_mm,
        surface=surface,
        pad_offset_mm=pad_offset_mm,
    )


def main():
    parser = argparse.ArgumentParser(
        description="UR10e & OnRobot RG-2 Vision-to-Actuation Grasp and Handover Mission"
    )
    parser.add_argument("--target", type=str, default="scissors", help="Target tool label (default: scissors)")
    parser.add_argument("--sim", action="store_true", help="Run against MuJoCo physics simulation")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode (no 3D GUI)")
    parser.add_argument("--robot-ip", type=str, default=None, help="UR10e robot IP address (default: 192.168.1.105)")
    parser.add_argument("--gripper-ip", type=str, default=None, help="OnRobot Compute Box IP (default: 192.168.1.101)")
    parser.add_argument(
        "--source", type=str, default="realsense",
        help="Camera source ('realsense', 'auto', or image file path)",
    )
    parser.add_argument("--speed", type=float, default=0.2, help="Joint motion speed multiplier (default: 0.2 rad/s)")
    parser.add_argument(
        "--surface", type=str, choices=["padded", "table"], default="padded",
        help="Target workspace surface reference (default: 'padded')",
    )
    parser.add_argument(
        "--pad-offset-mm", type=float, default=0.0,
        help="Fine-tuning vertical offset in mm for varying foam pad stacks (default: 0.0 mm)",
    )
    parser.add_argument(
        "--force-threshold", type=float, default=5.0,
        help="Surgeon pull trigger force threshold in Newtons (default: 5.0 N)",
    )
    parser.add_argument(
        "--handover-timeout", type=float, default=60.0,
        help="Max seconds to wait for surgeon pull (default: 60.0s)",
    )
    parser.add_argument("--sim-auto-pull", action="store_true", help="Trigger automated surgeon pull in simulation")
    parser.add_argument(
        "--sim-pull-delay", type=float, default=2.0,
        help="Seconds before triggering simulated pull (default: 2.0s)",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold (default: 0.15)")
    parser.add_argument("--model", type=str, default=None, help="Optional model checkpoint path")
    parser.add_argument("--fps", type=int, default=15, help="Camera stream framerate (default: 15)")
    parser.add_argument("--exposure", type=int, default=66000, help="Manual exposure in microseconds (default: 66000)")
    parser.add_argument("--gain", type=int, default=32, help="Sensor hardware gain (default: 32)")
    parser.add_argument(
        "--touch-force", type=float, default=2.0,
        help="Contact force threshold in N to detect surface touch during plunge (default: 2.0 N)",
    )
    parser.add_argument(
        "--touch-speed", type=float, default=0.010,
        help="Creep speed in m/s for second stage guarded descent to touch (default: 0.010 m/s = 1.0 cm/s)",
    )
    parser.add_argument(
        "--lift-mm", type=float, default=1.2,
        help="Micro-lift distance in mm above contact before grasping (default: 1.2 mm)",
    )
    parser.add_argument(
        "--grasp-force", type=float, default=None,
        help="Gripper clamping force in N for tool grasp (default: tool-specific, 12-20 N)",
    )
    parser.add_argument(
        "--offset-x", type=float, default=0.0,
        help="Lateral X compensation offset in mm (shifts tool grasp left/right, default: 0.0)",
    )
    parser.add_argument(
        "--offset-y", type=float, default=0.0,
        help="Lateral Y compensation offset in mm (shifts tool grasp forward/backward, default: 0.0)",
    )

    args = parser.parse_args()

    result = run_mission(
        target=args.target,
        sim=args.sim,
        headless=args.headless,
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        source=args.source,
        speed=args.speed,
        force_threshold=args.force_threshold,
        handover_timeout=args.handover_timeout,
        sim_auto_pull=args.sim_auto_pull,
        sim_pull_delay=args.sim_pull_delay,
        conf=args.conf,
        model=args.model,
        fps=args.fps,
        exposure=args.exposure,
        gain=args.gain,
        touch_force=args.touch_force,
        touch_speed=args.touch_speed,
        lift_mm=args.lift_mm,
        grasp_force=args.grasp_force,
        offset_x_mm=args.offset_x,
        offset_y_mm=args.offset_y,
        surface=args.surface,
        pad_offset_mm=args.pad_offset_mm,
    )

    if result.success:
        print("\n=======================================================")
        print(" MISSION COMPLETED SUCCESSFULLY")
        print("=======================================================")
        print(f" Target Object    : {result.target}")
        print(f" Grip Confirmed   : {result.grip_detected}")
        print(f" Handover Release : {result.handover_released}")
        print(f" Total Duration   : {result.elapsed_time_s:.2f} s")
        print("=======================================================\n")
    else:
        print("\n=======================================================")
        print(" MISSION EXECUTION FAILED")
        print("=======================================================")
        print(f" Phase Failed     : {result.phase}")
        print(f" Error            : {result.message}")
        print(f" Elapsed Time     : {result.elapsed_time_s:.2f} s")
        print("=======================================================\n")

    if args.sim and not args.headless:
        print("[MuJoCo] Simulation complete. Interactive 3D viewer is active.")
        print("[MuJoCo] Inspect the scene with mouse controls, then close the viewer window or press Ctrl+C to exit.\n")
        try:
            from src.actuators.hal.sim_shims import get_shared_sim
            sim_ctrl = get_shared_sim()
            if sim_ctrl and sim_ctrl.viewer:
                while sim_ctrl.viewer.is_running():
                    time.sleep(0.1)
                sim_ctrl.close_viewer()
        except (KeyboardInterrupt, BaseException):
            try:
                from src.actuators.hal.sim_shims import reset_shared_sim
                reset_shared_sim()
            except BaseException:
                pass

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
