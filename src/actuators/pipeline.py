#!/usr/bin/env python3
"""
Tool Retrieval and Sterile Handover Execution Pipeline.

Implements the multi-stage vision-to-actuation grasp workflow for UR10e & RG-2:
1. Joint space transit to calibrated tray scan pose (moveJ).
2. On-demand snap-and-stop RealSense camera capture and YOLOv8-seg detection.
3. Affordance grasp calculation via PCA with RG-2 linkage arc compensation.
4. Pre-open gripper fingers at standoff height (approach pose).
5. Guarded tactile plunge with F/T contact stopping (|Fz| >= touch_force).
6. Micro-lift above surface and force-controlled clamping with grip confirmation.
7. Retract and transit to handover pose (moveJ).
8. High-frequency wrist F/T pull force monitoring (compliant release).
9. Safe return to home pose.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.context import ActuatorContext
from src.vision.detector import DetectionResult
from src.vision.grasp_estimator import GraspEstimate, GraspEstimator
from src.vision.realsense_stream import RealSenseStream

DEFAULT_EXTRINSICS_PATH = Path(__file__).resolve().parent / "camera_extrinsics.json"
logger = logging.getLogger("actuators.pipeline")

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

# Nominal surgical tool clamped shank / handle thickness in mm
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
    toward the wrist caused by its parallel four-bar linkage arc as opening width increases.
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
    """Captures an on-demand snapshot via RealSenseStream and computes 3D grasp pose."""
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
    """Stabilizes at handover_pose and monitors for surgeon pull force (F_pull >= force_threshold_n)."""
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


class ToolRetrievalPipeline:
    """Modular, stateful execution pipeline for surgical tool pick and handover."""

    @staticmethod
    def execute(
        target: str,
        context: ActuatorContext,
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
        """Executes full tool retrieval and handover cycle on given ActuatorContext."""
        start_time = time.time()
        current_phase = "init"
        sim = context.sim
        rtde_c = context.rtde_c
        rtde_r = context.rtde_r
        gripper = context.gripper
        waypoints = context.waypoints

        logger.info("Executing ToolRetrievalPipeline for '%s' (Mode: %s)...",
                    target, "Simulation" if sim else "Physical Hardware")

        try:
            # 1. Phase 1: Transit to Tray
            current_phase = "transit_home"
            logger.info("[Phase 1] Moving to home_pose (speed=%.2f rad/s)...", speed)
            rtde_c.moveJ(waypoints["home_pose"]["joints_rad"], speed=speed, acceleration=0.25)
            logger.info("[Phase 1] Closing gripper fingers together at home_pose...")
            gripper.close(force_n=20.0)

            current_phase = "transit_tray"
            logger.info("[Phase 1] Moving to tray_pose (speed=%.2f rad/s)...", speed)
            rtde_c.moveJ(waypoints["tray_pose"]["joints_rad"], speed=speed, acceleration=0.25)
            logger.info("[Phase 1] Ensuring gripper fingers touch together at tray_pose before vision snapshot...")
            gripper.close(force_n=20.0)
            time.sleep(0.3)  # Vibration damping dwell

            tray_tcp = None
            if "tray_pose" in waypoints and "tcp_pose" in waypoints["tray_pose"]:
                tray_tcp = [float(v) for v in waypoints["tray_pose"]["tcp_pose"]]

            try:
                actual_tcp = rtde_r.getActualTCPPose()
                if actual_tcp and len(actual_tcp) == 6 and not sim:
                    tray_tcp = [float(v) for v in actual_tcp]
            except Exception as e:
                logger.debug("Could not query live TCP pose via RTDE: %s", e)

            # 2. Phase 2: On-Demand Vision Capture ("Snap-and-Stop")
            current_phase = "vision_grasp_estimation"
            eff_source = source
            if sim and (not source or source.lower() in ("realsense", "auto")):
                eff_source = "sim"

            logger.info("[Phase 2] Capturing snapshot & calculating grasp pose for '%s'...", target)
            estimate = acquire_and_estimate_grasp(
                source=eff_source,
                target=target,
                conf=conf,
                model=model,
                sim=sim,
                extrinsics_path=extrinsics_path or DEFAULT_EXTRINSICS_PATH,
                fps=fps,
                exposure=exposure,
                gain=gain,
                camera_base_tcp=tray_tcp,
            )

            # Workspace surface reference resolution
            table_floor_z = -0.0025
            if "table_contact_pose" in waypoints and "tcp_pose" in waypoints["table_contact_pose"]:
                table_floor_z = float(waypoints["table_contact_pose"]["tcp_pose"][2])

            surface_calib_z = 0.0151
            if surface == "padded":
                if "padded_contact_pose" in waypoints and "tcp_pose" in waypoints["padded_contact_pose"]:
                    surface_calib_z = float(waypoints["padded_contact_pose"]["tcp_pose"][2])
                elif "tray_contact_pose" in waypoints and "tcp_pose" in waypoints["tray_contact_pose"]:
                    surface_calib_z = float(waypoints["tray_contact_pose"]["tcp_pose"][2])
                surface_calib_z += (pad_offset_mm / 1000.0)
            else:
                surface_calib_z = table_floor_z

            fingertip_offset_m = compute_rg2_fingertip_offset_m(estimate.gripper_open_width_mm)
            nominal_contact_z = surface_calib_z - fingertip_offset_m

            # Hard floor clamp
            surface_calib_z = max(surface_calib_z, table_floor_z)
            estimate.robot_pick_pose[2] = surface_calib_z

            if abs(offset_x_mm) > 1e-4 or abs(offset_y_mm) > 1e-4:
                dx_m = offset_x_mm / 1000.0
                dy_m = offset_y_mm / 1000.0
                estimate.robot_approach_pose[0] += dx_m
                estimate.robot_approach_pose[1] += dy_m
                estimate.robot_pick_pose[0] += dx_m
                estimate.robot_pick_pose[1] += dy_m

            # 3. Phase 3: Guarded Grasp Sequence
            current_phase = "pre_open_gripper"
            logger.info("[Phase 3] Pre-opening RG-2 gripper to %.1f mm...", estimate.gripper_open_width_mm)
            gripper.move(estimate.gripper_open_width_mm, force_n=20.0)

            current_phase = "approach"
            logger.info("[Phase 3] Moving linearly to approach_pose (Z=%.4f m)...", estimate.robot_approach_pose[2])
            ok_approach = rtde_c.moveL(estimate.robot_approach_pose, speed=min(speed, 0.08), acceleration=0.15)
            if not ok_approach:
                return MissionResult(
                    success=False,
                    target=target,
                    phase="approach_failed",
                    message="Linear motion to approach pose failed: target TCP unreachable",
                    grasp_estimate=estimate,
                    elapsed_time_s=time.time() - start_time,
                )

            current_phase = "fast_hover"
            hover_clearance_m = 0.020
            hover_pose = list(estimate.robot_approach_pose)
            hover_pose[0] = estimate.robot_pick_pose[0]
            hover_pose[1] = estimate.robot_pick_pose[1]
            hover_pose[2] = nominal_contact_z + hover_clearance_m

            ok_hover = rtde_c.moveL(hover_pose, speed=min(speed, 0.06), acceleration=0.15)
            if not ok_hover:
                return MissionResult(
                    success=False,
                    target=target,
                    phase="hover_failed",
                    message="Rapid descent to hover pose failed: target TCP unreachable",
                    grasp_estimate=estimate,
                    elapsed_time_s=time.time() - start_time,
                )

            current_phase = "quiescent_tare"
            time.sleep(0.15)
            try:
                rtde_c.zeroFtSensor()
                time.sleep(0.05)
            except Exception as e:
                logger.debug("zeroFtSensor warning: %s", e)

            current_phase = "guarded_touch"
            target_touch = list(hover_pose)
            target_touch[2] = table_floor_z
            creep_speed = max(0.002, float(touch_speed))
            max_plunge_stroke_m = max(0.015, (hover_pose[2] - table_floor_z) + 0.005)

            ok_plunge = rtde_c.moveL(target_touch, speed=creep_speed, acceleration=0.15, asynchronous=True)
            if not ok_plunge:
                return MissionResult(
                    success=False,
                    target=target,
                    phase="plunge_failed",
                    message="Guarded plunge moveL initiation failed: target TCP unreachable",
                    grasp_estimate=estimate,
                    elapsed_time_s=time.time() - start_time,
                )

            contact_detected = False
            plunge_t0 = time.time()
            timeout_plunge = max(6.0, (max_plunge_stroke_m / creep_speed) * 1.8)
            consecutive_force_hits = 0

            while time.time() - plunge_t0 < timeout_plunge:
                curr_tcp = rtde_r.getActualTCPPose()
                curr_z = curr_tcp[2] if curr_tcp and len(curr_tcp) >= 3 else hover_pose[2]

                forces = rtde_r.getActualTCPForce()
                if forces and len(forces) >= 3:
                    fz_contact = abs(float(forces[2]))
                    if fz_contact >= touch_force:
                        consecutive_force_hits += 1
                        if consecutive_force_hits >= 2:
                            rtde_c.stopL(a=2.0)
                            contact_detected = True
                            break
                    else:
                        consecutive_force_hits = 0

                if curr_tcp and curr_tcp[2] <= table_floor_z + 0.0005:
                    rtde_c.stopL(a=2.0)
                    break

                if (hover_pose[2] - curr_z) >= max_plunge_stroke_m:
                    rtde_c.stopL(a=2.0)
                    break

                time.sleep(0.004)

            if not contact_detected:
                try:
                    rtde_c.stopL(a=2.0)
                except Exception:
                    pass
            time.sleep(0.05)

            contact_tcp = rtde_r.getActualTCPPose()
            if not contact_tcp or len(contact_tcp) < 6:
                contact_tcp = list(hover_pose)
                contact_tcp[2] = nominal_contact_z

            contact_z = float(contact_tcp[2])

            # Micro-lift
            current_phase = "micro_lift"
            pad_clearance_m = float(lift_mm) / 1000.0
            lift_pose = list(contact_tcp)
            lift_pose[2] = contact_z + pad_clearance_m
            rtde_c.moveL(lift_pose, speed=0.03, acceleration=0.2)
            time.sleep(0.1)

            # Clamp
            current_phase = "grasp"
            eff_grasp_force = grasp_force
            if eff_grasp_force is None:
                label_lower = target.lower()
                eff_grasp_force = 20.0
                for k_tool, f_val in DEFAULT_TOOL_GRASP_FORCES.items():
                    if k_tool in label_lower:
                        eff_grasp_force = f_val
                        break

            if sim and hasattr(gripper, "_sim") and eff_source != "sim":
                gripper._sim._simulated_contact_override = True

            close_state = gripper.close(force_n=eff_grasp_force)
            grip_detected = bool(close_state.get("grip_detected", False))
            width_mm = float(close_state.get("width_mm", 0.0))

            if not grip_detected or width_mm < 1.0:
                try:
                    gripper.open(force_n=20.0)
                except Exception:
                    pass
                rtde_c.moveL(estimate.robot_approach_pose, speed=0.05, acceleration=0.1)
                return MissionResult(
                    success=False,
                    target=target,
                    phase="grasp_failed",
                    message=f"Grasp verification failed: grip not detected on '{target}' (width={width_mm:.1f} mm)",
                    grasp_estimate=estimate,
                    grip_detected=False,
                    handover_released=False,
                    elapsed_time_s=time.time() - start_time,
                )

            current_phase = "retract"
            rtde_c.moveL(estimate.robot_approach_pose, speed=0.05, acceleration=0.1)

            # 4. Phase 4: Handover
            current_phase = "traverse_handover"
            rtde_c.moveJ(waypoints["handover_pose"]["joints_rad"], speed=speed, acceleration=0.25)

            current_phase = "handover_release"
            handover_released = monitor_handover_release(
                rtde_c=rtde_c,
                rtde_r=rtde_r,
                gripper=gripper,
                force_threshold_n=force_threshold,
                timeout_s=handover_timeout,
                sim_auto_pull=sim_auto_pull,
                sim_pull_delay_s=sim_pull_delay,
            )

            if not handover_released:
                return MissionResult(
                    success=False,
                    target=target,
                    phase="handover_timeout",
                    message=f"Surgeon pull not detected within {handover_timeout:.1f}s timeout.",
                    grasp_estimate=estimate,
                    grip_detected=grip_detected,
                    handover_released=False,
                    elapsed_time_s=time.time() - start_time,
                )

            current_phase = "return_home"
            rtde_c.moveJ(waypoints["home_pose"]["joints_rad"], speed=speed, acceleration=0.25)
            gripper.close(force_n=20.0)

            elapsed = time.time() - start_time
            return MissionResult(
                success=True,
                target=target,
                phase="completed",
                message="Mission executed successfully",
                grasp_estimate=estimate,
                grip_detected=grip_detected,
                handover_released=handover_released,
                elapsed_time_s=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Mission failed in phase '%s': %s", current_phase, e, exc_info=True)
            return MissionResult(
                success=False,
                target=target,
                phase=current_phase,
                message=str(e),
                elapsed_time_s=elapsed,
            )
