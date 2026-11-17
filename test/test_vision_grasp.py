#!/usr/bin/env python3
"""
Interactive Lab Testing & Verification Runner for Vision Recognition and Grasping.

Enables live testing in the lab with Intel RealSense or static photos (data/vision/desk.jpg).
Draws visual overlays (bounding boxes, masks, major axes, grasp points, UR10e coordinates)
and supports automated unit testing & latency benchmarking.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Dynamic path resolution according to AGENTS.md Rule 4
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEST_IMAGE = REPO_ROOT / "data" / "vision" / "desk.jpg"

from src.vision.detector import DetectionResult, ToolDetector, HAS_ULTRALYTICS
from src.vision.grasp_estimator import GraspEstimate, GraspEstimator
from src.vision.realsense_stream import RealSenseStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VisionTest] %(message)s",
)
logger = logging.getLogger("test_vision_grasp")


def draw_hud_and_grasps(
    image: np.ndarray,
    detections: List[DetectionResult],
    grasps: List[GraspEstimate],
    fps: float,
    backend_name: str,
    model_name: str = "",
    exclusion_polygon: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Renders visual overlays, crosshairs, and kinematic HUD on the display frame."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # 1. Overlay Segmentation Masks with Alpha Blend
    mask_overlay = np.zeros_like(annotated)
    palette = [
        (40, 200, 40),   # Green
        (220, 100, 30),  # Blue
        (40, 160, 240),  # Orange
        (200, 40, 200),  # Magenta
    ]

    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        if det.mask is not None:
            mask_overlay[det.mask > 0] = color
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, tag, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Blend masks with 35% opacity
    cv2.addWeighted(mask_overlay, 0.35, annotated, 0.65, 0, annotated)

    # 2. Draw Gripper Self-Occlusion Exclusion Zone
    if exclusion_polygon:
        poly_px = np.array(
            [[int(x * w), int(y * h)] for x, y in exclusion_polygon],
            dtype=np.int32,
        )
        zone_overlay = annotated.copy()
        cv2.fillPoly(zone_overlay, [poly_px], (20, 20, 20))
        cv2.addWeighted(zone_overlay, 0.40, annotated, 0.60, 0, annotated)
        cv2.polylines(annotated, [poly_px], True, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, "Gripper Blindspot (Masked)", (int(w * 0.58), int(h * 0.96)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 255), 1, cv2.LINE_AA)

    # 3. Draw Grasp Axes & Crosshairs
    for idx, g in enumerate(grasps):
        color = palette[idx % len(palette)]
        gx, gy = int(g.pixel_grasp[0]), int(g.pixel_grasp[1])
        tx, ty = int(g.pixel_tip[0]), int(g.pixel_tip[1])
        bx, by = int(g.pixel_tail[0]), int(g.pixel_tail[1])

        # Centerline (major axis)
        cv2.line(annotated, (tx, ty), (bx, by), (255, 255, 0), 2)
        # Tip marker (cyan) and Base marker (yellow)
        cv2.circle(annotated, (tx, ty), 5, (255, 255, 0), -1)
        cv2.circle(annotated, (bx, by), 5, (0, 200, 255), -1)

        # Grasp Target Crosshair (Red/White target)
        cv2.circle(annotated, (gx, gy), 10, (0, 0, 255), 2)
        cv2.circle(annotated, (gx, gy), 3, (0, 0, 255), -1)
        cv2.line(annotated, (gx - 14, gy), (gx + 14, gy), (0, 0, 255), 2)
        cv2.line(annotated, (gx, gy - 14), (gx, gy + 14), (0, 0, 255), 2)

        # Grasp annotation tag
        label_text = f"Grip θ:{g.yaw_deg:.0f}° RG2:{g.gripper_open_width_mm:.0f}mm"
        cv2.putText(annotated, label_text, (gx + 15, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. Heads-Up Display (HUD) Banner in top-left
    hud_bg_w, hud_bg_h = min(w, 480), 125
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (hud_bg_w, hud_bg_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    model_tag = f"Model: {model_name}" if model_name else "Model: YOLOv8-seg"
    cv2.putText(annotated, f"Source: {backend_name.upper()} | {model_tag} | FPS: {fps:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

    if grasps:
        top_g = grasps[0]
        tx, ty, tz = top_g.robot_pick_pose[0], top_g.robot_pick_pose[1], top_g.robot_pick_pose[2]
        cv2.putText(annotated, f"Target Tool : {top_g.label.upper()} (conf: {top_g.confidence:.2f})",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"UR10e Target: X={tx:+.3f}m, Y={ty:+.3f}m, Z={tz:+.3f}m",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"Wrist Yaw   : {top_g.yaw_deg:+.1f} deg | RG-2 Pre-open: {top_g.gripper_open_width_mm:.1f} mm",
            (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1, cv2.LINE_AA,
        )
        cv2.putText(annotated, f"Camera Rel  : X={top_g.camera_coords_m[0]:+.3f}m, Y={top_g.camera_coords_m[1]:+.3f}m",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "Scanning tray... (No target tools detected)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    return annotated


def run_benchmark(
    stream: RealSenseStream, detector: ToolDetector, estimator: GraspEstimator, runs: int = 20
) -> None:
    """Profiles latency and resource efficiency over N frames."""
    print("=" * 65)
    print(f" VISION & GRASP BENCHMARK ({runs} iterations)")
    print("=" * 65)

    latencies_detect = []
    latencies_grasp = []
    latencies_total = []

    for _ in range(runs):
        t0 = time.perf_counter()
        img, depth_m, intrinsics = stream.get_frame()
        t1 = time.perf_counter()

        dets = detector.detect(img)
        t2 = time.perf_counter()

        estimator.estimate_all(dets, intrinsics, depth_m)
        t3 = time.perf_counter()

        latencies_detect.append((t2 - t1) * 1000.0)
        latencies_grasp.append((t3 - t2) * 1000.0)
        latencies_total.append((t3 - t0) * 1000.0)

    mean_cap = np.mean(latencies_total) - np.mean(latencies_detect) - np.mean(latencies_grasp)
    print(f" Backend              : {stream.active_backend}")
    print(f" Model Checkpoint     : {detector.model_name}")
    print(f" Mean Capture Latency : {mean_cap:.2f} ms")
    print(
        f" Mean Detect Latency  : {np.mean(latencies_detect):.2f} ms "
        f"(Min: {np.min(latencies_detect):.2f} ms, Max: {np.max(latencies_detect):.2f} ms)"
    )
    print(f" Mean Grasp Latency   : {np.mean(latencies_grasp):.2f} ms")
    print(f" Mean Total Pipeline  : {np.mean(latencies_total):.2f} ms -> {1000.0 / np.mean(latencies_total):.1f} FPS")
    print("=" * 65)


def run_interactive(
    source: str = "auto",
    model_name: Optional[str] = None,
    conf: float = 0.15,
    save_snapshot: Optional[str] = None,
    no_gui: bool = False,
    benchmark_runs: Optional[int] = None,
    target_classes: Optional[List[str]] = None,
    enable_mask: bool = True,
    fps: int = 15,
    exposure: int = 66000,
    gain: int = 32,
) -> None:
    """Main interactive runner loop."""
    print("Initializing RealSense / Image Streamer...")
    stream = RealSenseStream(
        source=source,
        fps=fps,
        exposure=exposure,
        gain=gain,
    )
    backend = stream.start()
    print(f"✓ Source Backend active: {backend}")

    # If simulation backend, automatically transit arm to tray scan pose
    if backend == "sim" and hasattr(stream, "_sim") and stream._sim is not None:
        print("[VisionTest] Moving simulation arm to tray_pose for scanning...")
        stream._sim.moveJ(stream._sim.tray_pose, duration=0.4)
        print("[VisionTest] Closing gripper fingers together to minimize camera exclusion mask...")
        stream._sim.close_gripper(force_n=20.0, duration=0.2)

    print(f"Initializing Tool Detector (model: {model_name or 'default'}, mask: {enable_mask})...")
    detector = ToolDetector(model_name=model_name, conf_threshold=conf, enable_mask=enable_mask)
    print(f"✓ Detector active: {detector.model_name}")

    waypoints_file = REPO_ROOT / "src" / "actuators" / "waypoints.json"
    tray_tcp = None
    if waypoints_file.exists():
        try:
            with open(waypoints_file, "r", encoding="utf-8") as f:
                wp_data = json.load(f)
                if "tray_pose" in wp_data and "tcp_pose" in wp_data["tray_pose"]:
                    tray_tcp = [float(v) for v in wp_data["tray_pose"]["tcp_pose"]]
        except Exception:
            pass

    print("Initializing Grasp Estimator (UR10e + RG-2)...")
    estimator = GraspEstimator(camera_base_tcp=tray_tcp)
    print("✓ Grasp Estimator loaded.\n")

    if benchmark_runs and benchmark_runs > 0:
        run_benchmark(stream, detector, estimator, runs=benchmark_runs)
        stream.stop()
        return

    print("=" * 65)
    print(f" VISION RECOGNITION & GRASP POSE ESTIMATOR [{backend.upper()}]")
    print(f" Active Model : {detector.model_name}")
    print(" Running live stream... Press Ctrl+C in terminal to stop.")
    print("=" * 65)

    fps = 0.0
    frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()
            img, depth_m, intrinsics = stream.get_frame()

            # Run detection & grasping
            detections = detector.detect(img, target_classes=target_classes)
            grasps = estimator.estimate_all(detections, intrinsics, depth_m)

            # Measure performance
            dt = time.perf_counter() - t0
            current_fps = 1.0 / max(dt, 1e-4)
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            # Render visual overlays
            annotated = draw_hud_and_grasps(
                img,
                detections,
                grasps,
                fps,
                backend,
                detector.model_name,
                exclusion_polygon=detector.exclusion_polygon,
            )

            # Console reporting
            frame_count += 1
            if frame_count % 30 == 0 or backend == "file":
                if grasps:
                    for idx, g in enumerate(grasps):
                        print(f"[{g.label.upper()}] Conf: {g.confidence:.2f} | "
                              f"Grasp θ: {g.yaw_deg:+.1f}° | UR10e Approach: {g.robot_approach_pose[:3]} | "
                              f"UR10e Pick: {g.robot_pick_pose[:3]} | RG-2: {g.gripper_open_width_mm:.1f}mm")
                elif backend == "file":
                    print(f"[INFO] No valid tool grasps calculated. Raw detections found: {len(detections)}")
                    if detections:
                        for d in detections:
                            print(f"  → Found candidate: '{d.label}' (conf: {d.confidence:.2f})")
                    else:
                        print("  → Tip: Lower threshold with '--conf 0.12' or supply a model via '--model <ckpt>'.")

            if not no_gui:
                cv2.imshow("UR10e Vision & Grasp Estimator", annotated)
                wait_ms = 0 if backend == "file" else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("\nExiting display...")
                    break
                elif key in (ord("s"), ord("S")):
                    out_path = f"vision_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(out_path, annotated)
                    print(f"✓ Saved snapshot to: {out_path}")
                elif backend == "file":
                    break
                # Check if window was closed by the user
                try:
                    if cv2.getWindowProperty("UR10e Vision & Grasp Estimator", cv2.WND_PROP_VISIBLE) < 1:
                        print("\nWindow closed.")
                        break
                except Exception:
                    print("\nWindow closed.")
                    break
            else:
                if backend == "file" or (frame_count >= 10 and not save_snapshot):
                    break

            if save_snapshot:
                cv2.imwrite(save_snapshot, annotated)
                print(f"✓ Saved snapshot to {save_snapshot}")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        stream.stop()
        if not no_gui:
            cv2.destroyAllWindows()


# ===========================================================================
# Automated Unit Test Suite (python3 -m unittest test/test_vision_grasp.py)
# ===========================================================================

class TestVisionGraspSubsystem(unittest.TestCase):
    """Automated integration tests for the vision pipeline using desk image sample."""

    def setUp(self):
        self.test_image_path = DEFAULT_TEST_IMAGE
        self.assertTrue(self.test_image_path.exists(), f"Sample test image not found at: {self.test_image_path}")
        self.stream = RealSenseStream(source=str(self.test_image_path))
        self.stream.start()
        self.estimator = GraspEstimator()
        self.detector = ToolDetector() if HAS_ULTRALYTICS else None

    def tearDown(self):
        self.stream.stop()

    def test_image_frame_acquisition(self):
        img, depth_m, intrinsics = self.stream.get_frame()
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 3)
        self.assertIsNone(depth_m)
        self.assertIsNotNone(intrinsics)
        self.assertGreater(intrinsics.fx, 100.0)

    @unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics required for ToolDetector")
    def test_tool_detection_from_image(self):
        img, depth_m, intrinsics = self.stream.get_frame()
        detections = self.detector.detect(img)
        self.assertGreaterEqual(len(detections), 1, "Should detect at least 1 object in sample desk image")

    def test_grasp_pose_estimation(self):
        img, depth_m, intrinsics = self.stream.get_frame()
        if HAS_ULTRALYTICS and self.detector is not None:
            detections = self.detector.detect(img)
        else:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (int(w * 0.12), int(h * 0.03)), 35, 0, 360, 255, -1)
            ring_x = int(cx + w * 0.10 * math.cos(math.radians(35)))
            ring_y = int(cy + w * 0.10 * math.sin(math.radians(35)))
            cv2.circle(mask, (ring_x, ring_y), int(h * 0.05), 255, -1)
            x1, y1 = max(0, cx - int(w * 0.15)), max(0, cy - int(h * 0.12))
            x2, y2 = min(w, cx + int(w * 0.15)), min(h, cy + int(h * 0.12))
            detections = [
                DetectionResult(
                    label="scissors",
                    confidence=0.95,
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                )
            ]
        self.assertTrue(len(detections) > 0, "Failed to detect objects in sample desk image")

        grasps = self.estimator.estimate_all(detections, intrinsics, depth_m)
        self.assertTrue(len(grasps) > 0, "Failed to estimate grasp pose from detections")

        top_g = grasps[0]
        # Check pixel boundaries
        self.assertGreater(top_g.pixel_grasp[0], 0.0)
        self.assertLess(top_g.pixel_grasp[0], intrinsics.width)
        # Check UR10e pose structure
        self.assertEqual(len(top_g.robot_approach_pose), 6)
        self.assertEqual(len(top_g.robot_pick_pose), 6)
        # Approach height should be scan height, pick height should be contact height
        self.assertAlmostEqual(top_g.robot_approach_pose[2], self.estimator.scan_height_m, places=3)
        self.assertAlmostEqual(top_g.robot_pick_pose[2], self.estimator.contact_height_m, places=3)
        # Check gripper width is bounded
        self.assertGreaterEqual(top_g.gripper_open_width_mm, 25.0)
        self.assertLessEqual(top_g.gripper_open_width_mm, 95.0)


def main():
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
        return

    parser = argparse.ArgumentParser(description="Lab Vision Recognition & Grasp Estimation Runner")
    parser.add_argument(
        "--source", type=str, default="auto",
        help="Input source: 'auto', 'realsense', or path to an image file (e.g. 'data/vision/desk.jpg')",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model checkpoint in src/vision/models/ (e.g. 'yolov8n-seg.pt' or 'surgical_tools.pt')",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    parser.add_argument("--save-snapshot", type=str, default=None, help="Save annotated output frame to file and exit")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening GUI window (headless/SSH)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark run count (default: 20)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated target classes")
    parser.add_argument("--no-mask", action="store_true", help="Disable gripper self-occlusion exclusion mask")
    parser.add_argument("--fps", type=int, default=15, help="Camera stream framerate (default: 15)")
    parser.add_argument("--exposure", type=int, default=66000, help="Manual exposure in microseconds (default: 66000)")
    parser.add_argument("--gain", type=int, default=32, help="Sensor hardware gain (default: 32)")
    parser.add_argument("--test", action="store_true", help="Run automated unit test suite")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    benchmark_runs = args.runs if args.benchmark else None

    run_interactive(
        source=args.source,
        model_name=args.model,
        conf=args.conf,
        save_snapshot=args.save_snapshot,
        no_gui=args.no_gui,
        benchmark_runs=benchmark_runs,
        target_classes=classes,
        enable_mask=not args.no_mask,
        fps=args.fps,
        exposure=args.exposure,
        gain=args.gain,
    )


if __name__ == "__main__":
    main()
