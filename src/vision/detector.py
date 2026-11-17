#!/usr/bin/env python3
"""
YOLO Instance Segmentation Detector for Surgical Tools & Instruments.

Loads fine-tuned Roboflow checkpoints or pre-trained YOLOv8-seg weights to detect
tools and extract pixel-accurate binary masks for grasp pose estimation.
Supports static robot self-occlusion masking to ignore wrist-mounted grippers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False

os.environ["YOLO_OFFLINE"] = "True"
logger = logging.getLogger("vision.detector")

# Path to local models directory
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_NAME = "yolov8n-seg.pt"

DEFAULT_EXTRINSICS_PATH = (
    Path(__file__).resolve().parent.parent / "actuators" / "camera_extrinsics.json"
)

# Normalized [x, y] coordinates of robot gripper exclusion zone in camera FOV
# Accurately envelopes the RG-2 gripper fingers and chassis when touching together at tray_pose
# following the exact contour of the visible gripper portion while minimizing masked tray space
DEFAULT_GRIPPER_EXCLUSION_POLYGON: List[Tuple[float, float]] = [
    (0.500, 1.000),
    (0.525, 0.780),
    (0.555, 0.690),
    (0.620, 0.650),
    (0.740, 0.650),
    (0.820, 0.700),
    (1.000, 0.780),
    (1.000, 1.000),
]

# Canonical surgical and operating room instruments
DEFAULT_SURGICAL_CLASSES = {
    "scissors",
    "scalpel",
    "knife",
    "forceps",
    "clamp",
    "tweezers",
    "hemostat",
    "needle_driver",
}


@dataclass
class DetectionResult:
    """Individual object detection and segmentation result."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2) in pixel space
    mask: np.ndarray                      # Binary uint8 mask (0 or 255) [H, W]


def resolve_model_path(model_name: Optional[str] = None) -> Path:
    """Returns path to model checkpoint inside src/vision/models/."""
    filename = Path(model_name).name if model_name else DEFAULT_MODEL_NAME
    return MODELS_DIR / filename


def load_exclusion_polygon(
    custom_polygon: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Loads normalized exclusion polygon from custom argument or camera_extrinsics.json."""
    if custom_polygon:
        return custom_polygon
    if DEFAULT_EXTRINSICS_PATH.exists():
        try:
            with open(DEFAULT_EXTRINSICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "gripper_exclusion_polygon" in data:
                    return [tuple(pt) for pt in data["gripper_exclusion_polygon"]]
        except Exception as e:
            logger.debug("Could not read exclusion polygon from %s: %s", DEFAULT_EXTRINSICS_PATH, e)
    return DEFAULT_GRIPPER_EXCLUSION_POLYGON


def detect_simulated_tools(
    image_bgr: np.ndarray,
    target_classes: Optional[List[str]] = None,
    exclusion_polygon: Optional[List[Tuple[float, float]]] = None,
) -> List[DetectionResult]:
    """
    Lightweight color and contour-based segmentation detector for simulated
    surgical tools (scissors and scalpel) in the MuJoCo surgical tray.
    Enables authentic segmentation directly on simulated camera frames
    without requiring external neural network checkpoints.
    """
    h, w = image_bgr.shape[:2]
    allowed = [c.lower() for c in target_classes] if target_classes else ["scissors", "scalpel"]
    detections: List[DetectionResult] = []
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    poly = exclusion_polygon or DEFAULT_GRIPPER_EXCLUSION_POLYGON
    poly_px = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
    gripper_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(gripper_mask, [poly_px], 255)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Metallic steel mask (for scissors blades and stainless steel scalpel)
    steel_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 75, 255])) & (gripper_mask == 0)

    # 1. Kitchen / Surgical Scissors (distinctive red handles + metallic shank and blades)
    scissors_mask = np.zeros((h, w), dtype=np.uint8)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    red_mask = (m1 | m2) & (gripper_mask == 0)

    if np.count_nonzero(red_mask) > 30:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 150))
        dilated = cv2.dilate(red_mask, kernel_dilate, iterations=1)
        combined = red_mask | (steel_mask & (dilated > 0))
        scissors_mask = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

        # Filter out any disconnected background reflections
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(scissors_mask)
        clean_mask = np.zeros_like(scissors_mask)
        for lbl in range(1, num_labels):
            if np.count_nonzero((labels == lbl) & (red_mask > 0)) > 20:
                clean_mask[labels == lbl] = 255

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 80:
                x, y, bw, bh = cv2.boundingRect(cnt)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                scissors_mask = mask
                if any("scissor" in c for c in allowed):
                    detections.append(
                        DetectionResult(
                            label="scissors",
                            confidence=0.95,
                            bbox=(x, y, x + bw, y + bh),
                            mask=mask,
                        )
                    )

    # 2. Scalpel (stainless steel blade + ribbed handle)
    if any("scalpel" in c or "knife" in c for c in allowed):
        # Steel regions not part of scissors and not on gripper
        scalpel_candidate = steel_mask & (scissors_mask == 0)
        scalpel_candidate = cv2.morphologyEx(scalpel_candidate, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(scalpel_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 150:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Keep contours inside the tray (away from extreme borders)
                if x > 30 and y > 30 and (x + bw) < (w - 30):
                    aspect = max(bw, bh) / max(min(bw, bh), 1)
                    if aspect > 1.8:
                        valid_cnts.append((cnt, area, (x, y, bw, bh)))

        if valid_cnts:
            cnt, area, (x, y, bw, bh) = max(valid_cnts, key=lambda item: item[1])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            detections.append(
                DetectionResult(
                    label="scalpel",
                    confidence=0.95,
                    bbox=(x, y, x + bw, y + bh),
                    mask=mask,
                )
            )

    return detections


class ToolDetector:
    """
    Lean YOLO instance segmentation detector with self-occlusion exclusion masking.
    Falls back to color-based tool segmentation when ultralytics weights are unavailable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: float = 0.20,
        exclusion_polygon: Optional[List[Tuple[float, float]]] = None,
        enable_mask: bool = True,
        allow_fallback: bool = True,
    ) -> None:
        """
        Args:
            model_name: Checkpoint filename in src/vision/models/ (e.g. 'surgical_tools.pt').
            conf_threshold: Minimum confidence score to retain detection.
            exclusion_polygon: Optional custom normalized polygon [(x, y), ...] for gripper blindspot.
            enable_mask: Whether to mask out the robot gripper corner (default: True).
            allow_fallback: Allow color-based segmentation fallback if YOLO unavailable.
        """
        if not HAS_ULTRALYTICS and not allow_fallback:
            raise ImportError(
                "The 'ultralytics' package is required for ToolDetector. "
                "Install it using: pip install ultralytics"
            )

        self.conf_threshold = conf_threshold
        self.enable_mask = enable_mask
        self.exclusion_polygon = load_exclusion_polygon(exclusion_polygon) if enable_mask else None
        self.checkpoint_path = resolve_model_path(model_name)
        self.model = None
        self.model_name = self.checkpoint_path.name if self.checkpoint_path else "color_fallback"
        self.model_path = str(self.checkpoint_path) if self.checkpoint_path else ""

        if HAS_ULTRALYTICS:
            try:
                self.model = YOLO(str(self.checkpoint_path))
                self.model_name = self.checkpoint_path.name
                self.model_path = str(self.checkpoint_path)
                logger.info(
                    "Initializing YOLO segmentation engine with: %s (Mask enabled: %s)",
                    self.checkpoint_path, self.enable_mask,
                )
            except Exception as e:
                logger.warning("Could not load YOLO model: %s. Using color-based fallback.", e)
        else:
            logger.info("Ultralytics not installed. ToolDetector using color-based segmentation fallback.")

    def detect(
        self,
        image_bgr: np.ndarray,
        target_classes: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """
        Runs YOLO segmentation inference and extracts polygon masks.
        Falls back to color-based segmentation if YOLO model is unavailable or finds no tools.

        Args:
            image_bgr: BGR color image [H, W, 3].
            target_classes: Optional class names to filter by (case-insensitive).

        Returns:
            List of DetectionResult instances sorted by confidence descending.
        """
        h, w = image_bgr.shape[:2]
        allowed = [c.lower() for c in target_classes] if target_classes else None

        # Apply exclusion mask to blank out the physical gripper before inference
        infer_img = image_bgr
        poly_px = None
        if self.enable_mask and self.exclusion_polygon:
            infer_img = image_bgr.copy()
            poly_px = np.array(
                [[int(x * w), int(y * h)] for x, y in self.exclusion_polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(infer_img, [poly_px], (0, 0, 0))

        detections: List[DetectionResult] = []
        if self.model is not None:
            results = self.model.predict(
                source=infer_img,
                conf=self.conf_threshold,
                verbose=False,
            )
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                first_res = results[0]
                boxes = first_res.boxes
                masks = first_res.masks
                names = self.model.names

                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    label = names.get(cls_id, f"class_{cls_id}").lower()
                    conf = float(box.conf[0].item())

                    if allowed:
                        if not any(cand in label for cand in allowed):
                            continue
                    else:
                        if not any(cand in label for cand in DEFAULT_SURGICAL_CLASSES):
                            continue

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    if poly_px is not None:
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        if cv2.pointPolygonTest(poly_px, (cx, cy), False) >= 0:
                            continue

                    if masks is not None and len(masks) > idx:
                        raw_mask = masks[idx].data[0].cpu().numpy()
                        mask_resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    else:
                        mask_binary = np.zeros((h, w), dtype=np.uint8)
                        mask_binary[y1:y2, x1:x2] = 255

                    detections.append(
                        DetectionResult(
                            label=label,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            mask=mask_binary,
                        )
                    )

        # Fallback to simulated tool segmentation if no neural detections
        if not detections:
            detections = detect_simulated_tools(
                infer_img,
                target_classes=target_classes,
                exclusion_polygon=self.exclusion_polygon,
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
