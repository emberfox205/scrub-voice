"""
Vision recognition and grasp pose estimation subsystem for Universal Robots UR10e.

Provides Intel RealSense RGB-D capture, object segmentation (YOLOv8-Seg / YOLO-World / Classical),
oriented grasp pose estimation, and camera-to-robot coordinate projection.
"""

from src.vision.detector import DetectionResult, ToolDetector
from src.vision.grasp_estimator import GraspEstimate, GraspEstimator
from src.vision.realsense_stream import CameraIntrinsics, RealSenseStream

__all__ = [
    "RealSenseStream",
    "CameraIntrinsics",
    "ToolDetector",
    "DetectionResult",
    "GraspEstimator",
    "GraspEstimate",
]
