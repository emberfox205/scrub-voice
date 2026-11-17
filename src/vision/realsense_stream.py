#!/usr/bin/env python3
"""
Intel RealSense RGB-D & Still Image Streamer.

Acquires aligned color and depth frames from Intel RealSense D400-series cameras,
simulation environments (MuJoCo), or static test images on disk.
Operates in deterministic manual exposure mode for stable, flicker-free robotic vision.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("vision.realsense")

DEFAULT_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "vision" / "desk.jpg"
)

# Calibrated nominal indoor lighting defaults (300-600 lux)
DEFAULT_EXPOSURE_US = 66000  # 66ms nominal exposure (full 15 FPS shutter window, 2x light collection)
DEFAULT_GAIN = 32            # Clean analog sensor gain
DEFAULT_WARMUP_FRAMES = 2    # Fixed minimal buffer clearance (instantaneous ~0.13s @ 15fps)

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def deproject_pixel_to_point(self, u: float, v: float, depth_m: float) -> Tuple[float, float, float]:
        """
        Deprojects a 2D image pixel (u, v) and depth (meters) to 3D camera coordinates (X, Y, Z).
        Convention: +X is right, +Y is down, +Z is forward along the optical axis.
        """
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return (x, y, z)


class RealSenseStream:
    """
    Manages frame acquisition from Intel RealSense, MuJoCo simulation, or static image files.
    Operates with locked manual exposure and analog gain for reliable machine vision.
    """

    def __init__(
        self,
        source: str = "auto",
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        exposure: int = DEFAULT_EXPOSURE_US,
        gain: int = DEFAULT_GAIN,
        enable_depth: bool = True,
    ) -> None:
        """
        Args:
            source: 'auto' (RealSense with image fallback), 'realsense', 'sim', or image path.
            width: Image width in pixels (default: 640 for robust USB 2/3 compatibility).
            height: Image height in pixels (default: 480).
            fps: Frame rate for camera streams (default: 15 for 2x exposure window).
            exposure: Manual exposure in microseconds (default: 66000 μs / 66ms).
            gain: Hardware analog gain (default: 32).
            enable_depth: Whether to negotiate and align depth stream (default: True).
        """
        self.requested_source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.gain = gain
        self.enable_depth = enable_depth

        self.active_backend: Optional[str] = None
        self.pipeline: Optional[Any] = None
        self.align: Optional[Any] = None
        self._sim: Optional[Any] = None
        self.static_image: Optional[np.ndarray] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self._is_running = False

    def start(self) -> str:
        """Initializes and starts the selected camera or image backend."""
        if self._is_running:
            return self.active_backend or "unknown"

        src_lower = self.requested_source.lower()

        # 1. Simulation Backend
        if src_lower == "sim" or (src_lower == "auto" and os.environ.get("ROBOT_BACKEND") == "sim"):
            from src.actuators.hal import get_shared_sim
            self._sim = get_shared_sim()
            self.active_backend = "sim"
            self._is_running = True
            _, _, self.intrinsics = self._sim.capture_camera_frame(
                "wrist_realsense", width=self.width, height=self.height
            )
            logger.info("Simulation wrist camera stream active (%dx%d)", self.width, self.height)
            return self.active_backend

        # 2. Explicit File Input
        if src_lower not in ("auto", "realsense", "sim"):
            file_path = Path(self.requested_source)
            if not file_path.is_absolute() and not file_path.exists():
                candidate = DEFAULT_IMAGE_PATH.parent / file_path.name
                if candidate.exists():
                    file_path = candidate
            if not file_path.is_file():
                raise FileNotFoundError(f"Specified image source not found: {self.requested_source}")
            return self._init_file(file_path)

        # 3. Physical Intel RealSense
        if src_lower == "realsense" and not HAS_REALSENSE:
            raise ImportError(
                "The 'pyrealsense2' package is required to connect to the Intel RealSense camera.\n"
                "Please install it via: pip install pyrealsense2"
            )

        if self._try_init_realsense():
            self.active_backend = "realsense"
            self._is_running = True
            logger.info("RealSense stream active at %dx%d @ %dfps", self.width, self.height, self.fps)
            return self.active_backend

        if src_lower == "realsense":
            raise RuntimeError(
                "Intel RealSense requested, but no compatible device was detected by librealsense.\n"
                "If using WSL2 with usbipd, ensure USB device access permissions are granted:\n"
                "  1. Test permissions: sudo chmod -R 777 /dev/bus/usb\n"
                "  2. Or install udev rules: sudo apt install -y librealsense2-udev-rules"
            )

        # 4. Fallback: Default test image from data/vision/desk.jpg
        if DEFAULT_IMAGE_PATH.exists():
            return self._init_file(DEFAULT_IMAGE_PATH, is_fallback=True)

        raise RuntimeError(
            f"No RealSense camera detected and default fallback image not found at {DEFAULT_IMAGE_PATH}."
        )

    def _init_file(self, path: Path, is_fallback: bool = False) -> str:
        """Loads a static image from disk and derives pinhole camera intrinsics."""
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to decode image at {path}")

        self.static_image = img
        h, w = img.shape[:2]
        self.width, self.height = w, h
        fx = float(w) * 0.9
        self.intrinsics = CameraIntrinsics(
            fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0, width=w, height=h
        )
        self.active_backend = "file"
        self._is_running = True

        prefix = "RealSense not found. Using default test image" if is_fallback else "Using still image input"
        logger.info("%s: %s (%dx%d)", prefix, path, self.width, self.height)
        return self.active_backend

    def _try_init_realsense(self) -> bool:
        if not HAS_REALSENSE:
            logger.debug("pyrealsense2 is not installed.")
            return False

        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                logger.debug("No RealSense devices found on USB bus.")
                return False

            device = devices[0]
            dev_name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            usb_type = (
                device.get_info(rs.camera_info.usb_type_descriptor)
                if device.supports(rs.camera_info.usb_type_descriptor)
                else "unknown"
            )
            is_usb2 = "2." in str(usb_type)
            logger.info("Found RealSense device: %s (Serial: %s, USB: %s)", dev_name, serial, usb_type)

            # Build candidates ordered by bandwidth efficiency
            if is_usb2:
                # USB 2.0 / USB 2.1: hardware MJPEG prevents bus choking at high resolutions
                candidates = [
                    (self.width, self.height, self.fps, rs.format.mjpeg),
                    (640, 480, min(self.fps, 30), rs.format.bgr8),
                    (640, 480, 15, rs.format.bgr8),
                    (424, 240, min(self.fps, 30), rs.format.bgr8),
                ]
            else:
                # USB 3.0+: full uncompressed BGR streams
                candidates = [
                    (self.width, self.height, self.fps, rs.format.bgr8),
                    (848, 480, min(self.fps, 30), rs.format.bgr8),
                    (640, 480, min(self.fps, 30), rs.format.bgr8),
                    (640, 480, 15, rs.format.bgr8),
                ]

            profile = None
            pipe = None
            for w_cand, h_cand, fps_cand, fmt_cand in list(dict.fromkeys(candidates)):
                try:
                    cfg = rs.config()
                    cfg.enable_stream(rs.stream.color, w_cand, h_cand, fmt_cand, fps_cand)
                    if self.enable_depth:
                        cfg.enable_stream(rs.stream.depth, w_cand, h_cand, rs.format.z16, fps_cand)
                    p = rs.pipeline()
                    profile = p.start(cfg)
                    pipe = p
                    self.width, self.height, self.fps = w_cand, h_cand, fps_cand
                    logger.info("RealSense stream negotiated at %dx%d @ %dfps (%s)%s",
                                self.width, self.height, self.fps, fmt_cand,
                                " [depth enabled]" if self.enable_depth else " [color only]")
                    break
                except Exception as ex:
                    logger.debug("Profile %dx%d@%d (%s) failed on USB %s: %s",
                                 w_cand, h_cand, fps_cand, fmt_cand, usb_type, ex)
                    continue

            if pipe is None:
                raise RuntimeError(
                    f"Unable to negotiate matching stream on USB {usb_type}. "
                    "Ensure device is connected to a high-speed USB 3.0 port."
                )

            self.pipeline = pipe
            if self.enable_depth:
                self.align = rs.align(rs.stream.color)
            else:
                self.align = None

            # Retrieve optical intrinsics from color stream profile
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            raw_intrinsics = color_stream.get_intrinsics()

            self.width = raw_intrinsics.width
            self.height = raw_intrinsics.height
            self.intrinsics = CameraIntrinsics(
                fx=float(raw_intrinsics.fx),
                fy=float(raw_intrinsics.fy),
                cx=float(raw_intrinsics.ppx),
                cy=float(raw_intrinsics.ppy),
                width=raw_intrinsics.width,
                height=raw_intrinsics.height,
            )

            # Lock sensor to deterministic manual exposure and analog gain
            self._configure_color_sensor(profile)
            return True

        except Exception as e:
            logger.warning("Failed to initialize RealSense pipeline: %s", e)
            self.stop()
            return False

    def _configure_color_sensor(self, profile: Any) -> None:
        """Locks sensor to deterministic manual exposure and analog gain."""
        try:
            dev = profile.get_device()
            color_sensor = dev.first_color_sensor()
            if not color_sensor:
                return

            # Explicitly disable auto-exposure (eliminates all AE hunting & underexposure)
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)

            # Set hardware shutter duration in microseconds
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(self.exposure))

            # Set analog sensor gain
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, float(self.gain))

            # Auto White Balance
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

            logger.info("RealSense color sensor locked to manual mode: exposure=%s μs, gain=%s",
                        self.exposure, self.gain)
        except Exception as ex:
            logger.debug("Could not configure color sensor options: %s", ex)

    def get_frame(self) -> Tuple[np.ndarray, Optional[np.ndarray], CameraIntrinsics]:
        """
        Captures or retrieves the frame from the active backend.

        Returns:
            Tuple of:
                - color_image (np.ndarray): Clean BGR uint8 image [H, W, 3].
                - depth_image_m (Optional[np.ndarray]): Depth map in meters (float32) or None.
                - intrinsics (CameraIntrinsics): Focal lengths and optical center.
        """
        if not self._is_running:
            self.start()

        if self.intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable.")

        if self.active_backend == "file" and self.static_image is not None:
            return self.static_image.copy(), None, self.intrinsics

        if self.active_backend == "sim" and self._sim is not None:
            color_image, depth_image, intrinsics = self._sim.capture_camera_frame(
                "wrist_realsense", width=self.width, height=self.height
            )
            self.intrinsics = intrinsics
            return color_image, depth_image, intrinsics

        if self.active_backend == "realsense" and self.pipeline:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            if self.enable_depth and self.align:
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    raise RuntimeError("Failed to acquire valid frames from RealSense pipeline.")

                raw_color = np.asanyarray(color_frame.get_data())
                color_image = cv2.imdecode(raw_color, cv2.IMREAD_COLOR) if raw_color.ndim == 1 else raw_color

                # RealSense z16 is in millimeters -> convert to meters
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()

                return color_image, depth_image, self.intrinsics
            else:
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("Failed to acquire valid color frame from RealSense pipeline.")

                raw_color = np.asanyarray(color_frame.get_data())
                color_image = cv2.imdecode(raw_color, cv2.IMREAD_COLOR) if raw_color.ndim == 1 else raw_color

                return color_image, None, self.intrinsics

        raise RuntimeError(f"Streamer backend '{self.active_backend}' is not initialized or invalid.")

    def capture_snapshot(
        self,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], CameraIntrinsics]:
        """
        Executes on-demand 'snap-and-stop' frame acquisition:
        Starts the pipeline, clears initial FIFO buffer (default: 2 frames / ~0.06s),
        retrieves a single aligned (RGB, depth, intrinsics) tuple, and immediately stops
        the pipeline to reduce CPU, USB bandwidth, and thermal load on the Raspberry Pi.

        Args:
            warmup_frames: Number of initial buffer frames to skip (default: 2).

        Returns:
            Tuple of (color_image, depth_image_m, intrinsics).
        """
        try:
            if not self._is_running:
                self.start()

            # Discard initial frames to flush FIFO buffer
            if self.active_backend == "realsense" and self.pipeline:
                for _ in range(max(0, warmup_frames)):
                    try:
                        self.pipeline.wait_for_frames(timeout_ms=2000)
                    except Exception as e:
                        logger.debug("Warmup frame skip: %s", e)

            return self.get_frame()
        finally:
            self.stop()

    def stop(self) -> None:
        """Releases all camera and pipeline resources cleanly."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

        self.align = None
        self._sim = None
        self._is_running = False
        logger.info("Streamer stopped.")

    def __enter__(self) -> RealSenseStream:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
