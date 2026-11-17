#!/usr/bin/env python3
"""
Camera Calibration Viewport Utility.

Minimalist camera viewer for manual camera-to-gripper offset calibration in WSL2.
Highlights the viewport center with crossing diagonal lines and a high-contrast center dot.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calib_viewer")


class CalibrationViewer:
    """Manages camera acquisition, crosshair reticle rendering, and display."""

    def __init__(
        self,
        device: str = "0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        show_hud: bool = True,
    ) -> None:
        self.device_str = device
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.show_hud = show_hud

        self.cap: Optional[cv2.VideoCapture] = None
        self.rs_stream = None
        self._is_realsense = False
        self.actual_width = width
        self.actual_height = height
        self._last_valid_frame: Optional[np.ndarray] = None
        self._consecutive_drops: int = 0

        # Web streaming state
        self._current_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._running = False

    def open_camera(self) -> bool:
        """Opens physical camera (Intel RealSense or standard OpenCV /dev/video device)."""
        # 1. RealSense check
        if self.device_str.lower() in ("realsense", "rs"):
            try:
                from src.vision.realsense_stream import HAS_REALSENSE, RealSenseStream

                if not HAS_REALSENSE:
                    logger.error("pyrealsense2 is not installed.")
                    return False
                self.rs_stream = RealSenseStream(
                    source="realsense",
                    width=self.target_width,
                    height=self.target_height,
                    fps=self.target_fps,
                    enable_depth=False,
                )
                self.rs_stream.start()
                self._is_realsense = True
                self.actual_width = self.rs_stream.width
                self.actual_height = self.rs_stream.height
                logger.info("RealSense stream opened (%dx%d, depth disabled for bandwidth)", self.actual_width, self.actual_height)
                return True
            except Exception as e:
                logger.error("Failed to open RealSense stream: %s", e)
                return False

        # 2. Standard V4L2 / USB device index
        try:
            device_index = int(self.device_str)
        except ValueError:
            device_index = self.device_str  # Path like /dev/video0

        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            logger.error(
                "Could not open camera device '%s'.\n"
                "Ensure device is attached to WSL2 (e.g., via 'usbipd attach --wsl --busid <BUSID>').",
                self.device_str,
            )
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Retrieve negotiated resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.target_height
        logger.info("Camera device '%s' opened (%dx%d)", self.device_str, self.actual_width, self.actual_height)
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a single color frame from the active camera."""
        if self._is_realsense and self.rs_stream:
            try:
                color, _, _ = self.rs_stream.get_frame()
                if color is not None:
                    self._last_valid_frame = color
                    self._consecutive_drops = 0
                return color
            except Exception as e:
                self._consecutive_drops += 1
                if self._consecutive_drops == 1 or self._consecutive_drops % 30 == 0:
                    logger.warning(
                        "RealSense frame dropped (%s), reusing last frame (drops: %d).",
                        e, self._consecutive_drops,
                    )
                if self._consecutive_drops > 10:
                    try:
                        logger.info("Re-synchronizing RealSense pipeline after %d drops...", self._consecutive_drops)
                        self.rs_stream.stop()
                        time.sleep(0.1)
                        self.rs_stream.start()
                        self._consecutive_drops = 0
                    except Exception as resync_err:
                        logger.debug("Pipeline resync failed: %s", resync_err)
                return self._last_valid_frame

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._last_valid_frame = frame
                return frame
        return self._last_valid_frame

    def draw_overlays(self, frame: np.ndarray, fps: float = 0.0) -> np.ndarray:
        """
        Draws the crosshair reticle:
        - Two diagonal lines crossing corner-to-corner through the center.
        - Center dot at (cx, cy).
        - Optional minimal HUD overlay.
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Diagonal Crossing Lines (Dual-tone for contrast on any background)
        # Background shadow (black, thickness 3)
        cv2.line(frame, (0, 0), (w - 1, h - 1), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(frame, (w - 1, 0), (0, h - 1), (0, 0, 0), 3, cv2.LINE_AA)
        # Foreground line (bright green, thickness 1)
        cv2.line(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (w - 1, 0), (0, h - 1), (0, 255, 0), 1, cv2.LINE_AA)

        # 2. Center Dot (Radius 4 black border, Radius 2 bright green center)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)

        # 3. Minimal HUD
        if self.show_hud:
            hud_text_1 = f"RES: {w}x{h} | CENTER: ({cx}, {cy}) | FPS: {fps:.1f}"
            hud_text_2 = "KEYS: [h] Toggle HUD | [q/ESC] Quit"

            for i, text in enumerate((hud_text_1, hud_text_2)):
                y = 25 + i * 22
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        return frame

    def run_gui(self) -> None:
        """Runs the interactive OpenCV window display."""
        window_name = "Camera Calibration Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                prev_time = curr_time

                display_frame = self.draw_overlays(frame, fps=fps)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("h"):
                    self.show_hud = not self.show_hud
        finally:
            cv2.destroyAllWindows()
            self.close()

    def run_web(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Runs headless HTTP MJPEG streaming server."""
        self._running = True

        def capture_loop():
            prev_time = time.time()
            fps = 0.0
            while self._running:
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                prev_time = curr_time

                display_frame = self.draw_overlays(frame, fps=fps)
                ret, buf = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    with self._jpeg_lock:
                        self._current_jpeg = buf.tobytes()
                time.sleep(0.01)

        cap_thread = threading.Thread(target=capture_loop, daemon=True)
        cap_thread.start()

        viewer_self = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # Suppress noisy HTTP request logging

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    html = (
                        "<!DOCTYPE html><html><head><title>Camera Calibration Viewer</title>"
                        "<style>"
                        "body{margin:0;background:#111;color:#eee;font-family:sans-serif;"
                        "display:flex;flex-direction:column;align-items:center;"
                        "justify-content:center;min-height:100vh;}"
                        "img{border:2px solid #444;box-shadow:0 4px 12px rgba(0,0,0,0.5);max-width:95vw;}"
                        "p{margin-top:10px;color:#aaa;font-size:14px;}"
                        "</style></head><body>"
                        "<h2>Camera Calibration Viewer</h2>"
                        '<img src="/video_feed" alt="Camera Stream" />'
                        "<p>WSL2 HTTP Stream &bull; Press Ctrl+C in terminal to stop</p>"
                        "</body></html>"
                    )
                    self.wfile.write(html.encode("utf-8"))
                elif self.path == "/video_feed":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while viewer_self._running:
                            with viewer_self._jpeg_lock:
                                jpeg = viewer_self._current_jpeg
                            if jpeg is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(jpeg)))
                                self.end_headers()
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.033)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer((host, port), MJPEGHandler)
        logger.info("Web calibration stream running at http://localhost:%d", port)
        logger.info("Open this URL in your Windows browser to view the crosshair feed.")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            server.server_close()
            self.close()

    def close(self) -> None:
        """Releases camera resources cleanly."""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.rs_stream:
            try:
                self.rs_stream.stop()
            except Exception:
                pass
            self.rs_stream = None
        logger.info("Camera resources released.")


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Viewport Utility (WSL2)")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Camera device index (0, 1, ...) or 'realsense' (default: 0)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Stream via local HTTP server for Windows browser viewing",
    )
    parser.add_argument("--port", type=int, default=8080, help="Web streaming port (default: 8080)")
    args = parser.parse_args()

    viewer = CalibrationViewer(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    if not viewer.open_camera():
        sys.exit(1)

    # Determine display mode: explicit --web, or GUI if DISPLAY/WAYLAND_DISPLAY is available
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.web or not has_display:
        if not args.web and not has_display:
            logger.info("No GUI display detected in WSL session. Falling back to web streaming mode.")
        viewer.run_web(port=args.port)
    else:
        try:
            viewer.run_gui()
        except cv2.error as e:
            logger.warning("OpenCV GUI failed (%s). Switching to web stream mode.", e)
            viewer.run_web(port=args.port)
        except KeyboardInterrupt:
            logger.info("Calibration viewer stopped by user.")
        except Exception as e:
            logger.error("Calibration viewer runtime error: %s", e)


if __name__ == "__main__":
    main()
