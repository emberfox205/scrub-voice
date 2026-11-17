"""Unit tests for CalibrationViewer reticle rendering and math."""

import numpy as np
import pytest

from src.vision.calib_viewer.viewer import CalibrationViewer


def test_reticle_overlays_and_center_alignment():
    viewer = CalibrationViewer(device="0", width=640, height=480)
    # Create black canvas
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated = viewer.draw_overlays(blank_frame.copy(), fps=30.0)

    # Output shape and type must match
    assert annotated.shape == (480, 640, 3)
    assert annotated.dtype == np.uint8

    cx, cy = 640 // 2, 480 // 2
    # Verify that the center dot was rendered at (cx, cy)
    # Bright green is (0, 255, 0) in BGR
    center_pixel = annotated[cy, cx]
    assert center_pixel[1] > 200, f"Expected high green intensity at center, got {center_pixel}"

    # Verify corner pixels (start of diagonal lines)
    # Top-left corner (0, 0)
    assert annotated[0, 0][1] > 200 or annotated[0, 0].sum() > 0
    # Bottom-right corner (479, 639)
    assert annotated[479, 639][1] > 200 or annotated[479, 639].sum() > 0


def test_hud_toggle():
    viewer = CalibrationViewer(device="0", width=640, height=480, show_hud=False)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated_no_hud = viewer.draw_overlays(blank_frame.copy(), fps=30.0)
    # Top-left region should be black when HUD is off (except diagonal crossing near 0,0)
    assert np.all(annotated_no_hud[25, 50] == [0, 0, 0])

    viewer.show_hud = True
    annotated_with_hud = viewer.draw_overlays(blank_frame.copy(), fps=30.0)
    # Top-left text region should have HUD pixels
    assert np.any(annotated_with_hud[15:40, 15:200] > 0)


def test_missing_device_exits_cleanly():
    # Attempting to open an invalid device number should fail cleanly without unhandled exception
    viewer = CalibrationViewer(device="999")
    success = viewer.open_camera()
    assert success is False
    viewer.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
