# Camera Calibration Viewer (WSL2)

Temporary viewer program with corner-to-corner diagonal crosshair lines and center dot for manual camera-to-gripper offset calibration (`src/actuators/sim_ws/calibrate_offset.py`).

## Quick Start

### 1. Launch Native Window (WSLg / OpenCV GUI)
```bash
python3 -m src.vision.calib_viewer.viewer
```
Or directly:
```bash
python3 src/vision/calib_viewer/viewer.py
```

### 2. View in Windows Web Browser (`http://localhost:8080`)
If running without a GUI display or if you prefer Chrome/Edge on Windows:
```bash
python3 -m src.vision.calib_viewer.viewer --web
```
Then open `http://localhost:8080` in your Windows browser.

### 3. Intel RealSense Camera
If connecting directly to an Intel RealSense camera:
```bash
python3 -m src.vision.calib_viewer.viewer --device realsense
```

## CLI Options

| Argument | Default | Description |
|---|---|---|
| `--device` | `0` | Camera device index (`0`, `1`, `/dev/video0`) or `realsense` |
| `--width` | `640` | Target camera frame width |
| `--height` | `480` | Target camera frame height |
| `--fps` | `30` | Camera frame rate |
| `--web` | `False` | Run lightweight HTTP MJPEG stream at `http://localhost:8080` |
| `--port` | `8080` | Port for the HTTP stream |

## Hotkeys (OpenCV Window)

- `h`: Toggle HUD overlay (resolution, FPS, center coordinate)
- `q` / `ESC`: Quit viewer

## Cleanup (Delete when done)

Once calibration is completed, this entire folder can be deleted:
```bash
rm -rf src/vision/calib_viewer
```
