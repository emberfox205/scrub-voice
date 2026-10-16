#!/usr/bin/env python3
"""
Camera-to-Gripper Offset Calibration Utility

Calculates the physical translation and angular offset between the
eye-in-hand RealSense camera and the OnRobot RG-2 gripper tips.

Procedure:
1. Align a physical dot/marker with the center of the camera view -> Record Pose A.
2. In Freedrive, touch the gripper finger tips to that exact same dot -> Record Pose B.
3. Automatically computes ΔX, ΔY, ΔZ and saves to camera_extrinsics.json.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import rtde_receive
except ImportError:
    print("[ERROR] rtde_receive is not installed. Please activate your 'ur_arm' environment.")
    sys.exit(1)


OUTPUT_FILE = Path(__file__).resolve().parent.parent / "camera_extrinsics.json"
CANDIDATE_IPS = [
    os.getenv("ROBOT_IP"),
    "192.168.1.105",
    "192.168.1.102",
    "127.0.0.1",
]


def resolve_robot_ip(specified_ip: str | None) -> str:
    if specified_ip:
        return specified_ip
    import socket
    for ip in CANDIDATE_IPS:
        if not ip:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((ip, 30004)) == 0:
                    return ip
        except Exception:
            pass
    return "192.168.1.105"


def main():
    parser = argparse.ArgumentParser(description="Calibrate Camera-to-Gripper offset.")
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP address")
    args = parser.parse_args()

    robot_ip = resolve_robot_ip(args.robot_ip)
    print("=" * 65)
    print(" CAMERA-TO-GRIPPER CALIBRATION (Touch-the-Dot)")
    print("=" * 65)
    print(f"Connecting to UR RTDE at {robot_ip}:30004...")

    try:
        r = rtde_receive.RTDEReceiveInterface(robot_ip)
    except Exception as e:
        print(f"[ERROR] Could not connect to robot: {e}")
        sys.exit(1)

    print("Connected successfully!\n")

    # Step 1: Camera alignment
    print("--- STEP 1: ALIGN CAMERA ---")
    print("1. Look at your RealSense camera feed on your screen.")
    print("2. In Freedrive, move the arm until your small 2x2 cm paper marker")
    print("   is DEAD CENTER in the camera crosshair/window.")
    input("\nPress [ENTER] when the camera is centered over the marker > ")

    pose_cam = [round(float(v), 5) for v in r.getActualTCPPose()]
    q_cam = [round(float(v), 5) for v in r.getActualQ()]
    print(f"✓ Recorded Camera Pose: TCP {pose_cam}")

    # Step 2: Gripper alignment
    print("\n--- STEP 2: TOUCH WITH GRIPPER ---")
    print("1. Keep that same marker in the exact same spot on the table.")
    print("2. In Freedrive, move the arm until the closed tips of the RG-2")
    print("   gripper fingers are TOUCHING that exact marker.")
    input("\nPress [ENTER] when the gripper tips touch the marker > ")

    pose_grip = [round(float(v), 5) for v in r.getActualTCPPose()]
    q_grip = [round(float(v), 5) for v in r.getActualQ()]
    print(f"✓ Recorded Gripper Pose: TCP {pose_grip}")

    # Calculate offsets
    dx = round(pose_grip[0] - pose_cam[0], 5)
    dy = round(pose_grip[1] - pose_cam[1], 5)
    dz = round(pose_grip[2] - pose_cam[2], 5)

    drx_deg = round((pose_grip[3] - pose_cam[3]) * 57.2957795, 2)
    dry_deg = round((pose_grip[4] - pose_cam[4]) * 57.2957795, 2)
    drz_deg = round((pose_grip[5] - pose_cam[5]) * 57.2957795, 2)

    result = {
        "translation_offset_m": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
        "translation_offset_mm": {
            "dx": round(dx * 1000.0, 1),
            "dy": round(dy * 1000.0, 1),
            "dz": round(dz * 1000.0, 1),
        },
        "tilt_offset_deg": {
            "rx": drx_deg,
            "ry": dry_deg,
            "rz": drz_deg,
        },
        "calibration_poses": {
            "camera_tcp": pose_cam,
            "camera_joints": q_cam,
            "gripper_tcp": pose_grip,
            "gripper_joints": q_grip,
        },
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }

    print("\n" + "=" * 65)
    print(" CALIBRATION RESULTS")
    print("=" * 65)
    print(f" Linear Offset (mm) : ΔX={result['translation_offset_mm']['dx']} mm, ΔY={result['translation_offset_mm']['dy']} mm, ΔZ={result['translation_offset_mm']['dz']} mm")
    print(f" Angular Tilt (deg) : Rx={drx_deg}°, Ry={dry_deg}°, Rz={drz_deg}°")
    print("=" * 65)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved calibration config to:\n  {OUTPUT_FILE}")
    r.disconnect()
    print("Disconnected RTDE.")


if __name__ == "__main__":
    main()
