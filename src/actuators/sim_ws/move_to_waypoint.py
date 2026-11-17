#!/usr/bin/env python3
"""
Move to Waypoint Utility for Universal Robots (UR5 / UR10e / URSim)

Loads waypoints from waypoints.json and safely moves the robot arm
to the specified target pose (defaults to 'tray_pose') using moveJ.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Dynamic sys.path setup according to AGENTS.md Rule 4
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.hal import get_robot_interfaces


# --- Calibrated Tray Z Coordinates ---
# 1. Scanning / Standoff Height (tray_pose): TCP Z = 0.17045 m (~17.0 cm)
#    Camera views the tray; (dx=-43.1mm, dy=+15.3mm) offset is valid at this height.
# 2. Table Contact / Grasp Height (tray_contact_pose): TCP Z = 0.02561 m (~2.56 cm)
#    RG-2 gripper fingertips touch table surface to grasp tools (dz = -0.14484 m plunge).
CALIBRATED_SCAN_Z_M = 0.17045
CALIBRATED_TABLE_Z_M = 0.02561

WAYPOINTS_FILE = Path(__file__).resolve().parent.parent / "waypoints.json"
CANDIDATE_IPS = [
    os.getenv("ROBOT_IP"),
    "192.168.1.105",
    "192.168.1.102",
    "127.0.0.1",
]


def resolve_robot_ip(specified_ip: str | None) -> str:
    """Resolves robot IP from CLI argument, environment, or active network probe."""
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


def load_waypoints(file_path: Path) -> dict:
    if not file_path.exists():
        print(f"[ERROR] Waypoints file not found at: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Safely transition robot arm to a recorded waypoint.")
    parser.add_argument(
        "--pose",
        type=str,
        default="tray_pose",
        help="Target waypoint name (default: tray_pose). Available: home_pose, tray_pose, camera_start_pose, handover_pose.",
    )
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP address (auto-detected if omitted)")
    parser.add_argument("--speed", type=float, default=0.3, help="Joint speed in rad/s (default: 0.3 rad/s for safety)")
    parser.add_argument("--accel", type=float, default=0.4, help="Joint acceleration in rad/s^2 (default: 0.4)")
    parser.add_argument("--offset-x", type=float, default=0.0, help="Cartesian X offset in meters (using poseTrans)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Cartesian Y offset in meters (using poseTrans)")
    parser.add_argument("--offset-z", type=float, default=0.0, help="Cartesian Z offset in meters (using poseTrans)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--sim", action="store_true", help="Execute motion in MuJoCo physics simulation")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode without 3D GUI window")
    args = parser.parse_args()

    waypoints = load_waypoints(WAYPOINTS_FILE)
    if args.pose not in waypoints:
        print(f"[ERROR] '{args.pose}' not found in waypoints.json.")
        print(f"Available waypoints: {list(waypoints.keys())}")
        sys.exit(1)

    target_data = waypoints[args.pose]
    target_joints = target_data["joints_rad"]
    target_deg = target_data.get("joints_deg", [round(x * 57.2958, 1) for x in target_joints])
    target_tcp = target_data.get("tcp_pose", [])

    robot_ip = resolve_robot_ip(args.robot_ip) if not args.sim else "MuJoCo Simulation"
    print("=" * 60)
    print(f" Target Waypoint : {args.pose}")
    print(f" Target Joints   : {target_joints}")
    print(f" Target Degrees  : {target_deg}")
    if target_tcp:
        print(f" Target TCP Pose : {target_tcp}")
    print(f" Robot IP / Sim  : {robot_ip}")
    print(f" Speed / Accel   : {args.speed} rad/s, {args.accel} rad/s²")
    print("=" * 60)

    if not args.yes:
        confirm = input(f"Move robot to '{args.pose}'? (y/N) > ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Motion cancelled.")
            sys.exit(0)

    print(f"Connecting to UR RTDE at {robot_ip}...")
    try:
        rtde_c, rtde_r = get_robot_interfaces(
            robot_ip=args.robot_ip,
            sim=args.sim,
            headless=args.headless if args.sim else None,
        )
    except Exception as e:
        print(f"[ERROR] Could not connect to robot interfaces: {e}")
        sys.exit(1)

    actual_q = rtde_r.getActualQ()
    print(f"Current Joint Angles: {[round(x, 4) for x in actual_q]}")
    print(f"Executing moveJ to '{args.pose}'...")

    try:
        # 1. Move to the base waypoint via moveJ
        success = rtde_c.moveJ(target_joints, speed=args.speed, acceleration=args.accel)
        if success:
            print(f"✓ Reached base waypoint '{args.pose}'")
        else:
            print(f"[WARN] moveJ returned False. Check robot status.")

        # 2. If any Cartesian offsets are requested, apply via poseTrans + moveL
        has_offset = any(abs(v) > 1e-4 for v in [args.offset_x, args.offset_y, args.offset_z])
        if success and has_offset:
            current_tcp = rtde_r.getActualTCPPose()
            offset = [args.offset_x, args.offset_y, args.offset_z, 0.0, 0.0, 0.0]
            target_trans = rtde_c.poseTrans(current_tcp, offset)
            print(f"Applying poseTrans offset {offset[:3]} m...")
            print(f"Target Cartesian TCP: {[round(x, 4) for x in target_trans]}")
            move_l_ok = rtde_c.moveL(target_trans, speed=0.05, acceleration=0.1)
            if move_l_ok:
                print(f"✓ Reached offset pose!")
            else:
                print(f"[WARN] moveL returned False (target may be unreachable).")
    except Exception as e:
        print(f"[ERROR] Motion failed: {e}")
    finally:
        rtde_c.stopScript()
        rtde_r.disconnect()
        print("Disconnected RTDE.")


if __name__ == "__main__":
    main()
