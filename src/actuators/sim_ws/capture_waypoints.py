#!/usr/bin/env python3
"""
Waypoint Capture Utility for Universal Robots (UR5 / UR10e / URSim)

Captures live joint positions (radians & degrees) and Cartesian TCP pose
via RTDE (Port 30004), logging them into a shared waypoints.json configuration.
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
    print("[ERROR] rtde_receive is not installed. Please activate the 'ur_arm' environment.")
    sys.exit(1)


DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent.parent / "waypoints.json"
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

    for ip in CANDIDATE_IPS:
        if not ip:
            continue
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((ip, 30004)) == 0:
                    return ip
        except Exception:
            pass

    return "192.168.1.105"


def record_waypoint(rtde_r, label: str) -> dict:
    """Queries RTDE for current joint angles and TCP pose."""
    q_rad = [round(float(x), 4) for x in rtde_r.getActualQ()]
    q_deg = [round(float(x) * 57.2957795, 2) for x in rtde_r.getActualQ()]
    tcp = [round(float(x), 4) for x in rtde_r.getActualTCPPose()]

    data = {
        "joints_rad": q_rad,
        "joints_deg": q_deg,
        "tcp_pose": tcp,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return data


def save_waypoints(file_path: Path, waypoints: dict) -> None:
    """Saves waypoints dictionary to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(waypoints, f, indent=2)
    print(f"✓ Saved to {file_path}")


def load_waypoints(file_path: Path) -> dict:
    """Loads existing waypoints if present."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def print_waypoint_summary(label: str, data: dict) -> None:
    """Pretty prints waypoint details."""
    print("=" * 60)
    print(f" Waypoint: {label}")
    print("=" * 60)
    print(f" Joint Angles (rad) : {data['joints_rad']}")
    print(f" Joint Angles (deg) : {data['joints_deg']}")
    print(f" TCP Pose [X,Y,Z,Rx,Ry,Rz]: {data['tcp_pose']}")
    print(f" Timestamp          : {data['updated_at']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Capture live UR waypoints via RTDE.")
    parser.add_argument("--robot-ip", type=str, default=None, help="IP address of UR robot/URSim")
    parser.add_argument("--label", type=str, default="home_pose", help="Label for current pose (e.g. home_pose, handover_pose)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to output JSON")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive loop to capture multiple poses")
    args = parser.parse_args()

    robot_ip = resolve_robot_ip(args.robot_ip)
    print(f"Connecting to UR RTDE at {robot_ip}:30004...")

    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    except Exception as e:
        print(f"[ERROR] Failed to connect to {robot_ip}:30004 -> {e}")
        sys.exit(1)

    print("Connected successfully!\n")
    waypoints = load_waypoints(args.output)

    if args.interactive:
        print("Interactive Waypoint Capture Mode. Enter pose label or 'q' to quit.")
        try:
            while True:
                label = input("\nEnter waypoint label (e.g., home_pose, handover_pose, tray_pose) > ").strip()
                if not label:
                    continue
                if label.lower() in ("q", "quit", "exit"):
                    break
                data = record_waypoint(rtde_r, label)
                waypoints[label] = data
                print_waypoint_summary(label, data)
                save_waypoints(args.output, waypoints)
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
    else:
        label = args.label
        data = record_waypoint(rtde_r, label)
        waypoints[label] = data
        print_waypoint_summary(label, data)
        save_waypoints(args.output, waypoints)

    rtde_r.disconnect()
    print("RTDE disconnected.")


if __name__ == "__main__":
    main()
