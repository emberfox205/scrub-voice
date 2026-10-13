#!/usr/bin/env python3
"""
Interactive MuJoCo Console for Universal Robots UR-10e & OnRobot RG-2 Gripper.

Provides an interactive console to test arm motion to tray pose,
home pose, gripper open/close, and query robot telemetry with live 3D visualization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local imports work dynamically
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.sim_ws.mujoco_controller import MujocoSimController


def print_banner():
    """Prints clean interactive console menu."""
    banner = """
============================================================
       MuJoCo UR-10e & RG-2 Gripper Simulation Console
============================================================
 Available Commands:
   tray   - Smoothly move arm to surgical tray pose
   home   - Smoothly return arm to home standby pose
   open   - Open OnRobot RG-2 gripper
   close  - Close OnRobot RG-2 gripper
   status - Print current joint positions and TCP coordinates
   exit   - Close 3D viewer and exit console
============================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo Simulation Interactive Console for UR-10e and RG-2"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without graphical 3D viewer window (headless mode)",
    )
    args = parser.parse_args()

    # Initialize simulation controller
    sim = MujocoSimController(headless=args.headless)
    if not args.headless:
        sim.start_viewer()
    else:
        sim.start()

    print_banner()

    try:
        while sim.is_running():
            try:
                cmd = input("sim> ").strip()
            except EOFError:
                break

            if not cmd:
                continue

            cmd_lower = cmd.lower()

            if cmd_lower in ("exit", "quit", "q"):
                print("Exiting simulation...")
                break

            elif cmd_lower == "tray":
                sim.move_to_tray(duration=2.5)
                print(f"✓ Reached Tray Pose: {[round(d, 1) for d in sim.get_actual_q_deg()]}")

            elif cmd_lower == "home":
                sim.move_to_home(duration=2.5)
                print(f"✓ Reached Home Pose: {[round(d, 1) for d in sim.get_actual_q_deg()]}")

            elif cmd_lower == "open":
                sim.open_gripper(duration=1.0)
                print("✓ RG-2 Gripper Opened.")

            elif cmd_lower == "close":
                sim.close_gripper(duration=1.0)
                print("✓ RG-2 Gripper Closed.")

            elif cmd_lower == "status":
                q_deg = sim.get_actual_q_deg()
                q_rad = sim.get_actual_q()
                tcp = sim.get_actual_tcp_pose()
                print("--------------------------------------------------")
                print(f"Joint Angles (deg): {q_deg}")
                print(f"Joint Angles (rad): {[round(x, 4) for x in q_rad]}")
                print(f"TCP Pose [X,Y,Z,Rx,Ry,Rz]: {tcp}")
                print("--------------------------------------------------")

            else:
                print(f"Unknown command: '{cmd}'. Type: 'tray', 'home', 'open', 'close', 'status', or 'exit'.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing simulation...")
    finally:
        sim.close_viewer()
        print("Simulation closed successfully.")


if __name__ == "__main__":
    main()
