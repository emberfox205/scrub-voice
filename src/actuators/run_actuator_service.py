#!/usr/bin/env python3
"""
Actuator Service Daemon CLI for Scrub-Voice.

Starts the persistent ActuatorService, connecting to the UR10e and RG-2 interfaces
(physical hardware or MuJoCo simulation) and listening for commands on the ZeroMQ bus.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.actuators.service import DEFAULT_BUS_ADDR, ActuatorService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ActuatorService] %(message)s",
)
logger = logging.getLogger("actuators.service_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub-Voice Actuator Service Daemon")
    parser.add_argument("--bus-addr", default=DEFAULT_BUS_ADDR, help="Command bus ZMQ URI")
    parser.add_argument("--sim", action="store_true", help="Run with simulated robot/gripper HAL")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode (no MuJoCo GUI)")
    parser.add_argument("--robot-ip", default=None, help="Physical UR10e robot IP address")
    parser.add_argument("--gripper-ip", default=None, help="Physical RG-2 Compute Box IP address")
    parser.add_argument("--waypoints", default=None, help="Path to waypoints.json")
    args = parser.parse_args()

    logger.info("Initializing ActuatorService (sim=%s, headless=%s)...", args.sim, args.headless)

    service = ActuatorService(
        bus_addr=args.bus_addr,
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        sim=args.sim,
        headless=args.headless,
        waypoints_path=Path(args.waypoints) if args.waypoints else None,
    )

    stop_event = threading.Event()

    def handle_sig(sig, frame):
        logger.info("Received termination signal %s, stopping ActuatorService...", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        service.start()
        # Ready pattern expected by Supervisor
        print("Actuator Service active and connected to command bus", flush=True)
        logger.info("Actuator Service active and connected to command bus at %s", args.bus_addr)

        while not stop_event.is_set():
            time.sleep(0.5)
            if not service.is_alive():
                logger.error("ActuatorService worker thread died unexpectedly.")
                break
    finally:
        service.stop()
        logger.info("ActuatorService cleanly terminated.")


if __name__ == "__main__":
    main()
