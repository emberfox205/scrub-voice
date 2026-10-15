import os
import sys
import time
import socket
import struct
import argparse
from pathlib import Path

# Dynamic sys.path setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default OnRobot Compute Box IP
DEFAULT_GRIPPER_IP = os.getenv("GRIPPER_IP", "192.168.1.101")
MODBUS_PORT = 502
MODBUS_UNIT_ID = 65  # OnRobot Compute Box Slave ID

class RG2Gripper:
    """
    Direct Modbus TCP controller for OnRobot RG2 gripper via Compute Box.
    Zero external dependencies (uses Python standard library socket + struct).
    """
    MAX_WIDTH_MM = 110.0
    MIN_WIDTH_MM = 0.0
    MAX_FORCE_N = 40.0
    MIN_FORCE_N = 3.0

    def __init__(self, ip: str = DEFAULT_GRIPPER_IP, port: int = MODBUS_PORT):
        self.ip = ip
        self.port = port
        self.trans_id = 0

    def _next_trans_id(self) -> int:
        self.trans_id = (self.trans_id + 1) % 65535
        return self.trans_id

    def _send_modbus(self, payload: bytes) -> bytes:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((self.ip, self.port))
            s.sendall(payload)
            return s.recv(1024)

    def get_state(self) -> dict:
        """
        Reads registers 267 (actual width) and 268 (device status).
        """
        tid = self._next_trans_id()
        # Func 0x03 (Read Holding Registers), Addr 267, Count 2
        req = struct.pack(">HHHBBHH", tid, 0, 6, MODBUS_UNIT_ID, 3, 267, 2)
        resp = self._send_modbus(req)

        if len(resp) < 13:
            raise IOError(f"Invalid Modbus response from {self.ip}: {resp.hex()}")

        width_raw, status_raw = struct.unpack(">HH", resp[9:13])
        return {
            "width_mm": round(width_raw / 10.0, 1),
            "status_raw": status_raw,
            "busy": bool(status_raw & 1),
            "grip_detected": bool(status_raw & 2),
            "safety_circuit_active": bool(status_raw & 8 or status_raw & 32),
        }

    def move(self, width_mm: float, force_n: float = 20.0, wait_for_idle: bool = True) -> dict:
        """
        Commands the RG2 to target width (mm) and force limit (N).
        """
        # Clamp to hardware limits
        target_width = max(self.MIN_WIDTH_MM, min(self.MAX_WIDTH_MM, float(width_mm)))
        target_force = max(self.MIN_FORCE_N, min(self.MAX_FORCE_N, float(force_n)))

        # Convert to 1/10 mm and 1/10 N
        width_reg = int(target_width * 10)
        force_reg = int(target_force * 10)
        cmd_reg = 16  # 0x0010: grip with fingertip offset (1 = without offset)

        tid = self._next_trans_id()
        # Func 0x10 (Write Multiple Registers): Start Addr 0, Count 3, Bytes 6
        header = struct.pack(">HHHBBHHB", tid, 0, 13, MODBUS_UNIT_ID, 16, 0, 3, 6)
        values = struct.pack(">HHH", force_reg, width_reg, cmd_reg)
        req = header + values

        resp = self._send_modbus(req)
        if len(resp) < 12:
            raise IOError(f"Failed to execute move on RG2: {resp.hex()}")

        print(f"Dispatched: Target Width={target_width} mm | Force Limit={target_force} N")

        if wait_for_idle:
            time.sleep(0.3)
            start_t = time.time()
            while time.time() - start_t < 6.0:
                state = self.get_state()
                if not state["busy"]:
                    return state
                time.sleep(0.1)

        return self.get_state()

    def open(self, force_n: float = 20.0):
        return self.move(self.MAX_WIDTH_MM, force_n=force_n)

    def close(self, force_n: float = 20.0):
        return self.move(self.MIN_WIDTH_MM, force_n=force_n)


class MujocoRG2Adapter:
    """Adapter wrapping MujocoSimController with the RG2Gripper interface."""

    def __init__(self, headless: bool = False):
        from src.actuators.sim_ws.mujoco_controller import MujocoSimController
        self.ip = "MuJoCo Simulation"
        self.port = 0
        self.sim = MujocoSimController(headless=headless)
        if not headless:
            self.sim.start_viewer()

    def get_state(self) -> dict:
        return self.sim.get_gripper_state()

    def move(self, width_mm: float, force_n: float = 20.0, wait_for_idle: bool = True) -> dict:
        return self.sim.move_gripper(width_mm, force_n=force_n, wait=wait_for_idle)

    def open(self, force_n: float = 20.0):
        return self.sim.open_gripper(force_n=force_n)

    def close(self, force_n: float = 20.0):
        return self.sim.close_gripper(force_n=force_n)


def run_test_cycle(gripper):
    print("==================================================")
    print(f"Connecting to RG-2 at {gripper.ip}:{gripper.port}")
    print("==================================================")

    init_state = gripper.get_state()
    print(f"Initial State -> Width: {init_state['width_mm']} mm | Busy: {init_state['busy']} | Grip Detected: {init_state['grip_detected']}")

    print("\n[Step 1] Opening gripper to 60.0 mm (Force limit 20 N)...")
    res = gripper.move(60.0, force_n=20.0)
    print(f"Reached -> Width: {res['width_mm']} mm | Grip: {res['grip_detected']}")
    time.sleep(1.0)

    print("\n[Step 2] Closing gripper to 20.0 mm (Force limit 15 N)...")
    res = gripper.move(20.0, force_n=15.0)
    print(f"Reached -> Width: {res['width_mm']} mm | Grip: {res['grip_detected']}")
    time.sleep(1.0)

    print("\n[Step 3] Reopening to neutral 50.0 mm...")
    res = gripper.move(50.0, force_n=20.0)
    print(f"Reached -> Width: {res['width_mm']} mm | Grip: {res['grip_detected']}")

    print("\n--------------------------------------------------")
    print("RG-2 test cycle finished successfully!")
    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Control OnRobot RG-2 Gripper via Compute Box Modbus TCP or MuJoCo Simulation")
    parser.add_argument("ip", nargs="?", default=DEFAULT_GRIPPER_IP, help=f"Gripper/Compute Box IP (default: {DEFAULT_GRIPPER_IP})")
    parser.add_argument("--width", type=float, default=None, help="Target gripper width in mm (0 - 110)")
    parser.add_argument("--force", type=float, default=20.0, help="Grip force limit in Newtons (3 - 40, default: 20)")
    parser.add_argument("--status", action="store_true", help="Query and print current gripper status")
    parser.add_argument("--test", action="store_true", help="Run automated test motion")
    parser.add_argument("--sim", action="store_true", help="Run in MuJoCo physics simulation mode")
    parser.add_argument("--headless", action="store_true", help="Run simulation without 3D GUI window")

    args = parser.parse_args()

    if args.sim:
        gripper = MujocoRG2Adapter(headless=args.headless)
    else:
        gripper = RG2Gripper(ip=args.ip)

    if args.status:
        state = gripper.get_state()
        print(f"RG-2 Status: Width = {state['width_mm']} mm | Busy = {state['busy']} | Grip = {state['grip_detected']}")
    elif args.test or args.width is None:
        run_test_cycle(gripper)
    else:
        state = gripper.move(args.width, force_n=args.force)
        print(f"Finished Move: Current Width = {state['width_mm']} mm")

if __name__ == "__main__":
    main()
