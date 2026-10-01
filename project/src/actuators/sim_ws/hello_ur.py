import os
import socket
import time
import sys
import rtde_control
import rtde_receive

# Accepts IP from CLI argument, environment variable, or defaults to 127.0.0.1 (local)
ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ROBOT_IP", "127.0.0.1")
DASHBOARD_PORT = 29999

def ensure_robot_running(ip=ROBOT_IP, port=DASHBOARD_PORT):
    """Ensures the robot is powered on and brakes are released via Dashboard server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((ip, port))
            s.recv(1024)  # welcome message
            
            def send(cmd):
                s.sendall((cmd + "\n").encode())
                return s.recv(1024).decode().strip()

            mode = send("robotmode")
            print(f"Initial Robot Status: {mode}")

            if "RUNNING" not in mode:
                print("Powering ON robot motors...")
                send("power on")
                time.sleep(2)
                
                print("Releasing virtual brakes...")
                send("brake release")
                
                # Wait up to 15 seconds for RUNNING state
                for _ in range(15):
                    time.sleep(1)
                    mode = send("robotmode")
                    if "RUNNING" in mode:
                        print("Robot is now RUNNING and ready for motion!")
                        return True
            else:
                return True
    except Exception as e:
        print(f"Notice: Dashboard check skipped ({e}). Proceeding to RTDE...")
    return False

def main():
    print(f"Connecting to Universal Robots / URSim at {ROBOT_IP}...")
    
    # 1. Ensure robot is powered and brakes released
    ensure_robot_running(ROBOT_IP)

    # 2. Connect to RTDE interfaces
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    except Exception as e:
        print(f"\n[ERROR] Could not connect to UR RTDE at {ROBOT_IP}:30004.")
        print(f"Details: {e}")
        print("Please check: http://localhost:6080/vnc.html to confirm robot status.\n")
        sys.exit(1)

    # 3. Query current robot state
    q = rtde_r.getActualQ()
    tcp = rtde_r.getActualTCPPose()
    print("--------------------------------------------------")
    print("Connected successfully!")
    print(f"Current Joint Angles (deg): {[round(val * 57.2958, 1) for val in q]}")
    print(f"Current TCP Pose [X,Y,Z] (m): {[round(val, 3) for val in tcp[:3]]}")
    print("--------------------------------------------------")

    # 4. Define key joint waypoints (radians): [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    # Keeps Joint 5 (Wrist 2) bent at -90 deg (-1.57 rad) to avoid singularities
    home_pose = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    handover_pose = [0.6, -1.20, 1.20, -1.57, -1.57, 0.0]

    print("Executing: Moving to Home Pose...")
    rtde_c.moveJ(home_pose, speed=0.8, acceleration=0.8)
    time.sleep(1)

    print("Executing: Moving to Sterile Handover Zone...")
    rtde_c.moveJ(handover_pose, speed=0.8, acceleration=0.8)
    time.sleep(1)

    print("Executing: Returning to Home Pose...")
    rtde_c.moveJ(home_pose, speed=0.8, acceleration=0.8)
    time.sleep(0.5)

    rtde_c.stopScript()
    print("--------------------------------------------------")
    print("Motion test completed successfully!")

if __name__ == "__main__":
    main()
