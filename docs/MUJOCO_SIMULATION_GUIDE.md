# MuJoCo Physics Simulation Guide — UR-10e & RG-2 Gripper

This guide provides instructions for running and testing the Google DeepMind **MuJoCo** simulation of the **Universal Robots UR-10e** arm equipped with the authentic CAD model of the **OnRobot RG-2** parallel gripper.

---

## 1. Overview & Architecture

The MuJoCo simulation environment provides high-fidelity physical simulation:
* **Authentic RG-2 CAD Model:** Exact STL meshes and kinematic 4-bar parallelogram linkage (6 revolute joints, equality mimic constraints) derived from OnRobot CAD specs (`ekorudiawan/rg2_simulation`).
* **Wall-Clock Synchronized Trajectories:** Smooth minimum-jerk S-curve cosine trajectories synchronized 1:1 with wall-clock time at 60 FPS in the 3D viewer (no teleportation or physics jumping).
* **Interactive Terminal Console:** Clean, streamlined command interface launched via `python -m src.actuators.sim_ws.mujoco_pipe_server`.

```text
src/actuators/
├── mujoco/
│   └── models/
│       ├── ur10e/
│       │   ├── assets/       # DeepMind Menagerie UR-10e meshes (.obj)
│       │   └── ur10e.xml     # Menagerie UR-10e kinematics & actuators
│       ├── rg2/
│       │   ├── assets/       # Authentic RG-2 STL meshes (hand, moment, finger, truss)
│       │   └── rg2.xml       # RG-2 4-bar linkage MJCF model & equality constraints
│       └── scene.xml         # Full surgical scene: table, tray, scalpel, UR-10e & RG-2
└── sim_ws/
    ├── mujoco_controller.py  # Simulation controller & wall-clock trajectory engine
    └── mujoco_pipe_server.py # Streamlined interactive terminal console & 3D viewer
```

---

## 2. Environment Setup

The simulation requires `mujoco>=3.1.0` in the dedicated Python environment.

### Conda Environment (`ur_arm`)
```bash
# Activate environment
conda activate ur_arm

# Verify MuJoCo installation
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

---

## 3. Running the Simulation

### Launch Interactive 3D Viewer & Console
To open the interactive 3D viewer window and start the interactive console:
```bash
python -m src.actuators.sim_ws.mujoco_pipe_server
```

### Available Interactive Commands
Once launched, the console displays:
```text
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
```

* **`tray`**: Executes a 2.5s smooth, wall-clock synchronized S-curve trajectory to the sterile surgical tray pose (`[-0.4026, -1.6247, 2.2264, -2.1839, -1.5986, -2.0028]`).
* **`home`**: Returns smoothly to the home standby pose (`[-0.0388, -1.5986, 1.5708, -1.571, -1.5708, 0.0]`).
* **`open`**: Smoothly drives the RG-2 4-bar linkage to the open position.
* **`close`**: Drives the RG-2 4-bar linkage to the closed position.
* **`status`**: Prints current 6-DoF joint angles (degrees and radians) and TCP Cartesian coordinates $[X, Y, Z, R_x, R_y, R_z]$.
* **`exit`** or **`quit`**: Gracefully shuts down the viewer and exits.

---

## 4. Headless Execution

To run headless (e.g. in automated testing, CI, or remote terminal without X11/Wayland display):
```bash
python -m src.actuators.sim_ws.mujoco_pipe_server --headless
```

---

## 5. Automated Verification

Run the test suite using `pytest`:
```bash
pytest test/test_mujoco_sim.py -v
```

This validates:
1. Model loading, geometry degrees of freedom (19 DoF), and actuator count (7).
2. Forward kinematics and TCP pose computation.
3. RG-2 linkage actuation (opening and closing stroke).
4. Smooth trajectory execution to tray and home poses.
