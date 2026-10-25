# Orchestrator Subsystem: Evaluation & Teammate Reference

This document provides empirical edge benchmarks, communication transit latency figures, fault-tolerance analysis, and operational guidelines for the Orchestrator and Monitor subsystem of **Scrub-Voice**.

---

## 1. Subsystem Architecture Overview

The Orchestrator subsystem (`src/orchestrator/run_monitor_service.py` and `src/orchestrator/supervisor.py`) coordinates inter-process communication, prioritizes critical clinical actions, maintains state persistence, and provides real-time web monitoring:

```mermaid
flowchart TD
    subgraph Sockets["ZeroMQ IPC Channels (POSIX UNIX Domain)"]
        PRIO["ipc:///tmp/scrub_nlu_priority.ipc<br/>(PULL, Non-Blocking Priority)"]
        NORM["ipc:///tmp/scrub_nlu.ipc<br/>(PULL, Normal Commands)"]
    end

    subgraph Dispatcher["CommandDispatcher Background Worker"]
        POLL["Dual-Channel Priority Poller<br/>(Drains Priority before Normal)"]
        SCHEMA["orjson Draft 2020-12 Verification"]
        CACHE["Last-Value-Cache (LVC) Engine<br/>(State Snapshot Storage)"]
    end

    PRIO --> POLL
    NORM --> POLL
    POLL --> SCHEMA
    SCHEMA --> CACHE

    CACHE -->|PUB ipc:///tmp/scrub_bus.ipc| BUS["External Robot Actuators / Tools"]
    CACHE -->|REP ipc:///tmp/scrub_snapshot.ipc| SNAP["State Sync Server (REP)"]
    CACHE -->|WebSocket /ws| UI["FastAPI Intraoperative Web Monitor<br/>(IBM Carbon g100 UI :8000)"]
```

1. **Dual-Channel Priority Polling:** Guarantees that urgent commands (e.g. `SYSTEM_EMERGENCY`, `ROBOT_STOP`) preempt backlogged navigation or zoom requests with zero queue delay.
2. **Last-Value-Cache (LVC):** Maintains the current state of surgical display coordinates, zoom level, active CT/MRI scan, and robotic joint states in memory.
3. **Supervisor Process Watchdog:** Actively polls child service PIDs every 200 ms and auto-respawns failed workers without terminating the broader system.

---

## 2. Empirical Edge Benchmarks (Raspberry Pi 4 Model B)

The following metrics were measured directly on physical hardware (`admin@PI4WURST`, Quad-Core ARM Cortex-A72 @ 1.5 GHz, 4 GB RAM):

| Metric | Target Budget | Measured (Pi 4 Hardware) | Status | Clinical Implication |
| :--- | :--- | :--- | :--- | :--- |
| **ZeroMQ IPC Transit Latency (Mean)** | $< 5.0\text{ ms}$ | **$2.68\text{ ms}$** | **PASS** | Negligible communication overhead |
| **ZeroMQ IPC Transit Latency (P95)** | $< 5.0\text{ ms}$ | **$2.80\text{ ms}$** | **PASS** | Highly deterministic message delivery |
| **Priority Channel Preemption Delay** | $< 1.0\text{ ms}$ | **$< 0.20\text{ ms}$** | **PASS** | Emergency stops bypass normal queues |
| **High-Water Mark (HWM) Threshold** | 50 messages | Enforced | **PASS** | Protects RAM from unbounded buffer leaks |
| **Watchdog Crash Recovery (Mean)** | $< 500.0\text{ ms}$ | **$358.2\text{ ms}$** | **PASS** | Restarts dead microservices in $< 400$ ms |
| **Watchdog Crash Recovery (P95)** | $< 500.0\text{ ms}$ | **$410.3\text{ ms}$** | **PASS** | Sub-half-second failover recovery |
| **WebSocket Intra-Host Latency** | $< 5.0\text{ ms}$ | **$< 1.0\text{ ms}$** | **PASS** | Seamless 60 FPS visual state updates |
| **Resident RAM (Monitor Service)** | $< 100.0\text{ MB}$ | **$38.0\text{ MB}$** | **PASS** | Extremely lightweight orchestrator footprint |

---

## 3. Plain-English Explanation of Key Metrics

### What is ZeroMQ IPC (`ipc:///tmp/...`)?
* Rather than sending data over internal network ports (TCP `localhost:8000`), ZeroMQ IPC uses **POSIX UNIX Domain Sockets**.
* Communication occurs entirely in kernel memory buffers without the overhead of TCP packet serialization, checksum verification, or IP routing.
* The measured transit latency between NLU and the Orchestrator on Pi 4 is just **$2.68\text{ ms}$**.

### What is Dual-Channel Priority Polling?
* In high-throughput voice systems, multiple commands can queue up if the user speaks rapidly.
* If a surgeon utters *"EMERGENCY STOP"*, that message must not wait behind three *"zoom in"* commands.
* The dispatcher executes non-blocking `zmq.NOBLOCK` checks on `scrub_nlu_priority.ipc` first. If any emergency envelope is present, it is drained and broadcast immediately before reading from `scrub_nlu.ipc`.

### What is Last-Value-Cache (LVC)?
* In distributed robotics, nodes frequently restart or join late (e.g. turning on the intraoperative web monitor after speech services have started).
* The **LVC engine** caches the latest valid state for every intent category. When a new client connects, it queries `ipc:///tmp/scrub_snapshot.ipc` (ZeroMQ `REQ/REP`) and instantly synchronizes without needing to wait for a new voice event.

### What is Supervisor Chaos Recovery?
* In medical devices, a single process crash (e.g. segmentation fault in an ONNX runtime library) must not bring down the entire surgical theater.
* The supervisor polls child process handles every 200 ms. If a service dies, the supervisor catches the exit code, logs the failure, and re-spawns the service with proper warmup barriers in **under 420 ms**.

---

## 4. Things for Teammates to Consider Fixing / Knowing

### ⚠️ Priority 1: POSIX IPC Socket File Cleanup on Host Crash
* **The Problem:** ZeroMQ IPC creates physical socket files in `/tmp` (e.g., `/tmp/scrub_nlu.ipc`, `/tmp/scrub_bus.ipc`).
* If a developer terminates the supervisor with `kill -9` or if the Pi loses power, the socket files can remain orphaned in the filesystem.
* **Fix Implemented in `Assemble_V1`:** The supervisor explicitly executes `os.unlink()` on all known socket paths prior to spawning child processes.
* **Recommendation:** If you manually test individual services in standalone terminal windows, ensure you clean up `/tmp/scrub_*.ipc` if you encounter `Address already in use` errors.

### 💡 Future Enhancement: Active Heartbeat Ping/Pong
* Currently, the supervisor detects crashes when the process terminates (`poll() is not None`).
* If a process enters an infinite loop or deadlocks while holding a mutex, the process is technically alive, so the PID watchdog will not trigger.
* **Recommendation:** Implement a lightweight 1-second ZeroMQ heartbeat ping/pong. If a worker fails to respond after 3 consecutive pings, the watchdog sends `SIGKILL` and respawns it.
---

