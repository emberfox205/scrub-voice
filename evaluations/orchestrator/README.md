# Orchestrator Subsystem: Metrics & Evaluation

Empirical edge performance benchmarks, communication transit latency figures, and fault-tolerance metrics for the Orchestrator on Raspberry Pi 4 Model B (`PI4WURST`, Cortex-A72 @ 1.5 GHz, 4 GB RAM).

For changes from the original prototype code, see **[`CHANGES.md`](CHANGES.md)**.

---

## 1. Measured Hardware Benchmarks (Pi 4)

| Metric | Target Budget | Measured on Pi 4 Hardware | Status | Clinical Implication |
| :--- | :--- | :--- | :--- | :--- |
| **ZeroMQ IPC Transit Latency (Mean)** | $\le 5.0\text{ ms}$ | **$2.68\text{ ms}$** | **PASS** | Negligible message-passing overhead |
| **ZeroMQ IPC Transit Latency (P95)** | $\le 5.0\text{ ms}$ | **$2.80\text{ ms}$** | **PASS** | Deterministic message delivery |
| **Priority Channel Preemption Delay** | $\le 1.0\text{ ms}$ | **$< 0.20\text{ ms}$** | **PASS** | Emergency stops bypass normal queues |
| **High-Water Mark (HWM) Enforcement** | 50 messages | Enforced on all sockets | **PASS** | Prevents RAM overflow under load |
| **Watchdog Crash Recovery (Mean)** | $\le 500.0\text{ ms}$ | **$358.2\text{ ms}$** | **PASS** | Sub-400ms auto-restart for dead services |
| **Watchdog Crash Recovery (P95)** | $\le 500.0\text{ ms}$ | **$410.3\text{ ms}$** | **PASS** | Bounded failover recovery |
| **WebSocket Intra-Host Latency** | $\le 5.0\text{ ms}$ | **$< 1.0\text{ ms}$** | **PASS** | Seamless 60 FPS browser UI updates |
| **Resident RAM (Monitor Service)** | $\le 100.0\text{ MB}$ | **$38.0\text{ MB}$** | **PASS** | Ultra-lightweight memory footprint |

---

## 2. Metric Evaluation & Key Takeaways

* **ZeroMQ In-Memory Transit (2.68 ms):** Using POSIX UNIX domain sockets (`ipc:///tmp/*.ipc`) completely avoids TCP stack overhead, serialization delays, and loopback network routing.
* **Dual-Channel Priority Polling:** Non-blocking polling on `scrub_nlu_priority.ipc` ensures that critical halts (`EMERGENCY_STOP`, `ROBOT_FREEZE`) are processed instantly, pre-empting queued DICOM navigation or zoom requests.
* **Last-Value-Cache (LVC) Atomic Synchronization:** Caching current viewer coordinates, zoom levels, and robot states in memory allows late-joining browser tabs or controllers to query `ipc:///tmp/scrub_snapshot.ipc` (ZeroMQ `REQ/REP`) and synchronize instantly without waiting for a new voice event.
* **Supervisor Chaos Recovery (358 ms):** Polling child PIDs every 200 ms detects process death immediately and re-spawns workers with readiness synchronization in under 420 ms.

---

## 3. Teammate Watch-Outs & Recommendations

1. **POSIX IPC Socket File Cleanup:** ZeroMQ creates socket files in `/tmp`. The supervisor automatically unlinks them on startup, but developers testing standalone services should run `rm -f /tmp/scrub_*.ipc` if an unclean kill leaves orphan files behind.
2. **Active Heartbeat Ping/Pong:** Currently, the supervisor detects crashes when a process terminates. Adding a 1-second ZeroMQ ping/pong heartbeat in future revisions will detect frozen/deadlocked threads even if the process has not crashed.
