# Implementation Plan: Assemble_V1 (Supervised Voice-to-Monitor Pipeline)

**Branch:** `Assemble_V1`  
**Target Platform:** Raspberry Pi 4 (ARM64) / Development Workstations (Windows / Linux)  
**Scope:** Offline Voice-to-Screen Surgical Display (Excludes Robotics)

---

## 1. Architectural Philosophy: Why Multi-Process Supervision?

In an operating room, software reliability is safety-critical. If an Automatic Speech Recognition (ASR) process experiences an out-of-memory error or the microphone encounters a hardware buffer glitch, **the surgical monitor display must not crash or freeze**.

Instead of a monolithic Python script that imports every module into a single process, `Assemble_V1` implements an **Erlang / ROS-style Supervised Multi-Process Architecture**:

```text
               ┌────────────────────────────────────────────────────────┐
               │          Process Supervisor & Watchdog Engine          │
               │  • Spawns worker PIDs  • Aggregates logs  • Auto-heals │
               └──────┬──────────────┬──────────────┬─────────────┬─────┘
                      │              │              │             │
        ┌─────────────▼────┐   ┌─────▼────────┐   ┌─▼────────┐  ┌─▼──────────────┐
        │ 1. DSP Audio     │   │ 2. ASR       │   │ 3. NLU   │  │ 4. Orchestrator│
        │    Service       │   │    Service   │   │  Service │  │    + Monitor   │
        └──────┬───────────┘   └─────▲──┬─────┘   └──▲──┬────┘  └─▲──────────────┘
               │ 0.5s audio chunks   │  │ Utterance  │  │ 10-key  │ (Browser UI)
               └─────────────────────┘  │ JSON       │  │ Command │ ws://localhost:8000
                                        └────────────┘  │ JSON    │
                                                        └─────────┘
```

### Core Invariants:
1. **Process Isolation:** Each subsystem (DSP, ASR, NLU, Orchestrator+Monitor) runs in its own dedicated Operating System process (`subprocess.Popen`).
2. **Decoupled Messaging (ZeroMQ):** Processes communicate strictly via ZeroMQ message sockets (IPC pipes on Linux, TCP on Windows). Senders and receivers do not depend on each other's lifecycle; if ASR dies, DSP buffers continue to queue in memory without throwing unhandled socket exceptions.
3. **Automatic Fault Recovery (Auto-Restart):** The Supervisor monitors child Process IDs (PIDs). If a worker crashes or exits with a non-zero exit code, the Supervisor restarts it in under 500 ms while the clinical display remains live.
4. **Clean Scope Isolation:** Physical robot arm controls ([`src/actuators/`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/actuators)) are intentionally bypassed on this branch; the Orchestrator routes exclusively to the Intraoperative Monitor UI.

---

## 2. Subsystems & Data Contracts

```text
[DSP Audio] ──(ZMQ PUSH/PULL: 16kHz Chunks)──▶ [ASR] ──(ZMQ PUSH/PULL: Text)──▶ [NLU]
                                                                                   │
                                                                     (ZMQ Dual PUSH/PULL)
                                                                                   ▼
                                                                           [Orchestrator]
                                                                                   │
                                                                   (ZMQ PUB/SUB + Snapshot)
                                                                                   ▼
                                                                           [Monitor Gateway]
                                                                                   │ (WebSockets)
                                                                                   ▼
                                                                          [Browser Dashboard]
```

### Subsystem Breakdown:

| Subsystem | Service Script | Input | Output / Communication | Fault-Tolerance Policy |
| :--- | :--- | :--- | :--- | :--- |
| **1. DSP Front-End** | `src/dsp/run_dsp.py` (New Service Wrapper) | Multi-channel mic array (`arecord`) or offline test WAV stream | 0.5s mono float32 chunks (16 kHz) via `SCRUB_AUDIO_ADDR` | If sound card drops or buffer overruns (XRUN), restart hardware stream with 1.0s backoff. |
| **2. ASR (Speech-to-Text)** | [`src/asr/scripts/run_asr.py`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/asr/scripts/run_asr.py) | Audio chunks via ZeroMQ PULL | Transcribed text JSON via `SCRUB_ASR_ADDR` | If Vosk grammar decoder crashes, Supervisor restarts worker; ZeroMQ retains pending audio chunks. |
| **3. NLU (Intent & Slots)** | `src/nlu/run_nlu.py` (New Service Wrapper) | Transcribed text JSON via ZeroMQ PULL | Validated 10-key command JSON via `SCRUB_NLU_ADDR` / `_PRIORITY_ADDR` | Stateless worker; restarts instantly (< 200 ms) with zero state loss. |
| **4. Orchestrator & Monitor** | `src/orchestrator/run_monitor_service.py` (New Combined Gateway) | 10-key commands via ZeroMQ PULL | State snapshot (`REP`), Event bus (`PUB`), WebSocket stream (`/ws`) | **Must Never Crash.** Retains Last Value Cache (LVC) state. Automatically serves snapshots upon UI reconnection. |

---

## 3. Communication Network & Cross-Platform Addressing

To ensure 100% compatibility across both the target **Raspberry Pi 4 (Linux)** and developer workstations (**Windows**), socket addresses dynamically adapt based on `sys.platform`:

```python
import os, sys

IS_WINDOWS = sys.platform == "win32"

# Address Resolution Matrix
DEFAULT_AUDIO_ADDR = os.getenv("SCRUB_AUDIO_ADDR", "tcp://127.0.0.1:5554" if IS_WINDOWS else "ipc:///tmp/scrub_audio.ipc")
DEFAULT_ASR_ADDR   = os.getenv("SCRUB_ASR_ADDR",   "tcp://127.0.0.1:5555" if IS_WINDOWS else "ipc:///tmp/scrub_asr.ipc")
DEFAULT_NLU_ADDR   = os.getenv("SCRUB_NLU_ADDR",   "tcp://127.0.0.1:5556" if IS_WINDOWS else "ipc:///tmp/scrub_nlu.ipc")
DEFAULT_PRIO_ADDR  = os.getenv("SCRUB_PRIO_ADDR",  "tcp://127.0.0.1:5557" if IS_WINDOWS else "ipc:///tmp/scrub_nlu_prio.ipc")
DEFAULT_BUS_ADDR   = os.getenv("SCRUB_BUS_ADDR",   "tcp://127.0.0.1:5558" if IS_WINDOWS else "ipc:///tmp/scrub_bus.ipc")
DEFAULT_SNAP_ADDR  = os.getenv("SCRUB_SNAP_ADDR",  "tcp://127.0.0.1:5559" if IS_WINDOWS else "ipc:///tmp/scrub_snapshot.ipc")
```

---

## 4. Detailed Implementation Roadmap

### Phase 1: NLU Component Integration & Standardization
* **Objective:** Ensure the full NLU pipeline (lexical normalization, SetFit/rule intent classification, slot extraction, schema validation) is properly placed in [`src/nlu/`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu) in accordance with [`AGENTS.md`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/AGENTS.md).
* **Tasks:**
  1. Add [`src/nlu/normalizer.py`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/normalizer.py) (filler removal, numeric standardization).
  2. Add [`src/nlu/slot_extractor.py`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/slot_extractor.py) (entity parsing for tools, scans, slice numbers, zoom levels).
  3. Add [`src/nlu/dictionaries/`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/dictionaries) (`medical_slots.json`, `fillers.json`, `numbers.json`, `slot_constraints.json`).
  4. Create `src/nlu/run_nlu.py`: A standalone ZeroMQ daemon that listens on `SCRUB_ASR_ADDR` (PULL) and pushes validated envelopes to `SCRUB_NLU_ADDR` / `_PRIORITY_ADDR` (PUSH).

### Phase 2: DSP Service Packaging & Clean ONNX Integration
* **Objective:** Package the DSP front-end into an independent daemon using Quân's lightweight ONNX denoiser (avoiding heavy PyTorch/Rust libraries).
* **Tasks:**
  1. Verify [`src/dsp/source/audio_streamer.py`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/audio_streamer.py) (mic array) and add WAV simulation mode (for testing without a physical mic array).
  2. Connect Silero VAD ➔ SRP-PHAT Localizer ➔ DAS Beamformer ➔ Quân's DeepFilterNet ONNX ([`src/dsp/denoise/pi_inference.py`](file:///P:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/denoise/pi_inference.py)).
  3. Create `src/dsp/run_dsp.py`: Standalone service streaming 0.5s chunks over ZeroMQ `SCRUB_AUDIO_ADDR`.

### Phase 3: Combined Orchestrator & Monitor Service
* **Objective:** Ensure the central command dispatcher and FastAPI WebSocket monitor run reliably together.
* **Tasks:**
  1. Create `src/orchestrator/run_monitor_service.py` that starts the `CommandDispatcher` in a background thread and launches the `uvicorn` FastAPI web server (`http://0.0.0.0:8000`).
  2. Ensure the WebSocket gateway serves the offline IBM Carbon g100 dashboard and handles instant state synchronization on connect.

### Phase 4: Building the Supervisor Engine (`src/orchestrator/supervisor.py`)
* **Objective:** Implement the central process manager and watchdog.
* **Tasks:**
  1. **Process Registry:** Define process specs for `[DSP]`, `[ASR]`, `[NLU]`, and `[MONITOR]`.
  2. **Health Loop:** Poll worker exit statuses every 200 ms via `process.poll()`.
  3. **Auto-Restart Engine:** If a process terminates unexpectedly:
     * Log a critical error alert.
     * Apply exponential backoff (initial: 0.5s, max: 5s) to avoid restart thrashing.
     * Spawn a new instance of the failed worker with identical configuration.
  4. **Aggregated Logging:** Prefix stdout/stderr streams with color-coded tags (`[DSP]`, `[ASR]`, `[NLU]`, `[MONITOR]`) for centralized real-time debugging.
  5. **Clean Signal Handling:** Intercept `SIGINT` (Ctrl+C) and `SIGTERM` to gracefully shut down all children without leaving orphaned Python processes or hung sockets.

### Phase 5: Verification & Fault-Tolerance Chaos Testing
* **Objective:** Test end-to-end functionality and prove resilience against crashes.
* **Test Procedures:**
  1. **End-to-End Voice Navigation:** Speak or stream:
     * *"Open CT scan"* ➔ Monitor renders CT scan at slice 1.
     * *"Zoom in 2.5x"* ➔ Viewport zooms smoothly.
     * *"Advance 5 slices"* ➔ Slice index moves from 1 to 6.
     * *"Record milestone: trocar placed"* ➔ Event logged to EMR milestone ledger.
  2. **Chaos Testing (Process Crash & Recovery):**
     * Kill the ASR process (`kill -9 <PID>`).
     * Verify that the Supervisor detects the crash and revives ASR in < 500 ms.
     * Verify that the Monitor UI never disconnects or loses current slice/zoom state.

---

## 5. Acceptance Checklist for `Assemble_V1`

- [ ] All redundant code (`src/dsp/source/postfilter.py` and its tests) removed.
- [ ] No monolithic runner: each component runs as an independent OS process.
- [ ] Central Supervisor (`src/orchestrator/supervisor.py`) launches, monitors, and auto-restarts workers.
- [ ] ZeroMQ sockets automatically reconnect after worker restarts without throwing exceptions.
- [ ] Full touchless monitor control verified: scan opening, slice jumping, zooming, and milestone logging.
- [ ] Out-of-domain conversational chatter is safely rejected and never alters monitor state.
- [ ] Works cleanly on both Windows (dev workstations) and Raspberry Pi 4 (production).
