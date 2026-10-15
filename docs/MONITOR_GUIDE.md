# Intraoperative Monitor Subsystem Guide

This document describes the architecture, ZeroMQ integration, WebSocket streaming gateway, modular intent handlers, and offline clinical dashboard for the `scrub-voice` surgical assistant.

---

## 1. Subsystem Architecture

The intraoperative monitor is an offline, edge-capable visualization and audit dashboard for surgical suites (e.g. Raspberry Pi 4 and clinical workstations). It operates in real time without external internet access or CDN dependencies, adhering to the **IBM Carbon Design System (g100 Dark Theme)** and **OHIF DICOM standards**.

```text
[ Speech Audio ] ──────> [ NLU Parser ]
                               │ (ZeroMQ PULL/PUSH)
                               ▼
                    [ Central Orchestrator ]
                     ├── LVC State Cache
                     ├── ZMQ REP (Snapshot Server: ipc:///tmp/scrub_snapshot.ipc)
                     └── ZMQ PUB (Command Bus: ipc:///tmp/scrub_bus.ipc)
                               │
            ┌──────────────────┴──────────────────┐
            │ ZMQ SUB (Topic Bus)                 │ ZMQ REQ (State Snapshot)
            ▼                                     ▼
   [ Monitor Bridge ] ◄───────────────────────────┘
            │
            ▼ (Async Broadcast)
   [ FastAPI WebSocket Gateway ] (/ws, port 8000)
            │
            ▼ (JSON WebSocket Stream)
   [ Offline Dark-Mode Web UI ] (IBM Carbon g100 + OHIF Viewport + Real Scans)
```

---

## 2. Directory Layout

All implementation resides within `src/monitor/` in compliance with repository guidelines:

```text
src/monitor/
├── __init__.py
├── bridge.py                 # ZeroMQ SUB listener + REQ snapshot synchronizer
├── handlers/                 # Modular intent handlers
│   ├── __init__.py           # Exports handler registry and imports handlers
│   ├── emergency_handlers.py # halt_emergency, clear_emergency
│   ├── event_handlers.py     # log_event (EMR milestone auditing)
│   ├── registry.py           # @register_intent decorator and handle_command
│   ├── scan_handlers.py      # open_scan, close_scan, zoom_display, navigate_slice
│   └── tool_handlers.py      # pick_tool, park_arm
├── server.py                 # FastAPI HTTP + WebSocket application gateway
├── state.py                  # MonitorState and RobotTelemetry dataclasses
└── static/                   # 100% offline static assets (zero CDNs)
    ├── app.js                # WebSocket client & OHIF scan viewport renderer
    ├── index.html            # Carbon g100 flush grid layout
    ├── style.css             # 1px hairline dividers (#393939), IBM Plex Sans
    └── scans/                # Open-source CC-licensed clinical scans
        ├── ct.jpg            # 512×512 axial head CT slice
        ├── mri.jpg           # 1080×1080 axial brain MRI T2 slice
        └── xray.jpg          # Clinical PA chest radiograph
```

---

## 3. Modular Intent Handlers

When the orchestrator publishes validated commands over the bus, `handle_command` dispatches to modular registered handlers:

| Intent | Category | Required Slots | Optional Slots | State Mutation Effect |
| :--- | :--- | :--- | :--- | :--- |
| **`open_scan`** | `MONITOR_DISPLAY` | `scan_type` (`ct`, `mri`, `xray`) | *(none)* | Sets `active_scan`; resets `current_slice=1` and `zoom_factor=1.0`. |
| **`close_scan`** | `MONITOR_DISPLAY` | *(none)* | `scan_type` (`ct`, `mri`, `xray`) | Closes matching or active scan; resets viewport to standby. |
| **`zoom_display`** | `MONITOR_DISPLAY` | *(none)* | `direction` (`IN`, `OUT`), `factor` ($1.0-10.0$) | Scales `zoom_factor` clamped between $1.0\times$ and $10.0\times$. |
| **`navigate_slice`**| `MONITOR_DISPLAY` | `slice_index` or `direction` | `step_count` ($\ge 1$) | Updates `current_slice` (relative or direct); clamped to $\ge 1$. |
| **`log_event`** | `MONITOR_DISPLAY` | `event_name` (string) | `comment` (string) | Appends milestone to `recent_milestones` (FIFO max 50). |
| **`pick_tool`** | `TOOL_HANDLING` | `tool_identifier` (`scalpel`, etc.) | *(none)* | Sets `held_tool`, `telemetry.gripper_status`, and `telemetry.status`. |
| **`park_arm`** | `TOOL_HANDLING` | *(none)* | *(none)* | Clears `held_tool` to `None`; sets gripper to standby. |
| **`halt_emergency`**| `EMERGENCY` | *(none)* | *(none)* | Sets `emergency_active=True`, brakes locked status. |
| **`clear_emergency`**| `EMERGENCY` | *(none)* | *(none)* | Sets `emergency_active=False`, initiates $+Z$ cavity ascent. |

---

## 4. State Snapshot & HTTP/WebSocket Endpoints

* **Last Value Cache (LVC)**: Orchestrator maintains in-memory state and exposes a ZeroMQ `REP` socket on `ipc:///tmp/scrub_snapshot.ipc`. When `MonitorBridge` starts, it requests a `SNAPSHOT` via `REQ` to instantly hydrate `MonitorState`.
* **Endpoints**:
  * `GET /`: Serves Carbon g100 surgical monitor dashboard.
  * `WS /ws`: Full-duplex JSON stream broadcasting state updates and speech commands.
  * `GET /api/state`: Returns current intraoperative state snapshot JSON.
  * `GET /api/health`: Health status probe for process supervisors.
  * `GET /static/scans/{scan}`: Serves clinical scan imagery.

---

## 5. Running the Monitor and Automated Tests

### Start Central Orchestrator:
```bash
python -m src.orchestrator.dispatcher
```

### Start Monitor Gateway:
```bash
python -m src.monitor.server --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in any browser.

### Run Streamlined Automated Tests:
```bash
pytest test/test_monitor_handlers.py test/test_monitor_pipeline.py test/test_orchestrator.py
```
