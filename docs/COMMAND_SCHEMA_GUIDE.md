# Voice-to-Intent Command Schema Guide

This document defines the interface contract between the **Natural Language Understanding (NLU)** pipeline and the downstream physical/UI actuators (**Universal Robots UR10e Controller** and **Medical Image Viewer UI**).

The definitive schema file is located at:
`schemas/command_schema.json`

---

## 1. Top-Level Command Envelope

Every JSON payload produced by the NLP pipeline must contain these **10 mandatory keys**:

```json
{
  "command_id": "cmd_1002_001500_001",
  "timestamp": 1727800000.123,
  "latency_ms": 145.2,
  "raw_transcript": "hand me the scalpel",
  "intent": "pick_tool",
  "category": "TOOL_HANDLING",
  "slots": {
    "tool_identifier": "scalpel"
  },
  "confidence": 0.98,
  "priority": "NORMAL",
  "status": "EXECUTE"
}
```

### Key Definitions:
* **`command_id`** (`string`): Tracking ID matching pattern `^cmd_[a-zA-Z0-9_]+$` (Format: `cmd_<MMDD>_<HHMMSS>_<seq>`).
* **`timestamp`** (`number`): Unix epoch in seconds (float) when intent classification completed.
* **`latency_ms`** (`number`): Processing latency in milliseconds (float) from audio capture to intent classification completion.
* **`raw_transcript`** (`string`): The verbatim transcript from the ASR stage.
* **`intent`** (`string`): Canonical verb-first intent name (see Section 2).
* **`category`** (`string`): Subsystem category (`TOOL_HANDLING`, `EMERGENCY`, `CAMERA_LIGHT`, `MONITOR_DISPLAY`, `UNKNOWN`).
* **`slots`** (`object`): Intent-specific parameter dictionary.
* **`confidence`** (`number`): NLU confidence score between `0.0` and `1.0`.
* **`priority`** (`string`): `"LOW"`, `"NORMAL"`, or `"HIGH"` (`HIGH` is enforced for emergencies).
* **`status`** (`string`):
  * `"EXECUTE"`: Valid intent and slots ready for actuation.
  * `"OUT-OF-DOMAIN"`: Unrelated speech (e.g. conversational chatter).
  * `"AMBIGUOUS"`: Low confidence or missing mandatory parameters.

---

## 2. Intent Taxonomy, Slot Specifications & Default Behaviors

All intents follow a strict **verb-first** naming convention and are bound to 5 functional categories.

---

### Category 1: `TOOL_HANDLING` (Universal Robots UR10e Arm & Gripper)

| Intent | Spoken Examples | Required Slots | Optional Slots | Default Values & Omission Behavior | Subsystem Actuation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pick_tool`** | *"Hand scalpel"*, *"Pass the iris scissors"*, *"Give me the clamp"* | `tool_identifier`: `string`<br>Allowed: `["scalpel", "scissors", "hemostat", "forceps"]` | *(none)* | **No default**. If tool entity cannot be resolved, NLU flags `status: "AMBIGUOUS"`. Robot will not move. | Transitions UR10e to tool tray, detects $(X, Y, \theta)$ via 2D homography, clamps tool, and presents at handover waypoint. |
| **`park_arm`** | *"Park arm"*, *"Standby robot"*, *"Retract arm"* | *(none)* | *(none)* | Parameterless (`slots: {}` enforced by schema). | Safely moves arm in joint space (`moveJ`) to standby parking pose outside surgical field. |

---

### Category 2: `EMERGENCY` (Global Safety Interlock)

Emergency commands always enforce `priority: "HIGH"` and take immediate precedence over all queued operations.

| Intent | Spoken Examples | Required Slots | Optional Slots | Default Values & Omission Behavior | Subsystem Actuation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`clear_emergency`** | *"Emergency clear"*, *"Clear area"*, *"Back off"* | *(none)* | *(none)* | Parameterless (`slots: {}` enforced by schema). | Immediate high-speed Cartesian vertical ascent ($+Z$) to evacuate the surgical cavity to safe height. |
| **`halt_emergency`** | *"Emergency stop"*, *"Halt robot"*, *"Freeze"* | *(none)* | *(none)* | Parameterless (`slots: {}` enforced by schema). | Immediate hard stop via RTDE joint brake hold. All motion freezes instantaneously. |

---

### Category 3: `MONITOR_DISPLAY` (Medical Image Viewer UI & Surgical EMR Logging)

| Intent | Spoken Examples | Required Slots | Optional Slots | Default Values & Omission Behavior | Subsystem Actuation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`open_scan`** | *"Open CT scan"*, *"Show MRI"*, *"Display X-ray"* | `scan_type`: `string`<br>Allowed: `["ct", "mri", "xray"]` | *(none)* | **No default**. Modality must be specified. If omitted, NLU flags `status: "AMBIGUOUS"`. | Loads and renders requested DICOM imaging modality into the primary viewport. |
| **`close_scan`** | *"Close scan"*, *"Dismiss MRI"*, *"Close viewer"* | *(none)* | `scan_type`: `string`<br>Allowed: `["ct", "mri", "xray"]` | If `scan_type` is omitted (`slots: {}`), **closes the currently active/focused scan tab**. If specified, closes that exact modality tab. | Closes designated tab or active scan and resets viewport to blank/standby. |
| **`zoom_display`** | *"Zoom"*, *"Zoom in 2x"*, *"Zoom out"*, *"Magnify 3x"* | *(none)* | - `direction`: `string`<br>Allowed: `["IN", "OUT"]`<br>- `factor`: `number`<br>Range: `1.1` to `10.0` | - **`direction` omitted**: Defaults to **`"IN"`** (magnify).<br>- **`factor` omitted**: Defaults to **`2.0`** (2x zoom).<br>*Examples:*<br>• *"Zoom"* $\rightarrow$ `slots: {}` $\rightarrow$ zooms **IN** by **2.0x**.<br>• *"Zoom out"* $\rightarrow$ `slots: {"direction": "OUT"}` $\rightarrow$ zooms **OUT** by **2.0x**.<br>• *"Zoom 3x"* $\rightarrow$ `slots: {"factor": 3.0}` $\rightarrow$ zooms **IN** by **3.0x**. | Scales viewport display around image center point by the resolved factor. |
| **`navigate_slice`** | *"Next slice"*, *"Previous slice"*, *"Back 5 slices"*, *"Go to slice 45"* | *Either* `direction` *or* `slice_index` (`anyOf` constraint) | - `direction`: `string`<br>Allowed: `["NEXT", "PREVIOUS"]`<br>- `step_count`: `int` ($\ge 1$)<br>- `slice_index`: `int` ($\ge 1$) | - **Relative mode** (`direction` given): If `step_count` is omitted, defaults to **`1`** (advances or rewinds 1 slice).<br>- **Absolute mode** (`slice_index` given): Jumps directly to slice index; `direction` and `step_count` are omitted. | Steps forward/backward in the cross-sectional slice stack or jumps directly to slice number. |
| **`log_event`** | **Variant A (Event only):**<br>• *"Log event: first incision"*<br>• *"Record milestone: trocar insertion complete"*<br>• *"Log event: hepatic artery clamped"*<br>• *"Record event: biopsy taken"*<br><br>**Variant B (Event + Comment):**<br>• *"Log event: first incision, note patient vitals stable"*<br>• *"Record event: tumor resection complete, comment clean margins"* | `event_name`: `string`<br>(Min length: 1) | `comment`: `string` | - **`event_name`**: Strictly required. Identifies procedural milestone or action taken.<br>- **`comment`**: Optional clinical annotation. Defaults to omitted/none if no comment was spoken. | Manually records a timestamped surgical milestone and optional clinical notes into the intraoperative log/EMR on the monitor display. |

---

### Category 4: `CAMERA_LIGHT` (UR10e Endoscope / Light Wand Jogging)

| Intent | Spoken Examples | Required Slots | Optional Slots | Default Values & Omission Behavior | Subsystem Actuation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`move_camera`** | *"Camera left 5 cm"*, *"Move up"*, *"Camera forward"* | `direction`: `string`<br>Allowed: `["LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACKWARD"]` | `distance_cm`: `number`<br>Range: `1.0` to `25.0` | If `distance_cm` omitted, defaults to **`5.0` cm**. | Relative linear translation along tool-frame axes (`poseTrans`) at steady $v = 0.05\text{ m/s}$. |
| **`tilt_camera`** | *"Tilt down 15 degrees"*, *"Tilt right"*, *"Tilt up"* | `direction`: `string`<br>Allowed: `["UP", "DOWN", "LEFT", "RIGHT"]` | `angle_deg`: `number`<br>Range: `1.0` to `45.0` | If `angle_deg` omitted, defaults to **`15.0` degrees**. | Angular rotation around the Tool Center Point (TCP) in tool coordinates. |
| **`reset_camera`** | *"Reset camera"*, *"Center view"*, *"Camera home"* | *(none)* | *(none)* | Parameterless (`slots: {}` enforced by schema). | Returns endoscope wand to predefined overhead laparoscopic home viewpoint. |

---

### Category 5: `UNKNOWN` (Out-of-Domain & Ambiguous Fallbacks)

| Intent | Spoken Examples | Required Slots | Optional Slots | Default Values & Omission Behavior | Subsystem Actuation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`unknown`** | *"It is raining outside"*, *"Pass that thing"*, *"Hello"* | *(none)* | *(none)* | `slots: {}`. Emitted with `status: "OUT-OF-DOMAIN"` or `"AMBIGUOUS"`. | **Zero physical actuation**. Safe fallback; displays transcript or clarification prompt in UI. |

---

### Summary of Slot Defaults

Downstream actuators (UR10e controller and UI viewer) apply the following defaults whenever optional slots are omitted in incoming JSON payloads:

| Slot Key | Parent Intent | Type | Allowed Values / Range | Default Value When Omitted |
| :--- | :--- | :--- | :--- | :--- |
| **`direction`** | `zoom_display` | `string` | `["IN", "OUT"]` | **`"IN"`** (magnify) |
| **`factor`** | `zoom_display` | `number` | `1.1` to `10.0` | **`2.0`** (2x zoom) |
| **`step_count`** | `navigate_slice` | `integer` | $\ge 1$ | **`1`** (when `direction` is specified) |
| **`scan_type`** | `close_scan` | `string` | `["ct", "mri", "xray"]` | **Active / currently focused scan** |
| **`comment`** | `log_event` | `string` | any string | *(omitted)* No extra clinical annotation |
| **`distance_cm`**| `move_camera` | `number` | `1.0` to `25.0` | **`5.0` cm** |
| **`angle_deg`** | `tilt_camera` | `number` | `1.0` to `45.0` | **`15.0` deg** |

---

## 3. How the NLP Pipeline Uses This Schema

### A. Entity Canonicalization (Synonym Mapping)
The downstream robot controller **never** parses surgical slang or synonyms. The NLP stage must normalize raw spoken terms into canonical schema tokens before constructing the JSON:

```python
SYNONYM_MAP = {
    # Tool Synonyms -> Canonical token
    "knife": "scalpel",
    "ten blade": "scalpel",
    "blade": "scalpel",
    "iris": "scissors",
    "metz": "scissors",
    "metzenbaum": "scissors",
    "clamp": "hemostat",
    "snap": "hemostat",
    "tweezers": "forceps",
    "pickups": "forceps"
}

def resolve_tool(raw_tool_token: str) -> str:
    return SYNONYM_MAP.get(raw_tool_token.lower(), raw_tool_token)
```

---

### B. Validating Payloads Before Dispatch
Before publishing any command across the ZeroMQ/IPC bus, validate your output against `schemas/command_schema.json` using Python's `jsonschema` library:

```python
import json
import jsonschema

# Load once at startup
with open("schemas/command_schema.json") as f:
    SCHEMA = json.load(f)

def validate_and_send(payload: dict):
    try:
        # Raises jsonschema.ValidationError if any key, enum, or slot rule is violated
        jsonschema.validate(instance=payload, schema=SCHEMA)
        # Send over ZeroMQ / socket...
        zmq_socket.send_json(payload)
    except jsonschema.ValidationError as e:
        print(f"[ERROR] Invalid command payload generated by NLU: {e.message}")
```

---

### C. Example Payloads for `log_event`

#### Variant A: Event only (`comment` omitted)
*"Log event: first incision"*
```json
{
  "command_id": "cmd_1003_142010_001",
  "timestamp": 1727800100.150,
  "latency_ms": 112.4,
  "raw_transcript": "log event: first incision",
  "intent": "log_event",
  "category": "MONITOR_DISPLAY",
  "slots": {
    "event_name": "first incision"
  },
  "confidence": 0.97,
  "priority": "NORMAL",
  "status": "EXECUTE"
}
```

#### Variant B: Event + Comment combined
*"Log event: first incision, note patient vitals stable"*
```json
{
  "command_id": "cmd_1003_142015_002",
  "timestamp": 1727800105.210,
  "latency_ms": 125.8,
  "raw_transcript": "log event: first incision, note patient vitals stable",
  "intent": "log_event",
  "category": "MONITOR_DISPLAY",
  "slots": {
    "event_name": "first incision",
    "comment": "patient vitals stable"
  },
  "confidence": 0.96,
  "priority": "NORMAL",
  "status": "EXECUTE"
}
```

---

### D. Handling Non-Executable Speech (OOD & Ambiguous)
When speech is irrelevant or unclear, emit a valid payload with the appropriate status flag so the system UI can log or show feedback:

#### Out-of-Domain Example (Conversational banter):
```json
{
  "command_id": "cmd_1002_001505_002",
  "timestamp": 1727800005.0,
  "latency_ms": 85.4,
  "raw_transcript": "it is raining outside",
  "intent": "unknown",
  "category": "UNKNOWN",
  "slots": {},
  "confidence": 0.15,
  "priority": "LOW",
  "status": "OUT-OF-DOMAIN"
}
```

#### Ambiguous Example (Confidence too low or missing entity):
```json
{
  "command_id": "cmd_1002_001510_003",
  "timestamp": 1727800010.0,
  "latency_ms": 110.0,
  "raw_transcript": "pass me that thing",
  "intent": "pick_tool",
  "category": "TOOL_HANDLING",
  "slots": {},
  "confidence": 0.52,
  "priority": "NORMAL",
  "status": "AMBIGUOUS"
}
```
Downstream actuators will only execute if `"status": "EXECUTE"`.

