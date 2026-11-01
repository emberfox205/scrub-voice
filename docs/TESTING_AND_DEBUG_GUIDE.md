# Testing and Debug Guide — scrub-voice

> **Project:** `scrub-voice` — Offline Edge Intraoperative Surgical Voice Assistant  
> **Document Role:** Authoritative manual for subsystem validation, interactive debugging, audio listening, fault injection, and operational troubleshooting.

---

## Helper Execution Commands

```bash
# ------------------------------------------------------------------------------
# 1. Interactive Audio & DSP Listening
# ------------------------------------------------------------------------------
# Listen to Stage 1 (Raw) -> Stage 2 (Beamformed) -> Stage 3 (Denoised) offline:
python test/test_dsp_listen.py --input data/asr/test_mix_voice.wav

# Simulate a 6-microphone circular array on any arbitrary mono speech WAV:
python test/test_dsp_listen.py --input path/to/my_voice.wav --simulate-array

# Comparative DSP Denoiser evaluation against Vosk ASR & NLU:
python test/test_denoiser_eval.py --audio data/asr/test_clip_event.wav

# ------------------------------------------------------------------------------
# 2. Speech Recognition (ASR) & NLU Intent Validation
# ------------------------------------------------------------------------------
# Transcribe any audio file with word timestamps & surgical vocabulary check:
python test/test_asr_audio.py data/asr/test_clip_event.wav

# Transcribe with constrained grammar and downstream NLU intent preview:
python test/test_asr_audio.py data/asr/test_clip_event.wav --grammar --with-nlu

# Edge NLU cold-start, RAM delta, CPU core affinity, and 100% schema assertion:
python test/test_edge.py

# ------------------------------------------------------------------------------
# 3. Intraoperative Monitor UI & Prompt Testing
# ------------------------------------------------------------------------------
# Terminal 1: Launch Monitor Web Service + ZeroMQ Dispatcher (http://localhost:8000):
python src/orchestrator/run_monitor_service.py --host 0.0.0.0 --port 8000

# Terminal 2: Interactive Prompt Console (type natural language or numbers):
python test/test_monitor_prompt.py

# Terminal 2: Send single standardized voice prompt directly from CLI:
python test/test_monitor_prompt.py "open ct scan"
python test/test_monitor_prompt.py "zoom in 2.5x"
python test/test_monitor_prompt.py "emergency stop immediately"

# Inject simulated telemetry sequences (routine clinical workflow):
python test/mock_publisher.py --preset routine --interval 1.0

# ------------------------------------------------------------------------------
# 4. Orchestrator, Supervisor & Hardware Tests
# ------------------------------------------------------------------------------
# Unit test ZeroMQ topic dispatching, schema gating, and emergency preemption:
python -m unittest test/test_orchestrator.py

# Fault tolerance chaos test: SIGKILL worker processes and assert sub-2s recovery:
python -m unittest test/test_supervisor_chaos.py

# MuJoCo UR-10e robot arm kinematics and RG-2 gripper forward simulation:
python -m unittest test/test_mujoco_sim.py

# Hardware recording on Raspberry Pi (RAM-first /dev/shm staging, DMA aligned):
python test/record_audio.py -d 5 -o data/wav_files/test_recording.wav
```

---

## 1. Subsystem Testing Quick Reference

| Subsystem / Tool | Test Script | Primary Verification | Default Command |
| :--- | :--- | :--- | :--- |
| **DSP Audio Listening** | [`test/test_dsp_listen.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_dsp_listen.py) | Human listening test: Raw $\rightarrow$ Beam $\rightarrow$ Denoised | `python test/test_dsp_listen.py --input data/asr/test_mix_voice.wav` |
| **DSP Matrix Eval** | [`test/test_denoiser_eval.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_denoiser_eval.py) | Side-by-side comparative denoiser matrix against ASR & NLU | `python test/test_denoiser_eval.py --audio data/asr/test_clip_event.wav` |
| **ASR Speech-to-Text** | [`test/test_asr_audio.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_asr_audio.py) | Transcribe any audio, RTF latency, grammar vocabulary check | `python test/test_asr_audio.py data/asr/test_clip_event.wav` |
| **NLU Edge Profiler** | [`test/test_edge.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_edge.py) | Cold-start ($< 1\text{s}$), RAM ($< 300\text{ MB}$), 1 core, 100% schema | `python test/test_edge.py` |
| **Monitor Prompt Console** | [`test/test_monitor_prompt.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_monitor_prompt.py) | Interactive CLI console for standardized clinical commands | `python test/test_monitor_prompt.py` |
| **Monitor Web Dashboard** | [`src/orchestrator/run_monitor_service.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/orchestrator/run_monitor_service.py) | FastAPI WebSocket server (`:8000`) + Command Dispatcher | `python src/orchestrator/run_monitor_service.py` |
| **Monitor Handlers** | [`test/test_monitor_handlers.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_monitor_handlers.py) | State transitions, slice clamping, safety interlocks | `python -m unittest test/test_monitor_handlers.py` |
| **Monitor Pipeline** | [`test/test_monitor_pipeline.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_monitor_pipeline.py) | IPC PUB/SUB $\rightarrow$ MonitorBridge $\rightarrow$ WebSocket stream | `python -m unittest test/test_monitor_pipeline.py` |
| **Command Dispatcher** | [`test/test_orchestrator.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_orchestrator.py) | Priority preemption (Emergency rank 0), schema gating | `python -m unittest test/test_orchestrator.py` |
| **Supervisor Chaos** | [`test/test_supervisor_chaos.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_supervisor_chaos.py) | `SIGKILL` fault injection, automated sub-2s worker recovery | `python -m unittest test/test_supervisor_chaos.py` |
| **MuJoCo Simulation** | [`test/test_mujoco_sim.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_mujoco_sim.py) | UR-10e arm kinematics, RG-2 gripper actuation, trajectory | `python -m unittest test/test_mujoco_sim.py` |
| **Hardware Audio Rec** | [`test/record_audio.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/record_audio.py) | RAM-first staging, DMA alignment, live dB meters, tap test | `python test/record_audio.py -d 5` |
| **Mock Publisher** | [`test/mock_publisher.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/mock_publisher.py) | Injects preset command streams into ZeroMQ command bus | `python test/mock_publisher.py --preset routine` |

---

## 2. DSP Subsystem Testing & Step-by-Step Listening

DSP algorithms cannot be verified solely through unit-test assertions. Human listening is essential to confirm speech naturalness, background suppression, and absence of musical noise artifacts.

### 2.1 Listening to Intermediate Transformation Stages (`test_dsp_listen.py`)
Processes any input WAV file offline and exports three sequential stages to `artifacts/dsp_listen/`:
1. `_stage1_raw_mic1.wav`: Unfiltered ambient microphone capture (or simulated array Mic 0).
2. `_stage2_beamformed.wav`: Delay-and-Sum (DAS) beamformed audio steered toward the localized speaker Angle of Arrival (DOA).
3. `_stage3_denoised.wav`: Neural-enhanced output after DuoFilter ONNX suppression with Split-Band Consonant Shielding.

```bash
# Test on repository reference multi-channel recording:
python test/test_dsp_listen.py --input data/asr/test_mix_voice.wav

# Test on any single-channel speech file (synthesizes 6-mic circular array with noise):
python test/test_dsp_listen.py --input path/to/my_speech.wav --simulate-array --noise-db 5.0
```

#### What You Hear When Listening Back-to-Back:
* **Stage 1 (Raw):** Diffuse HVAC drone, background monitor bleeps, and room reverberation.
* **Stage 2 (Beamformed):** Directional acoustic focus; off-axis ambient sound drops by 3–6 dB; speaker voice centers.
* **Stage 3 (Denoised):** Continuous hiss and ventilation rumble vanish; vocal formants and consonant edges (/s/, /t/, /θ/) remain intelligible without robotic "watery" artifacts.

### 2.2 Comparative Denoiser Matrix Evaluation (`test_denoiser_eval.py`)
Benchmarks different denoiser checkpoints and attenuation strategies side-by-side against Vosk ASR and downstream NLU:
* Evaluates 4 conditions:
  1. `Beamformed` (baseline DAS)
  2. `Denoised Old` (full-band aggressive suppression)
  3. `Denoised New` (Split-band consonant shield at 2.5 kHz)
  4. `Denoised Dialed-back` (attenuation ceiling limited to 12 dB)
* Reports Word Error Rate (WER), keyword hit rate, and downstream NLU intent accuracy.

```bash
python test/test_denoiser_eval.py --audio data/asr/test_clip_event.wav
```

### 2.3 Diagnosing Common DSP Acoustic Glitches
* **Consonant Erasure (e.g. "three" $\rightarrow$ "ray" or "free"):** Aggressive neural suppression treating high-frequency fricatives as noise.
  * *Fix:* Enable Split-Band Consonant Shielding in `src/dsp/denoise/stateful_onnx/duofilter_streaming_denoiser.py` (`shield_cutoff_hz=2500`).
* **Frame Boundary Clicks:** Chunked processing without state continuity.
  * *Fix:* Ensure using `DuoFilterStreaming` with persistent state tensors rather than stateless frame filtering.
* **Clipping Distortion:** Audio exceeding $[-1.0, +1.0]$ float range.
  * *Fix:* Verify automatic peak normalization or soft limiting before 16-bit PCM export.

---

## 3. ASR Subsystem Testing & Vocabulary Inspection

### 3.1 Transcribing Any Audio File (`test_asr_audio.py`)
Feeds any audio file (DSP outputs, phone recordings, synthetic TTS) into the production Vosk speech recognition engine:

```bash
# Standard transcription with latency and Real-Time Factor (RTF) calculation:
python test/test_asr_audio.py data/asr/test_clip_event.wav

# Enable strict surgical grammar and preview downstream NLU classification:
python test/test_asr_audio.py data/asr/test_clip_event.wav --grammar --with-nlu

# Batch evaluation across a directory of audio files:
python test/test_asr_audio.py artifacts/dsp_listen/
```

#### Output Metrics:
1. **Timing & RTF:** Audio duration, wall-clock inference time (ms), and Real-Time Factor (RTF). Target RTF is $< 0.30$.
2. **Confidence:** Utterance confidence and token-level posteriors.
3. **Grammar Vocabulary Check:** Displays checkmark (`✓`) for words matching `src/asr/vocab.py` and flags (`✗`) out-of-vocabulary words.
4. **NLU Preview:** Displays the parsed intent, slots, and JSON schema compliance.

### 3.2 Direct Transcription CLI (`src/asr/scripts/transcribe.py`)
```bash
python src/asr/scripts/transcribe.py data/asr/test_mix_voice.wav --config vosk-grammar+freeform-log
```

### 3.3 Diagnosing ASR Failure Modes
* **Leading Word Truncation (e.g. "halt robot" $\rightarrow$ "robot"):** VAD triggering after initial phoneme onset.
  * *Fix:* Ensure DSP maintains a 500 ms pre-speech ring buffer so the opening consonant is not clipped.
* **Low Confidence on Valid Commands:** Audio sample rate or bit-depth mismatch.
  * *Fix:* Ensure audio is strictly 16 kHz 16-bit signed PCM (`subtype="PCM_16"`). Vosk misinterprets 44.1 kHz or 48 kHz audio as high-speed speech.

---

## 4. NLU Subsystem Testing & Edge Constraints

### 4.1 Edge Resource & Schema Verification (`test_edge.py`)
Validates the production `VoiceToIntentPipeline` against strict Raspberry Pi 4 edge constraints:

```bash
python test/test_edge.py
```

#### Checked Constraints:
* **Cold Model Load:** Initial model initialization in $< 1.0\text{ s}$ (measured: ~0.78 s).
* **RAM Footprint:** Resident memory delta $< 300\text{ MB}$ (measured: +50.0 MB).
* **Single Core Execution:** Pinned to single physical CPU core via PyTorch thread settings.
* **Inference Speed:** Average latency $< 25\text{ ms}$ (measured: ~7.5 ms).
* **Schema Conformance:** 100% compliance of returned envelopes against [`schemas/command_schema.json`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/schemas/command_schema.json).
* **Out-of-Domain Rejection:** Confirms conversational chatter (e.g. *"What time is it?"*, *"Let's grab lunch"*) routes to `status: "OUT-OF-DOMAIN"` with `intent: "unknown"`.

---

## 5. Intraoperative Monitor UI & Prompt Testing

### 5.1 Interactive Standardized Prompt Console (`test_monitor_prompt.py`)
Allows testers and clinical evaluators to drive the monitor screen using standardized natural language or quick numeric menus:

```bash
# Step 1: In Terminal 1, start the combined Monitor & Dispatcher service:
python src/orchestrator/run_monitor_service.py --host 0.0.0.0 --port 8000

# Step 2: Open browser to http://localhost:8000 (IBM Carbon g100 Clinical Dashboard)

# Step 3: In Terminal 2, launch the prompt console:
python test/test_monitor_prompt.py
```

#### Interactive Quick Menu:
```text
  [1] open ct scan          [5] hand me scalpel       [9] clear emergency
  [2] open mri scan         [6] park arm              [0] close scan
  [3] goto slice 45         [7] log event trocar      [h] show menu
  [4] zoom in 2.5x          [8] emergency stop        [q] quit console
```

#### Direct Single-Command Invocations:
```bash
python test/test_monitor_prompt.py "open ct scan"
python test/test_monitor_prompt.py "navigate to slice 75"
python test/test_monitor_prompt.py "zoom display by 3 times"
python test/test_monitor_prompt.py "emergency stop immediately"
```

### 5.2 Mock Publisher Presets (`test/mock_publisher.py`)
Injects pre-programmed clinical sequences into the ZeroMQ command bus:
```bash
# Automated routine surgical workflow (opens CT, scrolls slices, zooms, picks scalpel):
python test/mock_publisher.py --preset routine --interval 1.5

# Rapid emergency burst test:
python test/mock_publisher.py --preset emergency
```

### 5.3 Automated Monitor Unit & Pipeline Tests
```bash
# Test all 9 modular intent handlers (scan, slice, zoom, tool, emergency, event):
python -m unittest test/test_monitor_handlers.py

# Test ZeroMQ PUB/SUB -> MonitorBridge -> WebSocket async broadcasting:
python -m unittest test/test_monitor_pipeline.py
```

---

## 6. Orchestrator, Supervisor & Fault-Tolerance Chaos Testing

### 6.1 Command Dispatcher Routing & Preemption (`test_orchestrator.py`)
```bash
python -m unittest test/test_orchestrator.py
```
Validates:
1. Multi-topic routing: `TOOL_HANDLING`, `MONITOR_DISPLAY`, `EMERGENCY`.
2. Schema validation: Malformed envelopes are intercepted and directed to `ERROR` topic.
3. Priority Preemption: High-priority emergency commands jump ahead of 45 queued normal commands.

### 6.2 Supervisor Chaos & Auto-Recovery (`test_supervisor_chaos.py`)
```bash
python -m unittest test/test_supervisor_chaos.py
```
* **Fault Injection:** Injects `SIGKILL` (-9) into active subsystem child workers (DSP, ASR, NLU).
* **Assertion:** Verifies the supervisor watchdog detects the dead PID and revives the worker in $< 2.0\text{ s}$ without dropping the WebSocket dashboard connection or resetting state.

---

## 7. Hardware Recording & Audio Format Invariants

### 7.1 Recording Utility (`test/record_audio.py`)
Conforms strictly to `AGENTS.md` edge protection guardrails:
* **RAM-First Staging:** Records to `/dev/shm` in RAM first to prevent SD card controller lockups.
* **DMA Ring Buffer:** Configures `--buffer-size=65536 --period-size=8192` to eliminate ALSA XRUNs.
* **Clean Driver Release:** Intercepts `Ctrl+C` to ensure ALSA device handles close properly.

```bash
# 5-second recording:
python test/record_audio.py -d 5 -o data/wav_files/test_capture.wav

# Interactive manual recording (press Enter to stop):
python test/record_audio.py --manual

# Record and immediately run ASR + NLU evaluation:
python test/record_audio.py -d 5 --eval
```

---

## 8. MuJoCo Robot Arm & Gripper Simulation Testing

### 8.1 Automated Kinematics Test (`test_mujoco_sim.py`)
```bash
python -m unittest test/test_mujoco_sim.py
```
Verifies:
* Menagerie UR-10e kinematic model and OnRobot RG-2 4-bar linkage.
* Forward kinematics and Cartesian Tool Center Point (TCP) coordinates.
* Minimum-jerk S-curve cosine trajectories to tray pose and home pose.

### 8.2 Interactive 3D Simulation Console
```bash
python -m src.actuators.sim_ws.mujoco_pipe_server
```
Type `tray`, `home`, `open`, `close`, `status`, or `exit`.

---

## 9. Comprehensive Troubleshooting Matrix

| Symptom / Error | Root Cause | Exact Resolution |
| :--- | :--- | :--- |
| **`ALSA lib pcm.c: xrun: prepare error`** | Hardware DMA period buffer under/overflow on Raspberry Pi | Run recordings with aligned DMA periods: `--buffer-size=65536 --period-size=8192` (handled automatically by `record_audio.py`). |
| **`zmq.error.ZMQError: Address already in use`** | Previous service crashed without closing socket, or port 8000/5555 is occupied | Kill orphaned processes: `pkill -9 -f run_monitor_service` or run `netstat -ano \| findstr 8000` (Windows) / `lsof -i :8000` (Linux). |
| **`jsonschema.exceptions.ValidationError`** | Command envelope missing one of the 10 mandatory keys | Verify NLU output against `schemas/command_schema.json` via `test/test_edge.py`. Ensure `command_id` matches `^cmd_[a-zA-Z0-9_]+$`. |
| **ASR outputs empty string on speech** | WAV file is 44.1 kHz/48 kHz or 32-bit float | Convert audio to 16 kHz 16-bit signed PCM mono: `sox input.wav -r 16000 -c 1 -b 16 output.wav` or use `test_dsp_listen.py`. |
| **Monitor UI displays "Disconnected"** | WebSocket gateway not running on port 8000 | Start `python src/orchestrator/run_monitor_service.py` before opening browser or running `test_monitor_prompt.py`. |
| **PyTorch CPU throttling / lag on Pi 4** | PyTorch spawning threads across all cores, causing scheduling contention | Set `torch.set_num_threads(1)` or `(2)`. The NLU pipeline is pre-configured for 2 threads on edge ARM. |

