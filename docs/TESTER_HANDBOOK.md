# Tester Quick-Reference Handbook — scrub-voice

Welcome to the **scrub-voice** testing handbook. This guide is your single hand-reference for testing every module of the intraoperative voice-control pipeline, running interactive tools, and generating academic benchmarks for your supervisor.

---

## Quick Navigation

- [1. Subsystem Testing Cheatsheet](#1-subsystem-testing-cheatsheet)
- [2. DSP Testing: Listening to Stages](#2-dsp-testing-listening-to-stages)
- [3. ASR Testing: Transcribing Any Audio](#3-asr-testing-transcribing-any-audio)
- [4. NLU Testing: Edge Verification](#4-nlu-testing-edge-verification)
- [5. Monitor Testing: Live Prompt Console](#5-monitor-testing-live-prompt-console)
- [6. Orchestrator & Robot Actuator Testing](#6-orchestrator--robot-actuator-testing)
- [7. Benchmark Suite: Supervisor Delivery](#7-benchmark-suite-supervisor-delivery)

---

## 1. Subsystem Testing Cheatsheet

| Task / Module | Tool / Script | Command to Run |
| :--- | :--- | :--- |
| **DSP (Audio Listening)** | [`test/test_dsp_listen.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_dsp_listen.py) | `.venv/bin/python test/test_dsp_listen.py --input data/asr/test_mix_voice.wav` |
| **ASR (Speech Recognition)** | [`test/test_asr_audio.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_asr_audio.py) | `.venv/bin/python test/test_asr_audio.py data/asr/test_clip_event.wav` |
| **NLU (Edge Inference)** | [`test/test_edge.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_edge.py) | `.venv/bin/python test/test_edge.py` |
| **Monitor (Prompt Console)** | [`test/test_monitor_prompt.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_monitor_prompt.py) | `.venv/bin/python test/test_monitor_prompt.py` |
| **Monitor (Web Dashboard)** | [`src/orchestrator/run_monitor_service.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/orchestrator/run_monitor_service.py) | `.venv/bin/python src/orchestrator/run_monitor_service.py --host 0.0.0.0 --port 8000` |
| **Orchestrator (Preemption)** | [`test/test_orchestrator.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_orchestrator.py) | `.venv/bin/python -m unittest test/test_orchestrator.py` |
| **Supervisor (Crash Recovery)** | [`test/test_supervisor_chaos.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_supervisor_chaos.py) | `.venv/bin/python -m unittest test/test_supervisor_chaos.py` |
| **MuJoCo (Robot Simulator)** | [`test/test_mujoco_sim.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/test_mujoco_sim.py) | `.venv/bin/python -m unittest test/test_mujoco_sim.py` |
| **Full Benchmark Suite** | [`test/benchmark_suite.py`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/benchmark_suite.py) | `.venv/bin/python test/benchmark_suite.py --stages all` |

---

## 2. DSP Testing: Listening to Stages

DSP cannot be tested with pure code assertions; as a tester, you need to **hear** the acoustic difference stage by stage.

### How It Works
The script takes any WAV file (single-channel or multi-channel array recording), processes it through each DSP transformation offline, and saves 3 sequential WAV files:
1. `_stage1_raw_mic1.wav` — Unfiltered raw acoustic audio.
2. `_stage2_beamformed.wav` — Delay-and-Sum spatial beamforming directed at the speaker's angle.
3. `_stage3_denoised.wav` — Neural suppression (DuoFilter ONNX) removing ambient noise.

### Run on Reference Recording
```bash
.venv/bin/python test/test_dsp_listen.py --input data/asr/test_mix_voice.wav
```

### Run on Your Own Audio (with Array Simulation)
If you have a regular mono WAV file (e.g. from your phone, web, or mic), add `--simulate-array` to simulate a 6-microphone circular array with 5 dB noise:
```bash
.venv/bin/python test/test_dsp_listen.py --input path/to/my_speech.wav --simulate-array
```

### What You Hear When Listening Back-to-Back:
- **Stage 1 (Raw):** Heavy background noise, room reverberation, low voice clarity.
- **Stage 2 (Beamformed):** Background sounds from the side and back drop by 3–5 dB; the speaker's direct voice becomes centered.
- **Stage 3 (Denoised):** Constant hiss, air conditioning, and equipment hum disappear; voice is clean and crisp.

All audio files are saved in: [`artifacts/dsp_listen/`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/artifacts/dsp_listen/).

---

## 3. ASR Testing: Transcribing Any Audio

Feed any audio file (DSP outputs, internet recordings, or AI-generated speech from ElevenLabs/TTS) into the Vosk speech engine.

### Run on a Single Audio File
```bash
.venv/bin/python test/test_asr_audio.py data/asr/test_clip_event.wav
```

### Run on a Folder of Audio Files (e.g., DSP Outputs)
```bash
.venv/bin/python test/test_asr_audio.py artifacts/dsp_listen/
```

### What It Reports:
1. **Speed & Latency:** Audio duration, inference time (ms), and Real-Time Factor (RTF). Target RTF is $< 0.30$.
2. **Transcript & Confidence:** The recognized text and average confidence score.
3. **Word-Level Breakdown:** Start time, end time, individual confidence for each word, and a checkmark (`✓`) if the word belongs to the official surgical vocabulary.
4. **NLU Preview:** Automatically passes the transcript into SetFit NLU to show what intent (`open_scan`, `pick_tool`, etc.) and slots would be triggered!

---

## 4. NLU Testing: Edge Verification

The NLU module uses a lightweight SetFit model (`setfit_bge-micro-v2_8k`) optimized for edge inference on the Raspberry Pi 4.

### Run the Edge Test
```bash
.venv/bin/python test/test_edge.py
```

### What It Validates:
- **Cold Load Time:** Loads model in under 1 second (measured: ~0.78s).
- **RAM Footprint:** Consumes only +50 MB of memory (well within the 300 MB budget).
- **CPU Affinity:** Executes on a single physical CPU core.
- **Inference Speed:** ~7.5 ms per inference (budget is 25 ms).
- **100% Schema Validation:** Tests 7 clinical commands + out-of-domain phrases against [`schemas/command_schema.json`](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/schemas/command_schema.json).

---

## 5. Monitor Testing: Live Prompt Console

Test how the Intraoperative Monitor executes standardized commands in real time.

### Step 1: Start the Monitor Web Service
In Terminal 1:
```bash
.venv/bin/python src/orchestrator/run_monitor_service.py --host 0.0.0.0 --port 8000
```
Open your browser to: **`http://localhost:8000`** (shows CT/MRI viewer, slice counter, tool tray, emergency state).

### Step 2: Open the Interactive Prompt Console
In Terminal 2:
```bash
.venv/bin/python test/test_monitor_prompt.py
```

### Quick Menu & Shorthands:
You can press a number or type natural phrases:
- `[1]` or type `open ct scan` $\to$ Opens CT viewer on screen.
- `[2]` or type `open mri scan` $\to$ Switches to MRI series.
- `[3]` or type `goto slice 45` $\to$ Scrolls viewer to slice 45.
- `[4]` or type `zoom in 2.5x` $\to$ Magnifies image to 2.5x.
- `[5]` or type `hand me the scalpel` $\to$ Highlights Scalpel in tool tray.
- `[6]` or type `park arm` $\to$ Retracts robotic arm to home.
- `[7]` or type `log event trocar placed` $\to$ Appends milestone to audit log.
- `[8]` or type `emergency stop immediately` $\to$ **TRIGGERS RED EMERGENCY BANNER** via high-priority ZeroMQ channel.
- `[9]` or type `clear emergency` $\to$ Clears safety interlock.
- `[0]` or type `close scan` $\to$ Closes active imaging view.

### Batch Command (Direct from Command Line)
```bash
.venv/bin/python test/test_monitor_prompt.py "open ct scan"
.venv/bin/python test/test_monitor_prompt.py "zoom in 2.5x"
.venv/bin/python test/test_monitor_prompt.py "emergency stop"
```

---

## 6. Orchestrator & Robot Actuator Testing

### 1. Command Dispatcher Unit Tests
Tests multi-topic message routing (`TOOL_HANDLING`, `MONITOR_DISPLAY`), schema gating, and high-priority preemption:
```bash
.venv/bin/python -m unittest test/test_orchestrator.py
```

### 2. Supervisor Chaos & Auto-Recovery
Kills worker processes with `SIGKILL` and verifies that the supervisor detects the crash and restarts the worker in under 2 seconds without dropping the web dashboard:
```bash
.venv/bin/python -m unittest test/test_supervisor_chaos.py
```

### 3. MuJoCo Robot Arm Simulation
Tests UR-10e robotic arm kinematics, RG-2 gripper actuation, and surgical tool trajectory execution:
```bash
.venv/bin/python -m unittest test/test_mujoco_sim.py
```

---

## 7. Benchmark Suite: Supervisor Delivery

When you need to generate formal evaluation data for your supervisor or final thesis chapter:

### Fast Smoke Test (< 5 Seconds)
Verifies all pipelines, mathematics, and statistical adapters without loading heavy models:
```bash
.venv/bin/python test/benchmarks/bench_smoke.py
```

### Run Individual Subsystem Benchmarks
- **DSP Benchmark (RTF, SI-SDR, STOI, DOA error across SNR sweep):**
  ```bash
  .venv/bin/python test/benchmarks/bench_dsp.py --synthetic 2 --reps 10 --snr-sweep "15,0"
  ```
- **NLU Benchmark (SetFit Accuracy, Slot F1, Wilson 95% Confidence Intervals):**
  ```bash
  .venv/bin/python test/benchmarks/bench_nlu.py --reps-nlu 2
  ```
- **IPC Benchmark (ZeroMQ transport latency, priority preemption, fuzzing):**
  ```bash
  .venv/bin/python test/benchmarks/bench_ipc.py --reps-ipc 100
  ```
- **End-to-End Pipeline Benchmark (Post-speech latency across ablations):**
  ```bash
  .venv/bin/python test/benchmarks/bench_e2e.py --synthetic 1 --warmup 1
  ```

### Master Run (Generates All Reports for Supervisor)
```bash
.venv/bin/python test/benchmark_suite.py --stages dsp,nlu,ipc,e2e --output-dir reports/
```

### What You Deliver to Your Supervisor:
1. **`reports/report.json`**: Complete machine-readable experimental dataset with p50, p95, and Wilson 95% confidence intervals.
2. **`reports/table_budgets.tex`**: Publication-ready LaTeX table proving all subsystems meet edge constraints on the Raspberry Pi 4.
3. **`reports/table_ablation.tex`**: Statistical proof (paired Wilcoxon signed-rank test with Holm-Bonferroni correction, $p < 0.001$) demonstrating that beamforming + neural denoising recovers operating room voice recognition from 42% WER down to 11% WER.

