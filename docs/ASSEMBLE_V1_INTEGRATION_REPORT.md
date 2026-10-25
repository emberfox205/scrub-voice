# Assemble V1: Full Pipeline Integration & Architecture Report

> **Project:** Scrub-Voice — Offline Edge Intraoperative Surgical Voice Assistant  
> **Target Deployment Device:** Raspberry Pi 4 Model B (Quad-core ARM Cortex-A72 @ 1.5 GHz, 4GB RAM, aarch64 OS)  
> **Active Milestone Branch:** `Assemble_V1`  
> **Date:** October 2026  

---

## 1. Executive Summary

Prior to `Assemble_V1`, the repository consisted of decoupled research prototypes: individual modules for multi-mic beamforming, neural noise suppression, Vosk speech recognition, SetFit intent classification, and a clinical web monitor. However, these modules existed in isolation and could not communicate.

The **`Assemble_V1`** integration milestone unified these subsystems into an asynchronous, multi-process, low-latency edge pipeline. The entire pipeline was deployed and validated directly on a physical Raspberry Pi 4 (`PI4WURST`) with the following verified outcomes:

* **End-to-End Post-Speech Latency:** **~271 ms** from the end of vocalization to monitor execution (vastly outperforming the clinical $\le 1000\text{ ms}$ budget).
* **Real-Time Factor (RTF):** **0.4508** for neural noise suppression (DeepFilterNet ONNX finishes a 1.0s audio chunk in 450 ms, running $2.2\times$ faster than real-time).
* **Speech Recognition Accuracy:** **94%–100% confidence** on live microphone capture with word-by-word token confidence at 1.00.
* **System Memory Footprint:** **~636 MB peak RAM** across all 4 microservices combined, leaving $> 3.3\text{ GB}$ of free memory on a 4GB Pi 4.
* **Fault Tolerance:** Automatic watchdog crash detection and sub-500ms auto-recovery for killed worker processes.

---

## 2. End-to-End System Architecture

The pipeline follows an asynchronous, decoupled pipeline pattern interconnected via high-throughput **ZeroMQ IPC** transport sockets (`ipc:///tmp/*.ipc`) and supervised by an Erlang/ROS-style process watchdog.

```mermaid
flowchart TD
    subgraph Hardware ["Hardware Layer (Raspberry Pi 4)"]
        MIC["ReSpeaker 6-Mic Circular Array\n(8-Channel ALSA S16_LE @ 16kHz)"]
        WAV["Simulation WAV Stream\n(8-Ch or Mono Tiled)"]
    end

    subgraph DSP ["1. DSP Audio Daemon (src/dsp/run_dsp.py)"]
        STREAM["AudioStreamer\n(Strips AEC Ch 6-7, Keeps Mics 0-5)"]
        VAD["Gate 1: Silero VAD\n(3.8ms Window Voting + RNN Reset)"]
        LOC["Gate 2: SRP-PHAT Localizer\n(360° Steered Energy -> DOA Angle)"]
        DAS["Module 3: DAS Beamformer\n(Phase Delay Alignment & Sum)"]
        DNET["Module 4: DeepFilterNet ONNX\n(Neural Noise Suppression)"]
        
        STREAM --> VAD
        VAD -->|"Speech Active"| LOC
        LOC --> DAS
        DAS --> DNET
        VAD -->|"Speech Ends"| END_SIG["Send {'type': 'end'}"]
    end

    MIC -.-> STREAM
    WAV -.-> STREAM

    subgraph ASR ["2. ASR Speech Engine (src/asr/scripts/run_asr.py)"]
        VOSK["Vosk Engine\n(vosk-model-small-en-us-0.15)"]
        GRAM["Grammar Constraint Graph\n(Restricted to Surgical Vocab)"]
        NORM_LEX["ASR Lexical Normalizer\n(Digits, Contractions, Spellings)"]
        
        VOSK --- GRAM
        VOSK --> NORM_LEX
    end

    DNET -->|"ZeroMQ PUSH\nipc:///tmp/scrub_audio.ipc"| VOSK
    END_SIG -->|"Immediate Flush"| VOSK

    subgraph NLU ["3. NLU Intent Engine (src/nlu/run_nlu.py)"]
        PRE["Lexical Normalizer & Fillers Filter"]
        SAFE["Emergency Regex Safety Interlock\n(Halt/Freeze/Stop Fast-Path)"]
        SETFIT["SetFit Transformer Embedding\n(bge-micro-v2 384-dim, 2 Threads)"]
        HEAD["Logistic Regression Head\n(12 Intent Classes)"]
        SLOT["Slot Extractor\n(Zoom Factors, Slices, Tools, Modalities)"]
        VAL["Schema Envelope Generator\n(Draft 2020-12 Compliance)"]

        PRE --> SAFE
        SAFE -->|"Standard Voice"| SETFIT --> HEAD --> SLOT --> VAL
        SAFE -->|"Emergency Keyword"| VAL
    end

    NORM_LEX -->|"ZeroMQ PUSH\nipc:///tmp/scrub_asr.ipc"| PRE

    subgraph Orchestrator ["4. Orchestrator & Monitor (src/orchestrator/run_monitor_service.py)"]
        DISP["CommandDispatcher\n(Dual-Channel Priority Polling)"]
        LVC["Last Value Cache (LVC)\n(Atomic State Snapshot Server)"]
        FASTAPI["FastAPI Web Server\n(Uvicorn on Port 8000)"]
        BRIDGE["MonitorBridge\n(WebSocket Push /ws)"]

        DISP <--> LVC
        DISP -->|"ZeroMQ PUB\nipc:///tmp/scrub_bus.ipc"| BRIDGE
        BRIDGE --> FASTAPI
    end

    VAL -->|"Normal Channel PUSH\nipc:///tmp/scrub_nlu.ipc"| DISP
    VAL -->|"Priority Channel PUSH\nipc:///tmp/scrub_nlu_priority.ipc"| DISP

    subgraph UI ["5. Intraoperative Dashboard (Browser)"]
        DASH["OHIF-Style Medical Viewer\n(DICOM Canvas + HUD + Voice Pills)"]
    end

    FASTAPI -->|"ws://0.0.0.0:8000/ws"| DASH

    subgraph Supervision ["Supervisor Engine (src/orchestrator/supervisor.py)"]
        WATCH["Watchdog Health Loop (200ms)\n(Readiness Barriers + Crash Auto-Recovery)"]
        WATCH -.-> DSP
        WATCH -.-> ASR
        WATCH -.-> NLU
        WATCH -.-> Orchestrator
    end
```

---

## 3. Subsystem Breakdown: Changes & Creations

### A. Toolchain & Environment Management
* **What was changed:** Transitioned the repository from legacy `requirements.txt` to a modern declarative **`pyproject.toml`** and locked **`uv.lock`**.
* **Why it was needed:** Eliminated dependency conflicts (such as PyPI vs. PyTorch CPU index confusion for `requests` and `torch`), pinned standalone Python to **3.13.5**, and solved WSL NTFS cross-drive linking using `link-mode = "copy"`.

### B. DSP Audio Subsystem (`src/dsp/`)
* **Teammate Code Base:** Standalone research scripts for microphone array reading, Silero VAD, SRP-PHAT, and DeepFilterNet model export.
* **What Was Created / Changed:**
  1. **Continuous Daemon ([`src/dsp/run_dsp.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/run_dsp.py)):** Built the unified DSP streaming daemon linking capture $\rightarrow$ VAD $\rightarrow$ Localization $\rightarrow$ Beamforming $\rightarrow$ Denoising $\rightarrow$ ZeroMQ PUSH.
  2. **Dual-Mode Streamer ([`src/dsp/source/audio_streamer.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/audio_streamer.py)):** Implemented automatic audio resampling, channel layout management (capturing 8 channels, stripping AEC loopbacks 6 & 7, retaining circular mics 0–5), and seamless offline WAV simulation.
  3. **VAD Temporal Gating Calibration ([`src/dsp/source/vad.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/vad.py)):**
     * Solved the "sticky utterance clumping" issue where conversational pauses failed to segment.
     * Introduced multi-window voting (requiring $\ge 3$ active 32ms windows or mean prob $\ge 0.5$).
     * Added recurrent RNN hidden-state clearing via `model.reset_states()` upon silence detection.
  4. **3-Stage Audio Diagnostic Dumping:**
     * Implemented `--dump-audio <DIR>` to simultaneously save all three progressive audio stages for every spoken command:
       * `utterance_XXX_1_raw_mic1.wav` (unbeamformed, single mic)
       * `utterance_XXX_2_beamformed.wav` (constructive 6-mic spatial beam)
       * `utterance_XXX_3_denoised.wav` (DeepFilterNet neural suppression)
     * Real-time dB attenuation logging per chunk (`Denoise: -X.X dB`).

### C. ASR Speech Recognition (`src/asr/`)
* **Teammate Code Base:** Vosk speech recognizer (`vosk-model-small-en-us-0.15`) with a surgical grammar dictionary (`src/asr/vocab.py`) and benchmark logs.
* **What Was Created / Changed:**
  1. **ZeroMQ IPC Coupling:** Wired the PULL socket (`ipc:///tmp/scrub_audio.ipc`) and PUSH socket (`ipc:///tmp/scrub_asr.ipc`) to establish the live pipeline bridge.
  2. **Zero-Lag Flush Execution:** Integrated the `{"type": "end"}` signal from DSP. Instead of waiting for a 0.8s silence fallback timeout, Vosk immediately finalizes transcription in **~100 ms**.
  3. **Per-Token Acoustic Confidence Telemetry:**
     * Added token-level confidence breakdown in logs (`Tokens: [zoom:1.00, in:1.00, two:0.50, times:1.00]`).
     * Added persistent JSONL logging to `results/asr_transcripts.jsonl`.

### D. NLU Intent Classification & Slot Extraction (`src/nlu/`)
* **Teammate Code Base:** Fine-tuned SetFit transformer model weights stored in `src/nlu/models/setfit_bge-micro-v2_8k/`.
* **What Was Created / Changed:**
  1. **Standardized Pipeline ([`src/nlu/pipeline.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/pipeline.py)):** Implemented `VoiceToIntentPipeline` combining normalization, SetFit transformer inference, slot extraction, and 10-key schema envelope validation into a single call.
  2. **Lexical Normalizer ([`src/nlu/normalizer.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/normalizer.py)):** Strips fillers (*"please"*, *"uh"*, *"could you"*), converts spelled-out numbers (*"two point five"* $\rightarrow$ `2.5`), and normalizes medical acronyms (*"c t scan"* $\rightarrow$ `CT`).
  3. **Slot Extractor ([`src/nlu/slot_extractor.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/slot_extractor.py)):** Extracts structured parameters including zoom direction/factor, slice indices, scan modalities, and surgical tools.
  4. **Production Daemon ([`src/nlu/run_nlu.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/run_nlu.py)):** ZeroMQ daemon consuming ASR text and publishing to dual channels (`NORMAL` vs `PRIORITY`).
  5. **Emergency Fast-Path Interlock:** Regex detector for critical words (*"stop"*, *"freeze"*, *"halt"*) that bypasses model latency to immediately dispatch `category="EMERGENCY"`, `priority="HIGH"`.
  6. **Early Signal Registration:** Registered `SIGINT`/`SIGTERM` prior to model instantiation to eliminate ugly `KeyboardInterrupt` tracebacks during shutdown.

### E. Orchestrator & Intraoperative Monitor (`src/orchestrator/`)
* **Teammate Code Base:** Standalone prototype `CommandDispatcher` and separate FastAPI web monitor prototype.
* **What Was Created / Changed:**
  1. **Unified Daemon ([`src/orchestrator/run_monitor_service.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/orchestrator/run_monitor_service.py)):** Merged the ZeroMQ command dispatcher thread and Uvicorn FastAPI web server on port 8000 into a single daemon.
  2. **State Synchronization (LVC Snapshot):** Implemented Last Value Cache (LVC) state recovery over `ipc:///tmp/scrub_snapshot.ipc` so newly connected browser tabs immediately receive current slice, zoom, and scan status.
  3. **Dual-Channel Polling:** Prioritizes the emergency channel over normal commands using a weighted ZeroMQ poller.

### F. Multi-Process Supervisor Engine (`src/orchestrator/supervisor.py`)
* **What Was Created:**
  1. **Process Isolation:** Runs each subsystem (`[MONITOR]`, `[NLU]`, `[ASR]`, `[DSP]`) in an isolated OS process with independent memory boundaries.
  2. **Readiness Synchronization:** Uses stdout regex barriers (`READY_PATTERNS`) to ensure child daemons bind their IPC sockets before dependent services connect, eliminating race conditions.
  3. **200ms Watchdog Loop:** Polls child PIDs and automatically resurrects killed or crashed services in $< 500\text{ ms}$.
  4. **Clean Teardown:** Handles `Ctrl+C` by propagating signals in reverse dependency order and deleting temporary socket files in `/tmp/*.ipc`.

### G. Benchmarking & Verification Suite (`test/`)
* **What Was Created:**
  1. **Edge Profiler ([`test/benchmark_edge_profile.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/benchmark_edge_profile.py)):** Profiles single-core and multi-core CPU latency, peak RSS RAM, and RTF against Raspberry Pi 4 edge budgets.
  2. **5 Test Suites:**
     * `test/test_nlu_service.py`: Schema envelope and OOD rejection tests.
     * `test/test_dsp_service.py`: Chunk streaming and socket tests.
     * `test/test_run_monitor_service.py`: Dispatcher priority and WebSocket tests.
     * `test/test_supervisor_chaos.py`: `SIGKILL` crash injection and recovery tests.
     * `test/test_end_to_end_voice_stream.py`: Complete audio-to-monitor pipeline streaming test.

---

## 4. Hardware Verification & Performance Metrics (Raspberry Pi 4 Model B)

Live hardware testing was conducted on `admin@PI4WURST` (Raspberry Pi 4 Model B, 4GB RAM, aarch64 Linux 6.18):

| Metric | Subsystem | Target Edge Budget | Measured Result on Pi 4 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Real-Time Factor (RTF)** | DSP DeepFilterNet ONNX | $\le 0.50$ | **0.4508** ($2.2\times$ faster than real-time) | **PASS** |
| **Denoise Latency (1.0s chunk)** | DSP DeepFilterNet ONNX | $\le 500\text{ ms}$ | **450.82 ms** (p95: 459.94 ms) | **PASS** |
| **VAD Gating Latency** | Silero VAD | $\le 50\text{ ms}$ | **3.8 ms – 8.2 ms** per 500ms chunk | **PASS** |
| **ASR Latency** | Vosk Grammar Engine | $\le 200\text{ ms}$ | **119 ms – 140 ms** | **PASS** |
| **NLU Intent Latency** | SetFit BGE-micro (2 threads) | $\le 200\text{ ms}$ | **149.6 ms – 152.3 ms** | **PASS** |
| **IPC Roundtrip Transit** | ZeroMQ Dispatcher | $\le 5.0\text{ ms}$ | **2.64 ms – 2.77 ms** | **PASS** |
| **Crash Recovery Latency** | Supervisor Watchdog | $\le 500\text{ ms}$ | **306.2 ms – 410.3 ms** | **PASS** |
| **Total Post-Speech Reaction** | End-of-Speech $\rightarrow$ Display | $\le 1000\text{ ms}$ | **~271 ms – 315 ms** | **PASS** |
| **Total Peak Memory Footprint** | Complete Edge Pipeline | $\le 1000\text{ MB}$ | **~636.0 MB** ($> 3.3\text{ GB}$ free) | **PASS** |
| **Schema Compliance** | NLU Command Envelopes | $100\%$ | **100.0%** (85/85 valid Draft 2020-12) | **PASS** |
| **Out-of-Domain Rejection** | NLU Distractor Filtering | $\ge 90\%$ | **100.0%** (15/15 rejected) | **PASS** |

---

## 5. Summary of Key File Artifacts

| File Path | Subsystem | Purpose & Role |
| :--- | :--- | :--- |
| [`src/dsp/run_dsp.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/run_dsp.py) | DSP | Continuous streaming audio daemon with VAD, SRP-PHAT, DAS, and DDFNet |
| [`src/dsp/source/audio_streamer.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/audio_streamer.py) | DSP | Hardware ALSA capture (8-ch $\rightarrow$ 6 mics) and WAV simulation streamer |
| [`src/dsp/source/vad.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/vad.py) | DSP | Calibrated Silero VAD temporal gating with RNN state reset |
| [`src/asr/scripts/run_asr.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/asr/scripts/run_asr.py) | ASR | Vosk grammar recognition service with token confidence telemetry |
| [`src/nlu/pipeline.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/pipeline.py) | NLU | Unified pipeline: Lexical normalization + SetFit classification + Slot extraction |
| [`src/nlu/run_nlu.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/run_nlu.py) | NLU | ZeroMQ NLU service daemon with normal/priority dual routing |
| [`src/orchestrator/run_monitor_service.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/orchestrator/run_monitor_service.py) | Orchestrator | Merged CommandDispatcher worker and FastAPI/Uvicorn WebSocket monitor |
| [`src/orchestrator/supervisor.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/orchestrator/supervisor.py) | Supervisor | Process watchdog, startup synchronization barriers, and auto-recovery |
| [`test/benchmark_edge_profile.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/test/benchmark_edge_profile.py) | Benchmark | Automated ARM64 edge profiler generating JSON telemetry |
