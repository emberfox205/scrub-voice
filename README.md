# scrub-voice 

[![Python 3.13.5](https://img.shields.io/badge/python-3.13.5-blue.svg)](https://www.python.org/)

An Offline, Noise-Robust, Edge Voice-to-Intent Assistant for Sterile Medical Environments.

## Synopsis

`scrub-voice` provides touchless voice control for surgical teams in operating rooms and sterile environments. The system captures multi-channel audio, filters acoustic noise, transcribes medical commands, extracts intents and parameters offline, and directs physical actuation and clinical displays—operating entirely at the edge without cloud dependencies.

### Core Components

- **Digital Signal Processing (DSP):** Multi-channel microphone array audio capture, delay-and-sum beamforming, sound source localization (SRP-PHAT DOA), noise suppression (DeepFilterNet/DuoFilter), and voice activity detection (Silero VAD).
- **Automatic Speech Recognition (ASR):** Offline speech-to-text pipeline transcribing spoken surgical utterances into raw command text.
- **Natural Language Understanding (NLU):** Few-shot intent classification and slot extraction via a fine-tuned SetFit model (`setfit_bge-micro-v2_8k`), validating payloads against standardized JSON schemas.
- **Intraoperative Clinical Monitor:** Web-based DICOM medical image viewer (CT, MRI, X-ray) and surgical milestone event logger conforming to OHIF and IBM Carbon Dark standards.
- **Physical Systems & Actuation:** Universal Robots UR10e arm and OnRobot RG-2 gripper control via `ur_rtde` and MuJoCo physics simulation for hands-free surgical instrument delivery.

---

## Quick Start (Minimal Setup: Mic + Pi 4 + Web Browser + Sim)

This quick start walks through a minimal edge deployment:
- **Microphone:** ReSpeaker / Seeed 8-channel array (or USB mic / test WAV) connected to the Raspberry Pi 4.
- **Edge Compute:** Raspberry Pi 4 running the core pipeline daemons via the Supervisor.
- **Clinical Dashboard:** Any web browser accessing the intraoperative monitor UI over the local network.
- **Actuator Sim:** Headless MuJoCo physics simulation (no physical robot arm needed).

> [!NOTE]
> *Full installation instructions, hardware wiring, driver setup, and advanced configurations can be found in [`docs/`](docs/) (Work In Progress).*

### 1. Environment Setup

On the Raspberry Pi 4 (or development host):

```bash
git clone https://github.com/emberfox205/ur_arm.git
cd ur_arm

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Pipeline Supervisor

Launch the multi-process supervisor with live microphone input and simulated actuators:

```bash
# Using Seeed 8-mic array and headless MuJoCo robot simulation
python -m src.orchestrator.supervisor --sim --device "plughw:seeed8micvoicec,0"
```

*(If testing without a physical mic, stream a sample medical utterance WAV instead:)*
```bash
python -m src.orchestrator.supervisor --sim --wav data/audio/test_pick_scissors.wav
```

### 3. Open the Intraoperative Clinical Monitor

Open a web browser on any device on the same local network (or locally) and navigate to:

```text
http://<PI_IP>:8000
```
*(Use `http://localhost:8000` if running locally.)*

The dashboard renders the live DICOM viewport, active surgical tool telemetry, patient milestone log, and real-time speech transcription.

### 4. Issue Commands

- **Spoken Voice:** Speak commands clearly toward the microphone:
  - *"Hand me the scalpel"*
  - *"Open CT scan"*
  - *"Zoom in 2.5x"*
  - *"Advance 5 slices"*
  - *"Park arm"*
- **SSH Operator Prompt Console (Alternative):** In a second terminal on the Pi:
  ```bash
  # Send direct tool command
  python test/test_monitor_prompt.py --pick-tool scalpel

  # Or launch interactive console menu
  python test/test_monitor_prompt.py
  ```

---

## Documentation & Reference

For subsystem guides, benchmarks, and hardware deployment references, consult [`docs/`](docs/):

- [Raspberry Pi 4 Deployment Guide](docs/RPI4_DEPLOYMENT_GUIDE.md)
- [Robot & Gripper Control Guide](docs/ROBOT_CONTROL_GUIDE.md)
- [Multi-Mic Audio & ALSA Setup Guide](docs/AUDIO_GUIDE.md)
- [Command Schema Specification](docs/COMMAND_SCHEMA_GUIDE.md)
- [Benchmark & Evaluation Suite](docs/BENCHMARK_GUIDE.md) (and [`evaluations/`](evaluations/README.md))
