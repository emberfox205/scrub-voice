# scrub-voice 

[![Python 3.13.5](https://img.shields.io/badge/python-3.13.5-blue.svg)](https://www.python.org/)

An Offline, Noise-Robust, Edge Voice-to-Intent Assistant for Sterile Medical Environments.

## Synopsis

`scrub-voice` provides touchless voice control for surgical teams in operating rooms and sterile environments. The system captures multi-channel audio, filters acoustic noise, transcribes medical commands, extracts intents and parameters offline, and directs physical actuation and clinical displays—operating entirely at the edge without cloud dependencies.

### Core Components

- **Digital Signal Processing (DSP):** Multi-channel microphone array audio capture, delay-and-sum beamforming, sound source localization (SRP-PHAT DOA), noise suppression (DeepFilterNet), and voice activity detection (Silero VAD).
- **Automatic Speech Recognition (ASR):** Offline speech-to-text pipeline transcribing spoken surgical utterances into raw command text.
- **Natural Language Understanding (NLU):** Few-shot intent classification and slot extraction via a fine-tuned SetFit model (`setfit_bge-micro-v2_8k`), validating payloads against standardized JSON schemas.
- **Physical Systems & Actuation:**
  - **Universal Robots UR-10e:** Robot arm and OnRobot RG-2 gripper control via `ur_rtde` and MuJoCo physics simulation, handling surgical tool delivery, camera jogging, and emergency safety halts.
  - **DICOM Image Viewer:** Offline intraoperative monitor display conforming to OHIF standards and IBM Carbon Design System (g100 Dark Theme) for hands-free medical scan navigation (CT, MRI, X-ray) and surgical milestone logging.

## NLU Intent Classification (Raspberry Pi 4)

A fine-tuned SetFit intent classification model (`setfit_bge-micro-v2_8k`) is included in `src/nlu/models/setfit_bge-micro-v2_8k/` for offline edge execution on Raspberry Pi 4 (ARM64).

### Quick Start on Raspberry Pi 4

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run benchmark and interactive tester:
   ```bash
   python test/test_rpi_intent.py
   ```

Detailed instructions and performance expectations are available in [docs/RPI4_DEPLOYMENT_GUIDE.md](docs/RPI4_DEPLOYMENT_GUIDE.md).

## Subsystem Benchmarks & Edge Evaluations

Comprehensive empirical benchmarks, latency budgets, metric explanations, and teammate recommendations measured on physical Raspberry Pi 4 hardware are documented in the **[`evaluations/`](evaluations/README.md)** directory:

- 🔊 [DSP Evaluation (`evaluations/dsp/`)](evaluations/dsp/README.md): Beamforming, DeepFilterNet RTF (0.449), Silero VAD.
- 🎙️ [ASR Evaluation (`evaluations/asr/`)](evaluations/asr/README.md): Vosk grammar FST latency (120ms), token posterior confidences.
- 🧠 [NLU Evaluation (`evaluations/nlu/`)](evaluations/nlu/README.md): SetFit ARM64 thread scaling (152ms), Draft 2020-12 schema validation.
- ⚡ [Orchestrator Evaluation (`evaluations/orchestrator/`)](evaluations/orchestrator/README.md): ZeroMQ IPC transit (2.68ms), priority preemption.
- 🔬 [Full Pipeline Evaluation (`evaluations/full-pipe/`)](evaluations/full-pipe/README.md): End-to-end post-speech latency (~293ms), clinical ergonomics.
