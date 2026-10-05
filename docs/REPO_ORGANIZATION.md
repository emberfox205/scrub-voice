# Repository Organization & Directory Structure

This document defines the canonical folder hierarchy and placement rules for the `scrub-voice` repository. It serves as an authoritative guide for both human contributors and AI coding assistants to maintain structural integrity.

---

## 1. High-Level Repository Tree

```text
ur_arm/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── asr/
│   └── dsp/
├── docs/
├── proposal/
├── reports/
├── schemas/
├── src/
│   ├── actuators/
│   │   └── sim_ws/
│   ├── dsp/
│   │   ├── source/
│   │   └── test/
│   └── nlu/
│       └── models/
└── test/
```

---

## 2. Directory Responsibilities & Placement Guidelines

### `data/`
* **Purpose:** Static audio assets and multi-channel microphone test datasets used for offline DSP and ASR verification.
* **Subdirectories:**
  * `asr/`: Sample single-channel speech commands (WAV format) for transcription validation.
  * `dsp/`: Multi-channel recordings from circular microphone arrays across various SNR and diffuse noise configurations.
* **Rule:** Generated/processed audio outputs must not be committed to Git; keep them directed to `.gitignore`d output paths (`data/dsp/output/`).

### `docs/`
* **Purpose:** Technical guides, hardware deployment manuals, workflow policies, and project documentation.
* **Rule:** All markdown documentation files (except the root `README.md`) must reside here and follow the uppercase naming convention (`LIKE_THIS.md`).

### `proposal/`
* **Purpose:** Formal academic project proposal files, including LaTeX sources (`.tex`), graphics, and compiled presentation PDFs.

### `reports/`
* **Purpose:** Internal development notes, sprint summaries, technical source of truth (SoT) documents, and weekly report materials.

### `schemas/`
* **Purpose:** Machine-readable data contract definitions (JSON Schema format).
* **Rule:** Defines valid payload structures exchanged between pipeline stages (e.g., `command_schema.json` between NLU and robot actuators).

### `src/`
* **Purpose:** Core Python source code for the edge voice-to-intent pipeline.
* **Subdirectories:**
  * `actuators/`: Hardware actuation interfaces (Universal Robots `ur_rtde` controller bindings, gripper controls, and Cartesian trajectory routines).
    * `sim_ws/`: Docker Compose testbed for the official URSim simulator and virtual verification scripts.
  * `dsp/`: Digital Signal Processing algorithms (delay-and-sum beamforming, super-directivity, PHAT whitening, and VAD gating).
    * `source/`: Low-level hardware audio streaming interfaces (`arecord`) and voice activity detector wrappers.
    * `test/`: Subsystem unit tests for streaming buffers and DSP modules.
  * `nlu/`: Natural Language Understanding logic (intent prediction, slot extraction, confidence scoring).
    * `models/`: Local edge model directories containing fine-tuned checkpoints and tokenizer configurations (e.g., `setfit_bge-micro-v2_8k/`).

### `test/`
* **Purpose:** Top-level end-to-end testing, hardware benchmark runners, and edge performance profiling scripts (e.g., `test_rpi_intent.py` for Raspberry Pi 4).

---

## 3. Rules of Engagement for AI Agents

1. **No Unprompted Structural Refactoring:**  
   Do not create new top-level directories, nest `src/` or `data/` into subfolders, or move modules around unless explicitly commanded by the user.

2. **Source Code Placement:**  
   Always place new implementation code inside the appropriate `src/<module>/` package. Do not create loose standalone `.py` scripts in the repository root.

3. **Dynamic Path Resolution:**  
   Never hardcode absolute paths or assume a fixed working directory. Always resolve file paths relative to `Path(__file__).resolve()` or dynamically calculate `repo_root`.

4. **Model Artifacts Location:**  
   Edge models and weights belong strictly inside `src/nlu/models/<model_name>/`.

5. **Markdown Placement:**  
   Do not add new `.md` files at the root level. Place them in `docs/` using uppercase naming.
