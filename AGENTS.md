# Agent Guidelines & Guardrails — scrub-voice

## Core Invariants
1. **No Structural Refactoring:** Do not create new root-level directories, rename existing folders, or alter the repository hierarchy (such as nesting `src/` or `data/`) unless explicitly requested by the user.
2. **Implementation Placement:** Place all implementation code inside the appropriate module package in `src/` (e.g., `src/actuators/`, `src/dsp/`, `src/nlu/`). Avoid creating loose `.py` scripts in the repository root.
3. **Model & Checkpoint Storage:** Store local model weights, checkpoints, and tokenizer configs inside a dedicated `models/` directory within the corresponding subsystem package under `src/` (e.g., `src/nlu/models/`, `src/dsp/models/`).
4. **Dynamic Path Resolution:** Always resolve file and asset paths dynamically relative to `Path(__file__).resolve()` or the repository root. Never hardcode absolute user or machine paths.
5. **Documentation Standards:** Place new guides and reference documentation in `docs/` following the uppercase naming convention (`LIKE_THIS.md`).
6. **Edge I/O & Flash Storage Protection (RAM-First Staging):** All audio recordings, benchmark reports, and test artifacts generated on edge hardware (Raspberry Pi 4) must stage to RAM (`/dev/shm`) first before auto-saving/copying to persistent storage (`here`). Never stream high-bandwidth raw audio or burst telemetry directly to the microSD card, which triggers kernel dirty writeback freezes and flash controller lockups.
7. **Audio Format, Bit Depth & DMA Alignment:** Always enforce 16-bit signed PCM audio (`subtype="PCM_16"`, `S16_LE` format, 16 kHz sample rate) rather than 32-bit float for stored WAVs. For ALSA multi-mic hardware capture (`arecord`), strictly configure power-of-two buffer and period parameters (e.g., `--buffer-size=65536 --period-size=8192`) aligned with hardware DMA descriptors to eliminate XRUNs and driver stalls.
8. **Audio Driver Isolation & Passive Telemetry:** Never dynamically import or initialize hardware-probing audio libraries (e.g., `sounddevice`, PortAudio `Pa_Initialize()`) for passive inspection, hardware info, or telemetry. Package versions must be resolved statically via `importlib.metadata.version()`. Hardware recording processes must terminate via graceful signals (`SIGINT`) to guarantee ALSA DMA buffer release.

## Reference Documentation
Consult the guides in `docs/` whenever relevant to your current task:
* `docs/REPO_ORGANIZATION.md`: Repository layout, folder responsibilities, and module boundaries.
* `docs/COMMAND_SCHEMA_GUIDE.md`: Intent and slot JSON schema contracts between NLU and robot actuators.
* `docs/RPI4_DEPLOYMENT_GUIDE.md`: Edge deployment instructions, environment setup, and benchmark scripts.
* `docs/BENCHMARK_SUITE_GUIDE.md`: Modular benchmark runners, architectural budgets, and statistical reporting.
* `docs/AUDIO_GUIDE.md`: Multi-mic hardware audio recording, ALSA mixer tuning, and playback.
* `docs/GITHUB_WORKFLOW.md`: Git branching, commit guidelines, and PR workflow.
