# Agent Guidelines & Guardrails — scrub-voice

## Core Invariants
1. **No Structural Refactoring:** Do not create new root-level directories, rename existing folders, or alter the repository hierarchy (such as nesting `src/` or `data/`) unless explicitly requested by the user.
2. **Implementation Placement:** Place all implementation code inside the appropriate module package in `src/` (e.g., `src/actuators/`, `src/dsp/`, `src/nlu/`). Avoid creating loose `.py` scripts in the repository root.
3. **Model & Checkpoint Storage:** Store local model weights, checkpoints, and tokenizer configs inside a dedicated `models/` directory within the corresponding subsystem package under `src/` (e.g., `src/nlu/models/`, `src/dsp/models/`).
4. **Dynamic Path Resolution:** Always resolve file and asset paths dynamically relative to `Path(__file__).resolve()` or the repository root. Never hardcode absolute user or machine paths.
5. **Documentation Standards:** Place new guides and reference documentation in `docs/` following the uppercase naming convention (`LIKE_THIS.md`).

## Reference Documentation
Consult the guides in `docs/` whenever relevant to your current task:
* `docs/REPO_ORGANIZATION.md`: Repository layout, folder responsibilities, and module boundaries.
* `docs/COMMAND_SCHEMA_GUIDE.md`: Intent and slot JSON schema contracts between NLU and robot actuators.
* `docs/RPI4_DEPLOYMENT_GUIDE.md`: Edge deployment instructions, environment setup, and benchmark scripts.
* `docs/AUDIO_GUIDE.md`: Multi-mic hardware audio recording, ALSA mixer tuning, and playback.
* `docs/GITHUB_WORKFLOW.md`: Git branching, commit guidelines, and PR workflow.
