# NLU Subsystem: What Changed from Original Code

This document logs all architectural and algorithmic modifications made to the NLU subsystem in `Assemble_V1` relative to earlier prototype scripts.

---

## 1. Summary of Changes

| Component | Original Prototype State | Assemble_V1 Modified State | Motivation & Benefit |
| :--- | :--- | :--- | :--- |
| **Pipeline Integration** | Standalone SetFit model checkpoint in `models/` | Unified `VoiceToIntentPipeline` ([`src/nlu/pipeline.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/pipeline.py)) | Chains text normalization, SetFit inference, slot extraction, and schema validation into a single call. |
| **Execution Daemon** | None (offline test scripts only) | Production ZeroMQ service daemon ([`src/nlu/run_nlu.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/run_nlu.py)) | Consumes ASR transcripts and routes validated envelopes to normal or priority IPC channels. |
| **Safety Interlock** | All text routed through SetFit transformer | Fast-path regex interlock for emergency words (*"stop"*, *"halt"*, *"freeze"*) | Bypasses model latency; immediately dispatches `category="EMERGENCY"`, `priority="HIGH"`. |
| **Thread Parameterization** | Unconstrained PyTorch default (4 threads) | Explicit `torch.set_num_threads(2)` | Yields fastest execution time (152 ms, 25.5% faster) while reserving 2 cores for DSP/ASR. |
| **Signal Handling** | Default Python handlers | Early registration of `SIGINT` / `SIGTERM` before model loading | Prevents noisy `KeyboardInterrupt` tracebacks when supervisor stops the service during warmup. |
| **Envelope Ledger** | Ephemeral stdout printing | Persistent JSONL ledger (`results/nlu_envelopes.jsonl`) | Stores structured command envelopes for debugging and offline evaluation. |

---

## 2. Detailed Technical Modifications

### A. Unified Pipeline Implementation
* Created `VoiceToIntentPipeline` in [`src/nlu/pipeline.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/pipeline.py):
  1. Cleans input text via `TextNormalizer` (`src/nlu/normalizer.py`).
  2. Runs regex emergency safety interlock.
  3. Predicts intent and confidence via `SetFitModel`.
  4. Discards out-of-domain input (confidence $< 0.60 \rightarrow$ `intent="UNKNOWN"`).
  5. Extracts structured parameters via `SlotExtractor` (`src/nlu/slot_extractor.py`).
  6. Validates the 10-key envelope against Draft 2020-12 `schemas/command_schema.json`.

### B. Early Signal Registration Fix
* **Old Behavior:** In early assembly runs, killing the supervisor while NLU was loading weights resulted in an unhandled multi-line `KeyboardInterrupt` traceback during PyTorch tensor allocation.
* **New Behavior:** `signal.signal(signal.SIGINT, handle_signal)` and `signal.signal(signal.SIGTERM, handle_signal)` are registered at the very top of `main()` in [`src/nlu/run_nlu.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/nlu/run_nlu.py), ensuring clean, quiet exit.
