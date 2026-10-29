# DSP Subsystem: What Changed from Original Code

This document logs all architectural and algorithmic modifications made to the DSP subsystem in `Assemble_V1` relative to earlier prototype scripts.

---

## 1. Summary of Changes

| Component | Original Prototype State | Assemble_V1 Modified State | Motivation & Benefit |
| :--- | :--- | :--- | :--- |
| **Execution Architecture** | Loose, standalone scripts (`stream_audio.py`, `denoise.py`, `vad.py`) | Unified streaming daemon ([`src/dsp/run_dsp.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/run_dsp.py)) | Enables continuous end-to-end multi-process execution via ZeroMQ IPC (`ipc:///tmp/scrub_audio.ipc`). |
| **VAD Temporal Segmentation** | Single-window thresholding (0.35 prob); no hidden state clearing | Multi-window voting ($\ge 3$ active windows) + `model.reset_states()` on silence | Solved "sticky utterance clumping" where speech non-stop clustered into one giant segment. |
| **Channel Layout & Routing** | Undefined channel mapping | Explicit 8-ch capture: strips AEC loopbacks (ch 6-7), routes perimeter mics (ch 0-5) | Ensures phase alignment for circular beamformer ($r=43\text{ mm}$). |
| **Diagnostic Tooling** | None (audio discarded in memory) | 3-stage audio dumping via `--dump-audio <DIR>` | Simultaneously writes raw mic 1, beamformed, and denoised WAVs for acoustic validation. |
| **ASR Synchronization** | Fixed-length audio chunking | Event-driven `{"type": "end"}` framing signal sent over IPC | Allows downstream Vosk engine to flush speech immediately upon silence. |

---

## 2. Detailed Technical Modifications

### A. VAD "Sticky Clumping" Bug Fix
* **Old Behavior:** In [`src/dsp/source/vad.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/vad.py), any single 32 ms window exceeding 0.35 probability marked the entire 500 ms audio chunk as speech. Because Silero's internal LSTM hidden state was never cleared, noise energy accumulated, causing chunks 4 through 20 to be marked as speech continuously.
* **New Behavior:**
  1. Implemented window-based voting: an utterance starts only if $\ge 3$ consecutive windows (approx. 100 ms) exceed threshold, or if mean chunk probability $\ge 0.50$.
  2. Added `self.model.reset_states()` on speech-to-silence transitions to wipe accumulated recurrent hidden state tensors.
  3. Increased default detection threshold to `0.50`.

### B. 3-Stage Audio Diagnostic Dumper
* **Added to [`src/dsp/run_dsp.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/run_dsp.py):**
  When `--dump-audio <DIR>` is passed, the daemon automatically writes:
  * `utterance_XXX_1_raw_mic1.wav`: Unprocessed single microphone capture.
  * `utterance_XXX_2_beamformed.wav`: Constructively summed 6-mic directional audio.
  * `utterance_XXX_3_denoised.wav`: Neural noise suppressed output forwarded to ASR.
  * Live dB attenuation is computed and logged per chunk (`Denoise: -XX.X dB`).

### C. Clean Signal Teardown
* Added `signal.SIGINT` and `signal.SIGTERM` handlers to cleanly close ALSA capture streams and terminate ZeroMQ sockets without leaving orphaned file handles.

