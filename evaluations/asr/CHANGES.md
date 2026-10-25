# ASR Subsystem: What Changed from Original Code

This document logs all architectural and algorithmic modifications made to the ASR subsystem in `Assemble_V1` relative to earlier prototype scripts.

---

## 1. Summary of Changes

| Component | Original Prototype State | Assemble_V1 Modified State | Motivation & Benefit |
| :--- | :--- | :--- | :--- |
| **Pipeline Integration** | Standalone batch script (`src/asr/scripts/run_asr.py --wav <FILE>`) | Continuous ZeroMQ IPC daemon (`ipc:///tmp/scrub_audio.ipc` $\rightarrow$ `ipc:///tmp/scrub_asr.ipc`) | Live streaming connection between DSP and NLU microservices. |
| **Utterance Flushing** | Fixed silence timeout fallback ($\sim 800\text{ ms}$ delay) | Event-driven instant flush upon DSP `{"type": "end"}` signal | Eliminates 800 ms of post-speech latency, decoding in **$119 - 140\text{ ms}$**. |
| **Acoustic Telemetry** | Plain text string output (`"zoom in two times"`) | Word token confidence extraction (`Tokens: [zoom:1.00, in:1.00, two:0.50, times:1.00]`) | Exposes posterior alignment probabilities and word boundaries (`start`, `end`). |
| **Transcript Logging** | Ephemeral stdout printing | Persistent JSONL ledger (`results/asr_transcripts.jsonl`) | Stores detailed word-level diagnostic data for offline vocabulary mining. |
| **Dual-Mode Engine** | Grammar-only decoding | Dual-mode: Grammar FST primary + Freeform fallback logging | Records out-of-grammar clinical speech while preserving low-latency command execution. |

---

## 2. Detailed Technical Modifications

### A. Zero-Lag Framing Synchronization
* **Old Behavior:** In standalone mode, `KaldiRecognizer.FinalResult()` was only invoked after an audio file EOF or when internal Kaldi silence buffers timed out after $\sim 0.8\text{ s}$.
* **New Behavior:**
  * When `src/dsp/run_dsp.py` detects utterance completion via Silero VAD, it transmits an end-of-speech control frame: `{"type": "end", "timestamp": ...}`.
  * `src/asr/scripts/run_asr.py` immediately intercepts this signal, calls `FinalResult()`, and flushes the completed transcript downstream to NLU within **$\sim 120\text{ ms}$**.

### B. Per-Token Confidence & Word Timestamps
* Integrated Kaldi word lattice extraction:
  ```json
  {
    "text": "zoom in two times",
    "normalized": "zoom in 2 times",
    "confidence": 0.875,
    "words": [
      {"word": "zoom", "start": 0.0, "end": 0.39, "confidence": 1.0},
      {"word": "in", "start": 0.39, "end": 0.57, "confidence": 1.0},
      {"word": "two", "start": 0.57, "end": 0.84, "confidence": 0.5},
      {"word": "times", "start": 0.87, "end": 1.59, "confidence": 1.0}
    ]
  }
  ```
* Appended automatically to `results/asr_transcripts.jsonl`.
