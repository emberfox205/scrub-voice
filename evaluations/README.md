# Scrub-Voice: Subsystem Evaluations & Benchmark Index

Empirical edge benchmarks, metric budgets, and code modification logs for the **Scrub-Voice** intraoperative edge pipeline.

**Target Hardware:** Raspberry Pi 4 Model B (Quad-Core Cortex-A72 @ 1.5 GHz, 4 GB RAM, aarch64 OS `admin@PI4WURST`).

---

## 1. Edge Benchmark Dashboard

| Subsystem | Primary Engine | Mean Latency | P95 Latency | RTF / Rate | Peak RAM | Target Budget | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[DSP](dsp/)** | Delay-and-Sum + DeepFilterNet | 448.97 ms | 458.35 ms | **0.4490** | 629.6 MB | $\text{RTF} \le 0.50$ | **PASS** |
| **[ASR](asr/)** | Vosk Grammar (Kaldi FST) | 120.00 ms | 140.00 ms | ~0.1200 | 185.0 MB | Latency $\le 200\text{ ms}$ | **PASS** |
| **[NLU](nlu/)** | SetFit `bge-micro-v2` (2 Thr) | 152.85 ms | 155.77 ms | — | 635.5 MB | Latency $\le 200\text{ ms}$ | **PASS** |
| **[Orchestrator](orchestrator/)** | ZeroMQ IPC Dual-Channel | 2.68 ms | 2.80 ms | >10k msg/s | 38.0 MB | Transit $\le 5.0\text{ ms}$ | **PASS** |
| **[Full Pipeline](full-pipe/)** | End-to-End Post-Speech | **290.53 ms** | **316.92 ms** | — | **635.5 MB** | Total $\le 500\text{ ms}$ | **PASS** |

---

## 2. Directory & Documentation Map

Each module folder contains:
1. `README.md`: Concise empirical metrics, hardware profiles, and teammate watch-outs.
2. `CHANGES.md`: Detailed changelog explaining what was changed/created relative to earlier prototype code.

| Subsystem | Metrics & Evals | What Changed from Old Code | Key Hardware Finding |
| :--- | :--- | :--- | :--- |
| **DSP** | [DSP Metrics](dsp/README.md) | [DSP Changes](dsp/CHANGES.md) | RTF 0.449; VAD state reset fixes sticky utterance clumping |
| **ASR** | [ASR Metrics](asr/README.md) | [ASR Changes](asr/CHANGES.md) | 120ms latency; 100% keyword conf; grammar expansion has zero delay |
| **NLU** | [NLU Metrics](nlu/README.md) | [NLU Changes](nlu/CHANGES.md) | 2 threads is optimal (152ms, -25.5%); 100% schema compliance |
| **Orchestrator** | [Orchestrator Metrics](orchestrator/README.md) | [Orchestrator Changes](orchestrator/CHANGES.md) | 2.68ms IPC transit; 358ms watchdog crash auto-recovery |
| **Full Pipeline** | [Full-Pipe Metrics](full-pipe/README.md) | [Full-Pipe Changes](full-pipe/CHANGES.md) | 290ms post-speech response matches human conversation (~250ms) |

---

## 3. How to Run Edge Tests & Benchmarks

The test suite provides two complementary testing workflows:

### A. Subsystem Benchmark & Resource Profiler (`test/benchmark_edge_profile.py`)
Profiles isolated module performance (latency, RAM, Real-Time Factor, and thread scaling) without launching background daemons.

```bash
# Recommended baseline: 2 threads on Raspberry Pi 4 (without pinning)
uv run python test/benchmark_edge_profile.py --threads 2 --no-pin

# Benchmark denoiser using any custom audio file (interchangeable WAV path)
uv run python test/benchmark_edge_profile.py --threads 2 --no-pin --wav data/asr/test_mix_voice.wav
```

* **CLI Options:**
  * `--threads 2`: Allocates 2 PyTorch/ONNX worker threads (empirically proven optimal for Cortex-A72).
  * `--no-pin`: Disables physical core 0 affinity pinning (recommended when testing alongside other workloads).
  * `--wav <PATH>`: **Interchangeable WAV file path.** Loads any custom audio file to calculate real-world RTF on actual acoustic data instead of default noise audio.
  * `--iterations <N>`: Repetition count per sample vector (default: 5).
* **Outputs:**
  * Live console summary table with pass/fail threshold status.
  * Machine-readable telemetry saved to [`results/benchmark_edge_profile.json`](../results/benchmark_edge_profile.json).

---

### B. End-to-End Live Pipeline Simulation (`src/orchestrator/supervisor.py`)
Spawns all 4 decoupled microservices ([MONITOR], [NLU], [ASR], [DSP]), feeds simulated multi-channel or mono audio in real time, and renders live actions on the intraoperative DICOM dashboard.

```bash
# Stream custom audio in single-pass mode and dump 3-stage diagnostic WAVs
uv run python src/orchestrator/supervisor.py \
  --wav data/asr/test_mix_voice.wav \
  --single-pass \
  --dump-audio results/dsp_debug
```

* **CLI Options:**
  * `--wav <PATH>`: **100% interchangeable.** Streams any custom 16kHz WAV file into the DSP daemon as if it were a physical microphone array.
  * `--single-pass`: Automatically shuts down all microservices when the WAV file finishes playing (instead of looping indefinitely).
  * `--dump-audio <DIR>`: Simultaneously writes all 3 progressive audio stages for every detected utterance:
    1. `utterance_XXX_1_raw_mic1.wav`: Unprocessed single microphone capture.
    2. `utterance_XXX_2_beamformed.wav`: Constructively summed 6-mic directional beam.
    3. `utterance_XXX_3_denoised.wav`: DeepFilterNet suppressed speech sent to ASR.
* **Outputs & Live Inspection:**
  * Browser UI: Open `http://localhost:8000` (or `http://<PI_IP>:8000`) for the live DICOM viewer.
  * ASR Transcripts: Word tokens and confidences append to [`results/asr_transcripts.jsonl`](../results/asr_transcripts.jsonl).
  * NLU Envelopes: Validated command envelopes append to [`results/nlu_envelopes.jsonl`](../results/nlu_envelopes.jsonl).
  * Audio Verification: Inspect the 3 progressive WAV files in `results/dsp_debug/` with any media player or Audacity.
