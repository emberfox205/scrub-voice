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

## 3. Environment Setup & Model Weights (Prerequisites)

Because model checkpoints (`*.safetensors`) are heavy binary files (~69.5 MB), they are excluded from Git via `.gitignore`. Team members setting up a new checkout or pulling updates must synchronize dependencies and download the NLU weights before executing benchmarks or starting the supervisor:

### Step 1: Synchronize Virtual Environment
```bash
# Resolves and installs pinned dependencies from pyproject.toml / uv.lock
uv sync
```

### Step 2: Download NLU Transformer Weights (`model.safetensors`)
Download the fine-tuned SetFit transformer checkpoint directly into `src/nlu/models/setfit_bge-micro-v2_8k/`:

* **Google Drive URL:** [model.safetensors](https://drive.google.com/file/d/1nvYNYkfLq-hNlU9LnwKtE8j2GsXd3fVn/view?usp=drive_link)
* **File ID:** `1nvYNYkfLq-hNlU9LnwKtE8j2GsXd3fVn`

```bash
# Using uvx (runs gdown in an isolated temporary tool environment):
uvx gdown 1nvYNYkfLq-hNlU9LnwKtE8j2GsXd3fVn -O src/nlu/models/setfit_bge-micro-v2_8k/model.safetensors

# Alternatively using standard pip / gdown:
pip install gdown
gdown 1nvYNYkfLq-hNlU9LnwKtE8j2GsXd3fVn -O src/nlu/models/setfit_bge-micro-v2_8k/model.safetensors
```

---

## 4. How to Run Edge Tests & Benchmarks

The test suite provides two complementary testing workflows:

### A. Subsystem Benchmark & Resource Profiler (`test/benchmark_edge_profile.py`)
Profiles isolated module performance (latency, RAM, Real-Time Factor, and thread scaling) without launching background daemons.

```bash
# Recommended baseline: 2 threads on Raspberry Pi 4 with DuoFilter (default)
uv run python test/benchmark_edge_profile.py --threads 2 --no-pin

# Benchmark DuoFilter using custom audio file
uv run python test/benchmark_edge_profile.py --threads 2 --no-pin --wav data/asr/test_mix_voice.wav

# Benchmark DeepFilterNet using custom audio file
uv run python test/benchmark_edge_profile.py --threads 2 --no-pin --denoiser-type deepfilter --wav data/asr/test_mix_voice.wav
```

* **CLI Options:**
  * `--denoiser-type <duofilter|deepfilter>`: Selects neural denoiser model: `duofilter` (default) or `deepfilter`.
  * `--threads 2`: Allocates 2 PyTorch/ONNX worker threads (empirically proven optimal for Cortex-A72).
  * `--no-pin`: Disables physical core 0 affinity pinning (recommended when testing alongside other workloads).
  * `--wav <PATH>`: **Interchangeable WAV file path.** Loads any custom audio file to calculate real-world RTF on actual acoustic data instead of default noise audio.
  * `--iterations <N>`: Repetition count per sample vector (default: 5).
* **Outputs:**
  * Live console summary table with pass/fail threshold status.
  * Machine-readable telemetry saved to `results/benchmark_edge_profile.json`.

---

### B. End-to-End Live Pipeline Simulation (`src/orchestrator/supervisor.py`)
Spawns all 4 decoupled microservices ([MONITOR], [NLU], [ASR], [DSP]), feeds simulated multi-channel or mono audio in real time, and renders live actions on the intraoperative DICOM dashboard.

```bash
# Stream custom audio in single-pass mode and dump 3-stage diagnostic WAVs
uv run python src/orchestrator/supervisor.py \
  --wav data/asr/test_mix_voice.wav \
  --single-pass \
  --dump-audio results/dsp_debug

# 1. Run live pipeline simulation with DuoFilter (default)
uv run python src/orchestrator/supervisor.py --wav data/asr/test_mix_voice.wav --single-pass

# 2. Run live pipeline simulation with DeepFilterNet
uv run python src/orchestrator/supervisor.py --wav data/asr/test_mix_voice.wav --single-pass --denoiser-type deepfilter

# 3. Run DSP standalone with DuoFilter
uv run python src/dsp/run_dsp.py --wav data/asr/sample_01.wav --denoiser-type duofilter

# 4. Run DSP standalone with DeepFilterNet
uv run python src/dsp/run_dsp.py --wav data/asr/sample_01.wav --denoiser-type deepfilter
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
  * ASR Transcripts: Word tokens and confidences append to `results/asr_transcripts.jsonl`.
  * NLU Envelopes: Validated command envelopes append to `results/nlu_envelopes.jsonl`.
  * Audio Verification: Inspect the 3 progressive WAV files in `results/dsp_debug/` with any media player or Audacity.

---

## 5. Subsystem Evaluations & Clinical Findings (by HerrCult69)

Direct operational evaluation across all subsystems after live hardware execution on the complete, integrated pipeline:

### 🔊 1. DSP Audio Subsystem
* **Beamforming Reality:**
  * In current WAV simulation trials, mono audio is duplicated across 8 channels. Because all mics receive identical waveforms ($\Delta \tau = 0$), the Delay-and-Sum beamformer produces an output identical to the raw input.
  * *Next step:* True directional beamforming must be evaluated using physical 8-channel array capture (`arecord -c 8`) with spatial speaker offsets.
* **Denoiser Acoustic Coloration (Underwhelming):**
  * The DeepFilterNet neural denoiser output is currently underwhelming: static background noise is only moderately reduced, while the voice suffers from a "vacuum tube" coloration (flanging/comb filtering), fading in and out with subtle popping sounds (like speaking through a hollow pipe).
  * *Technical cause:* Aggressive ERB subband neural gain masking on reverberant audio creates spectral gating artifacts.
  * *Required fix:* Clamp maximum attenuation to a bounded floor (e.g., $-12\text{ dB}$ or $-15\text{ dB}$) and blend $20\%$ raw audio with $80\%$ denoised audio to preserve vocal timbre.
* **Acoustic Silence Detection Limitation:**
  * Ambient operating room noise (HVAC, suction, ventilator beeps) makes pure energy-based VAD silence detection difficult to control—risking either cutting off trailing words or introducing 400–800 ms of silence delay.

### 🎙️ 2. ASR Speech Recognition
* **Narrow Grammar vs. Phrasing Freedom:**
  * The current grammar is too restrictive. Surgeons lack the freedom to use colloquial medical phrasing and synonyms that carry identical clinical intent (*"bring up"*, *"pull up"*, *"scroll"*, *"advance"*, *"freeze"*, *"hold on"*, *"closer"*).
  * The grammar FST must be expanded by 30–50 medical synonyms. Because Kaldi uses an FST graph, this incurs **$< 10\text{ ms}$** of decoding overhead on the Pi 4 while granting surgeons full conversational freedom.
* **Implementation of the Explicit "END" Keyword (Who Does It?):**
  * **ASR (`src/asr/`) must implement this feature.**
  * *Why DSP cannot do it:* Silero VAD only sees raw decibels and voice probability ($P(\text{speech})$); it has zero vocabulary or speech recognition capability.
  * *Why NLU is too late:* NLU only receives text after ASR finishes; if ASR waited for silence, the latency penalty has already occurred.
  * *ASR Implementation Blueprint:* Add `"end"` to `src/asr/vocab.py`. In the streaming audio loop, check `recognizer.PartialResult()`. The instant `"end"` is detected as the trailing word, ASR immediately flushes `FinalResult()`, strips the delimiter, and pushes the command to NLU over ZeroMQ. This slashes post-speech latency from $\sim 290\text{ ms}$ down to **$\mathbf{< 150\text{ ms}}$**.

### 🧠 3. NLU Intent Engine
* **Evaluation:** Working very well and reliably.
* Intent classification (SetFit `bge-micro-v2`) and slot extraction reliably produce 100% valid Draft 2020-12 envelopes and correctly reject out-of-domain conversational chatter as `UNKNOWN`.
* Thread profiling confirmed that **2 PyTorch threads** is the optimal operating point on ARM Cortex-A72 ($152.85\text{ ms}$, leaving 2 cores 100% free for DSP and ASR).

### ⚡ 4. Orchestrator & Intraoperative Web Monitor
* **Dispatcher & Watchdog:** ZeroMQ dual-channel priority routing and supervisor watchdog recovery ($358\text{ ms}$) are verified and robust.
* **Monitor Capabilities Tested & Next Testing Priorities:**
  * Viewport magnification (`zoom_display`) is tested and working properly.
  * Testing now needs to expand across the remaining capabilities supported by the monitor:
    1. **`navigate_slice`:** Relative stepping (*"advance 5 slices"*, *"previous slice"*) and direct jumping (*"jump to slice 45"*).
    2. **`open_scan` / `close_scan`:** Modality switching between CT, MRI, and X-ray series.
    3. **`log_event`:** EMR intraoperative milestone audit logging (*"record milestone trocar placed note clip attached"*).
    4. **`pick_tool` / `park_arm`:** Robotic gripper holding telemetry (*"hand me the scalpel"*).
    5. **`halt_emergency` / `clear_emergency`:** Brake-locking emergency stops and +Z vertical cavity evacuation.

