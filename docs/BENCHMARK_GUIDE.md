# Benchmark Guide — Modular Suite & Master Orchestrator

> **Project:** `scrub-voice` — Offline Edge Intraoperative Surgical Voice Assistant  
> **Target Hardware:** Raspberry Pi 4 Model B (Quad-core ARM Cortex-A72 @ 1.5 GHz, 4 GB RAM)  
> **Document Role:** Practical operational manual for executing the modular benchmark suite, interpreting metrics, running sweeps, and generating publication artifacts.

---

## Helper Execution Commands

```bash
# ------------------------------------------------------------------------------
# 1. Fast CI/CD Smoke Benchmark (< 5s runtime, no weights or microphones needed)
# ------------------------------------------------------------------------------
python test/benchmarks/bench_smoke.py

# With neural SetFit weight loading included:
python test/benchmarks/bench_smoke.py --with-nlu

# ------------------------------------------------------------------------------
# 2. Isolated Subsystem Benchmarks
# ------------------------------------------------------------------------------
# IPC: ZeroMQ transport latency, HWM socket saturation, priority preemption:
python test/benchmarks/bench_ipc.py --reps-ipc 500 --out results/ipc

# DSP: Beamforming & Denoising RTF, SI-SDR gain, and DOA angle estimation:
python test/benchmarks/bench_dsp.py --synthetic 10 --reps 50 --snr-sweep 20,15,10,5,0 --out results/dsp

# NLU: Intent accuracy, slot match, Wilson 95% CIs, and thread scaling sweep:
python test/benchmarks/bench_nlu.py --reps-nlu 5 --nlu-thread-sweep --out results/nlu

# ASR: Vosk constrained grammar vs open vocabulary WER and finalize latency:
python test/benchmarks/bench_asr.py --manifest data/bench/manifest.jsonl --asrs vosk_grammar,vosk_open --out results/asr

# ------------------------------------------------------------------------------
# 3. System-Level, Live & Soak Benchmarks
# ------------------------------------------------------------------------------
# End-to-End: Full pipeline post-speech latency across 5-way ablation matrix:
python test/benchmarks/bench_e2e.py --synthetic 10 --snr-sweep 15,10,5,0 --out results/e2e

# Live: Real-time 10-second capture from 6-mic array on Raspberry Pi 4:
python test/benchmarks/bench_live.py --live-seconds 10 --live-device 1 --live-channels 8 --out results/live

# Soak: 2-Hour continuous burn-in testing RSS memory leak and latency drift:
python test/benchmarks/bench_soak.py --soak-minutes 120 --synthetic 10 --out results/soak

# ------------------------------------------------------------------------------
# 4. Master Orchestrator (Full Academic Evaluation)
# ------------------------------------------------------------------------------
python test/benchmark_suite.py \
  --manifest data/bench/manifest.jsonl \
  --noise-wav data/noise/or_noise.wav \
  --scope both \
  --baselines \
  --seeds 3 \
  --tag paper_eval \
  --out results/full_eval
```

---

## 1. Suite Architecture & Runner Hierarchy

To allow rapid feedback during development, isolated stage profiling, and reproducible research benchmarks, the evaluation harness is partitioned into specialized modules:

```text
test/
├── benchmark_suite.py           # Unified Master Orchestrator (100% CLI compatibility)
└── benchmarks/
    ├── bench_smoke.py           # Fast CI/CD Smoke Test (< 5s runtime, synthetic audio)
    ├── bench_ipc.py             # ZeroMQ IPC Transport & Command Dispatcher microbenchmark
    ├── bench_dsp.py             # Isolated DSP Runner (Beamforming, Denoising, SI-SDR, DOA)
    ├── bench_nlu.py             # Isolated NLU Runner (SetFit, Slots, Schema, Thread Scaling)
    ├── bench_asr.py             # Isolated ASR Runner (Vosk Grammar/Open, Whisper, WER)
    ├── bench_e2e.py             # End-to-End Pipeline (Post-Speech Latency, Ablations, Paired Tests)
    ├── bench_live.py            # Live Hardware Capture & Real-Time Streaming Verification
    ├── bench_soak.py            # Multi-hour Stress & Memory Leak Burn-in Test
    └── common/
        ├── stats.py             # Bootstrapped CIs, Wilson score, Holm-Bonferroni, Wilcoxon
        ├── acoustics.py         # Plane-wave circular array simulation, synthetic audio, SI-SDR
        ├── telemetry.py         # Per-core CPU %, RSS tracking, SoC temp, throttle flags
        ├── registry.py          # Lazy component factory with load-time and RSS accounting
        ├── reporting.py         # LaTeX booktabs, Markdown tables, CDF figures, CSV/JSON
        └── rig.py               # In-process ZeroMQ PUSH/SUB Dispatcher fixture
```

---

## 2. Edge Performance Budgets & Real-Time Thresholds

Every runner checks observed measurements against the formal system budgets defined in `common/registry.py`:

| Metric | Budget | Description | Validated By |
| :--- | :--- | :--- | :--- |
| **DSP RTF (p95)** | $\le 0.50$ | Real-time factor per 0.5 s audio chunk | `bench_dsp`, `bench_e2e` |
| **ASR Finalize** | $\le 150\text{ ms}$ | Time from end-of-speech to final transcript | `bench_asr`, `bench_e2e` |
| **NLU Latency** | $\le 180\text{ ms}$ (Pi 4) | Embedding + SetFit classification + slot extraction | `bench_nlu`, `bench_e2e` |
| **IPC Transit** | $\le 5.0\text{ ms}$ | ZeroMQ PUSH $\rightarrow$ PULL one-way transit | `bench_ipc` |
| **Dispatcher Round-Trip** | $\le 5.0\text{ ms}$ | NLU PUSH $\rightarrow$ Command Bus PUB | `bench_ipc`, `bench_e2e` |
| **E2E Post-Speech** | $\le 500\text{ ms}$ | Last chunk DSP + ASR finalize + NLU + IPC | `bench_e2e`, `bench_live` |
| **Peak Memory (RSS)** | $\le 800\text{ MB}$ | Total system resident memory footprint | `bench_soak`, `benchmark_suite` |
| **Memory Leak Drift** | $< 5.0\text{ MB/h}$ | Linear regression slope over process RSS | `bench_soak` |
| **Emergency Rank** | Rank 0 | Emergency commands preempt queued normal commands | `bench_ipc` |

---

## 3. Subsystem Benchmark Runners

### 3.1 Fast CI/CD Smoke Test (`bench_smoke.py`)
Validates acoustic plane-wave simulation, circular array steering, ZeroMQ dispatcher round-trip, statistical mathematics, and lexical normalization in **$< 5$ seconds** without requiring soundcards or model weights.

```bash
# Standard smoke check (< 5s):
python test/benchmarks/bench_smoke.py

# Include full SetFit neural weight loading:
python test/benchmarks/bench_smoke.py --with-nlu
```

### 3.2 IPC Transport & Dispatcher Microbenchmark (`bench_ipc.py`)
Evaluates POSIX IPC (`ipc://`), TCP loopback (`tcp://127.0.0.1`), and raw `AF_UNIX` socketpairs across command envelopes (256 B), small payloads (4 KB), and audio chunks (32 KB):
* **One-Way Transit Latency:** Mean, p50, p95, p99 across `--reps-ipc` iterations.
* **Throughput:** Sustained message rate (msgs/sec) under burst pressure.
* **Emergency Preemption Test:** Injects 1 emergency command behind 45 queued normal commands before starting dispatcher; passes if emergency command is published with Rank 0.
* **HWM Burst & Drop Policy:** Tests socket resilience under 200 non-blocking sends.
* **Malformed Payload Containment:** Sends 8 corrupt envelopes (truncated, bad types, missing keys) and asserts all route to `ERROR` topic with zero leaks.

```bash
python test/benchmarks/bench_ipc.py --reps-ipc 500 --out results/ipc
```

### 3.3 Isolated DSP Benchmark (`bench_dsp.py`)
Evaluates Delay-and-Sum beamforming (`ref_channel`, `ref_srp_das`, `repo_beamformer`) and neural denoising (`passthrough`, `spectral_gate`, `repo_denoiser`):
* **Streaming Timing:** Chunks audio into 0.5 s frames; calculates RTF, deadline miss rate (RTF $> 1.0$), and CPU-to-wall core utilization.
* **Quality Metrics:**
  * $\Delta$SI-SDR improvement (dB) over reference microphone.
  * Short-Time Objective Intelligibility (STOI, if `pystoi` is installed).
  * Direction of Arrival (DOA) error in degrees and % within $15^\circ$.

```bash
# Synthetic array audio with full SNR sweep:
python test/benchmarks/bench_dsp.py --synthetic 10 --reps 50 --snr-sweep 20,15,10,5,0

# Real multi-channel manifest and hospital noise:
python test/benchmarks/bench_dsp.py --manifest data/bench/manifest.jsonl --noise-wav data/noise/or_noise.wav
```

### 3.4 Isolated NLU Benchmark (`bench_nlu.py`)
Evaluates SetFit intent classification, slot extraction, and schema compliance:
* **Metrics:** Intent accuracy, slot exact match, OOD False Accept Rate (FAR), and In-Domain False Reject Rate (FRR) with Wilson 95% score intervals.
* **Thread Scaling Sweep (`--nlu-thread-sweep`):** Profiles inference latency across 1 thread (core-pinned), 2 threads, and 3 threads.
* **Paired Baseline Test:** Runs Wilcoxon signed-rank test against TF-IDF baseline if `--nlu-train` is provided.

```bash
python test/benchmarks/bench_nlu.py --reps-nlu 5 --nlu-thread-sweep --out results/nlu
```

### 3.5 Isolated ASR Benchmark (`bench_asr.py`)
Evaluates Vosk constrained grammar vs unconstrained open vocabulary:
* **Metrics:** Real-Time Factor (RTF), finalization latency (ms), and Word Error Rate (WER) with 1,000-sample bootstrap confidence intervals across clean and noisy conditions.

```bash
python test/benchmarks/bench_asr.py --manifest data/bench/manifest.jsonl --asrs vosk_grammar,vosk_open
```

---

## 4. System-Level & Operational Runners

### 4.1 End-to-End Pipeline & Ablation Matrix (`bench_e2e.py`)
Streams 0.5 s chunks concurrently through DSP, feeds chunks into ASR, and on the final chunk triggers ASR finalization, NLU classification, and Dispatcher round-trip:
* **Ablation Variants:**
  * `ours`: DAS beamformer + Stateful DuoFilter + Vosk grammar + SetFit
  * `ablate_no_dsp`: Raw Mic 0 + Passthrough + Vosk grammar + SetFit
  * `ablate_bf_only`: DAS beamformer + Passthrough + Vosk grammar + SetFit
  * `ablate_dn_only`: Raw Mic 0 + Stateful DuoFilter + Vosk grammar + SetFit
  * `base_deepfilternet`: DeepFilterNet denoiser baseline
  * `base_vosk_openvocab`: Unconstrained open-vocabulary baseline
  * `base_whisper`: OpenAI Whisper `tiny.en` baseline
* **Statistical Rigor:** Performs paired Wilcoxon signed-rank tests across identical noise seeds with Holm-Bonferroni correction.

```bash
python test/benchmarks/bench_e2e.py --synthetic 10 --snr-sweep 15,10,5,0 --out results/e2e
```

### 4.2 Live Hardware Capture (`bench_live.py`)
Runs directly on the Raspberry Pi 4 equipped with the ReSpeaker 6-mic circular array:
* Opens `sounddevice.InputStream` for `--live-seconds` (default 10 s).
* Processes 0.5 s frames in real time, asserting queue backlog $\le 1$ and zero input buffer overflows.
* Automatically writes the raw capture to `live_capture.wav` so the live acoustic session can be replayed offline.

```bash
python test/benchmarks/bench_live.py --live-seconds 10 --live-device 1 --live-channels 8
```

### 4.3 Multi-Hour Soak & Memory Leak Test (`bench_soak.py`)
Continuous streaming burn-in test designed to verify clinical reliability over extended surgical durations:
* **Memory Leak Detection:** Calculates the linear regression slope ($\text{MB/h}$) of resident memory (RSS). Passes if slope $< 5.0\text{ MB/h}$ and peak RSS $\le 800\text{ MB}$.
* **Latency Drift:** Calculates latency slope ($\text{ms/h}$) to verify no queuing backlogs form over time.
* **Thermal & Hardware Telemetry:** Continuously queries SoC core temperature and `vcgencmd get_throttled` for under-voltage or thermal throttling flags.

```bash
python test/benchmarks/bench_soak.py --soak-minutes 120 --synthetic 10 --out results/soak
```

---

## 5. Master Orchestrator (`test/benchmark_suite.py`)

The root `benchmark_suite.py` orchestrates any combination of stages and generates publication-grade tables and plots:

```bash
python test/benchmark_suite.py \
  --manifest data/bench/manifest.jsonl \
  --noise-wav data/noise/or_noise.wav \
  --scope both \
  --baselines \
  --seeds 3 \
  --tag paper_eval \
  --out results/full_eval
```

### Key Execution Options:
* `--stages <dsp,asr,nlu,ipc>`: Comma-separated list of isolated stages to run (default: `all`).
* `--scope <isolated|e2e|both>`: Run isolated subsystem benchmarks, end-to-end pipeline, or both.
* `--baselines`: Automatically include external comparison engines (`deepfilternet`, `whisper`, `vosk_open`, `tfidf_logreg`).
* `--seeds <N>`: Number of independent noise realizations and speaker azimuth angles (default: 3).
* `--soak-minutes <M>`: Attach a soak test of $M$ minutes to the end of the run.

---

## 6. Interpreting Benchmark Output Artifacts

All outputs are saved to `results/bench_<timestamp>[_tag]/`:

| Output File | Format | Purpose & Contents |
| :--- | :--- | :--- |
| **`report.json`** | JSON | Machine-readable results: hardware info, arguments, per-stage percentiles, confidence intervals, and budget PASS/FAIL flags. |
| **`raw_<stage>.csv`** | CSV | Per-sample raw measurements for plotting and regression in R or pandas. |
| **`table_budgets.tex`** | LaTeX | Booktabs formatted table verifying all subsystem metrics against edge hardware budgets. |
| **`table_ablation.tex`** | LaTeX | Statistical comparison of pipeline variants with paired Wilcoxon $p$-values. |
| **`tables.md`** | Markdown | Human-readable Markdown tables summarizing all experimental findings. |
| **`fig_e2e_latency_cdf.png`** | PNG / PDF | Cumulative distribution plot of post-speech latency across variants with the 500 ms budget line. |
| **`fig_wer_vs_snr.png`** | PNG / PDF | Multi-curve graph showing Word Error Rate resilience across SNR degradation. |
| **`live_capture.wav`** | WAV | Raw multi-channel recording saved during `--mode live` runs. |

