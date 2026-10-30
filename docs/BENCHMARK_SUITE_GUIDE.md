# SCRUB-VOICE Modular Benchmark Suite Guide

## 1. Overview & Architecture

The SCRUB-VOICE benchmark suite provides research-grade evaluation for offline, voice-controlled edge surgical robotics. To ensure rapid feedback during CI/CD, isolated developer workflows, and reproducible paper benchmarks, the test harness is decomposed into discrete, specialized modules.

```
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

## 2. Performance Budgets & Real-Time Guardrails

All benchmark runners validate execution times against strict edge budgets defined in `BUDGETS`:

| Metric | Budget | Description | Checked In |
|---|---|---|---|
| **DSP RTF** | $\le 0.50$ (target $\le 0.25$) | Real-time factor per 0.5 s chunk | `bench_dsp`, `bench_e2e` |
| **ASR Finalize** | $\le 150\text{ ms}$ | Time from end-of-speech to final transcript | `bench_asr`, `bench_e2e` |
| **NLU Latency** | $\le 80\text{ ms}$ (Pi 4 $\le 180\text{ ms}$) | Sentence embedding + intent + slot extraction | `bench_nlu`, `bench_e2e` |
| **IPC Transport** | $\le 5.0\text{ ms}$, $> 5{,}000\text{ msg/s}$ | ZeroMQ PUSH $\rightarrow$ PULL one-way latency | `bench_ipc` |
| **Dispatcher Round-Trip** | $\le 5.0\text{ ms}$ | NLU PUSH $\rightarrow$ Command Bus publish | `bench_ipc`, `bench_e2e` |
| **E2E Post-Speech** | $\le 500\text{ ms}$ | Last chunk DSP + ASR finalize + NLU + IPC | `bench_e2e`, `bench_live` |
| **Peak Memory (RSS)** | $\le 800\text{ MB}$ | Total system memory footprint | `bench_soak`, `benchmark_suite` |
| **Memory Leak Drift** | $< 5.0\text{ MB/h}$ | RSS linear regression slope under soak load | `bench_soak` |

---

## 3. Subsystem Benchmarks & CLI Reference

### 3.1 Fast CI/CD Smoke Test (`bench_smoke.py`)
Validates core acoustic simulation, DSP beamforming, ZeroMQ dispatcher round-trip, statistical math, and lexical normalizer in **< 5 seconds** without requiring soundcards or downloading external neural weights.

```bash
# Standard fast smoke test (< 5s):
python test/benchmarks/bench_smoke.py

# Include full SetFit neural weight loading:
python test/benchmarks/bench_smoke.py --with-nlu
```

### 3.2 IPC Transport & Dispatcher Microbenchmark (`bench_ipc.py`)
Benchmarks POSIX IPC (`ipc://`), TCP loopback, and `AF_UNIX` socketpairs across command envelopes (256 B), small payloads (4 KB), and float32 audio chunks (32 KB). Evaluates:
- Emergency priority preemption (safety-critical emergency command injected behind 45 normal commands must dispatch with rank 0).
- High-Water-Mark (HWM) burst load handling under socket saturation.
- Fuzzing / malformed payload recovery (error containment on `ERROR` topic with zero leaks).

```bash
python test/benchmarks/bench_ipc.py --reps-ipc 500 --out results/ipc
```

### 3.3 Isolated DSP Runner (`bench_dsp.py`)
Evaluates Delay-and-Sum beamforming (`ref_channel`, `ref_srp_das`, `repo_beamformer`) and denoising (`passthrough`, `spectral_gate`, `repo_denoiser`) without loading Vosk or PyTorch:
- Real-Time Factor (RTF) across 0.5s audio chunks (mean, p95, p99, CPU-to-wall ratio).
- SI-SDR improvement ($\Delta$SI-SDR in dB) over reference capsule.
- Azimuth localization accuracy (DOA error in degrees, % within 15°).

```bash
# Synthetic array audio:
python test/benchmarks/bench_dsp.py --synthetic 10 --reps 50 --snr-sweep 20,15,10,5,0

# Real recording + noise audio:
python test/benchmarks/bench_dsp.py --manifest data/bench/manifest.jsonl --noise-wav data/noise/or_noise.wav
```

### 3.4 Isolated NLU Runner (`bench_nlu.py`)
Evaluates intent classification accuracy, entity slot extraction, JSON Schema validation ([Draft 2020-12](file:///p:/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/schemas/command_schema.json)), and PyTorch CPU thread scaling:
- Wilson 95% confidence intervals for accuracy, OOD False Accept Rate (FAR), and In-Domain False Reject Rate (FRR).
- Thread scaling sweep across 1 thread (core-pinned), 2 threads, and 3 threads.
- Paired Wilcoxon signed-rank test against TF-IDF baseline if training data is provided.

```bash
python test/benchmarks/bench_nlu.py --reps-nlu 5 --nlu-thread-sweep --out results/nlu
```

### 3.5 Isolated ASR Runner (`bench_asr.py`)
Evaluates Vosk and Whisper speech-to-text decoders:
- Streaming 0.5s chunk processing time and RTF.
- Finalization latency vs 150 ms budget.
- Corpus Word Error Rate (WER) with 95% bootstrap confidence intervals across clean and noisy audio conditions.

```bash
python test/benchmarks/bench_asr.py --manifest data/bench/manifest.jsonl --asrs vosk_grammar,vosk_open
```

---

## 4. System-Level & Operational Runners

### 4.1 End-to-End Pipeline (`bench_e2e.py`)
Tests the full multi-stage cascade with ablation matrix and paired hypothesis testing:
- **Variants**: `ours`, `ablate_no_dsp`, `ablate_bf_only`, `ablate_dn_only`, and baselines (`base_deepfilternet`, `base_noisereduce`, `base_vosk_openvocab`, `base_whisper`).
- **Paired Statistics**: Wilcoxon signed-rank testing paired across identical noise realizations and azimuth angles, adjusted with Holm-Bonferroni correction.
- **Artifacts**: Generates `report_e2e.json`, `raw_e2e.csv`, `tables.md`, `tables.tex`, and latency CDF plots.

```bash
python test/benchmarks/bench_e2e.py --synthetic 10 --snr-sweep 15,10,5,0 --out results/e2e
```

### 4.2 Live Hardware Capture (`bench_live.py`)
Runs on Raspberry Pi 4 with a multi-channel circular microphone array:
- Records live stream for `--live-seconds` (e.g. 10s) via `sounddevice.InputStream`.
- Streams chunks in real time, checking queue backlog ($\le 1$) and input overflows ($0$).
- Saves raw multichannel capture to `live_capture.wav` for reproducible offline re-benchmarking.

```bash
python test/benchmarks/bench_live.py --live-seconds 10 --live-device 1 --live-channels 8
```

### 4.3 Soak & Memory Leak Test (`bench_soak.py`)
Continuous streaming burn-in test running for minutes or hours:
- Evaluates memory leak drift rate ($\text{MB/h}$) via linear regression over process RSS.
- Measures latency drift slope ($\text{ms/h}$).
- Tracks SoC thermal throttling and temperature.

```bash
python test/benchmarks/bench_soak.py --soak-minutes 30 --synthetic 10 --out results/soak
```

---

## 5. Master Orchestrator (`test/benchmark_suite.py`)

The root `benchmark_suite.py` remains 100% backward-compatible with all existing CI/CD invocations, paper evaluation scripts, and shell commands:

```bash
# Full isolated and end-to-end benchmark run:
python test/benchmark_suite.py \
    --manifest data/bench/manifest.jsonl \
    --noise-wav data/noise/or_noise.wav \
    --scope both \
    --baselines \
    --seeds 3 \
    --out results/full_eval
```

