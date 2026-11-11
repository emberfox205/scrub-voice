# SCRUB-VOICE Benchmark Suite: Implementation Blueprint

Companion document to `test/benchmark_suite.py`. It describes what the suite measures, how each measurement is made, what it outputs, and which assumptions a reader of the resulting paper should know about.

---

## 1. Goals

1. Benchmark every subsystem separately (**isolated**) and the whole chain (**end-to-end**): DSP (beamforming and denoising), ASR, NLU and the ZeroMQ orchestrator bus.
2. Run in two input modes:
   - **wav**: offline, on a pre-recorded multichannel WAV, a labelled dataset manifest, or synthetic clips.
   - **live**: on the Raspberry Pi with the array microphone, for about 10 seconds in real time.
3. Produce evidence at research-paper level: percentile latency distributions with confidence intervals, accuracy with confidence intervals, SNR sweeps, ablations, head-to-head baselines on identical hardware and data, and paired significance tests.
4. Emit machine-readable results (JSON, CSV) and paper-ready artifacts (Markdown and LaTeX tables, figures).

### Non-goals (not implemented)

- Supervisor SIGKILL chaos test (needs the supervisor's API).
- WebSocket multi-client scaling (needs a running FastAPI/Uvicorn server).
- VAD benchmarking (Silero VAD is not exercised by this script).
- Comparison against numbers quoted from papers. Baselines run on the same Pi with the same audio instead, because published figures come from different hardware and datasets.

---

## 2. Architecture of the script

Single file placed in `test/benchmark_suite.py`, organised in numbered sections:

| # | Section | Contents |
|---|---|---|
| 1 | Statistics | `summarize`, `wilson`, `holm`, `paired_test` |
| 2 | Hardware and resources | `hw_info`, `ResourceSampler` |
| 3 | Audio, geometry, simulation | loaders, circular array, plane-wave model, noise synthesis, `simulate_array` |
| 4 | Metrics | WER, SI-SDR, STOI, DOA error, slot match |
| 5 | Components | beamformers, denoisers, ASR engines, NLU engines, `Registry` |
| 6 | Dataset | `Item`, manifest loading, `make_sample`, `build_stream` |
| 7 | Dispatcher rig | `DispatcherRig` (ZeroMQ fixture around `CommandDispatcher`) |
| 8 | Benchmarks | `bench_dsp`, `bench_asr`, `bench_nlu`, `bench_ipc`, `bench_e2e`, `bench_live`, `bench_soak` |
| 9 | Reporting | JSON, CSV, Markdown/LaTeX tables, figures, budget table |
| 10 | Main | CLI, orchestration of stages |

### Design principles

- **Nothing is fatal.** A component that cannot load (library missing, plugin not wired) is recorded in `skipped_components` with the reason and the run continues.
- **Lazy loading with accounting.** `Registry.get(kind, name)` builds components on first use and records load time and RSS delta.
- **Plugin contract instead of guessing repo internals.** The repo's own beamformer, denoiser and ASR attach through `--plugin NAME=module:attr`.
- **Paired design.** Every pipeline variant sees the same noise realisation, azimuth and SNR for a given (seed, utterance), which makes paired tests valid.
- **Streaming semantics.** E2E runs process audio in 0.5 s chunks exactly as the live pipeline would, so only the work done after the last chunk counts toward post-speech latency.

---

## 3. Inputs

### 3.1 Dataset manifest (`--manifest`, JSONL)

One utterance per line:

```json
{"audio": "clips/a.wav", "text": "zoom in two times", "intent": "zoom_display",
 "status": "EXECUTE", "slots": {"direction": "IN", "factor": 2.0}, "azimuth": 30}
```

- `audio` is resolved relative to the manifest. Audio is resampled to 16 kHz.
- **Mono clean audio** triggers simulation: the signal is placed on the array and mixed with noise at each SNR. This provides ground truth for SI-SDR and DOA.
- **Multichannel audio** is treated as a real array recording. No SNR sweep; no clean reference, so only latency, WER and NLU metrics apply.
- `status` is `EXECUTE` or `OUT-OF-DOMAIN`. OOD items are scored on rejection.

### 3.2 Other inputs

| Flag | Purpose |
|---|---|
| `--wav` | Single WAV (any channel count). Latency, RTF and resource profiling only. |
| `--synthetic N` | N synthetic speech-like clips. Validates the harness; results are not meaningful for the paper. |
| `--noise-wav` | Real OR noise recording used for sweeps. Without it, synthetic OR-like noise is used. |
| `--nlu-eval` | JSONL `{text, intent, status[, slots]}`. Defaults to labelled manifest items, then to the 17 built-in utterances. |
| `--nlu-train` | JSONL training set for the TF-IDF baseline. |
| `--grammar` | JSON list or text lines of Vosk grammar phrases. If omitted, uses `src.asr.vocab.vosk_grammar_phrases()` automatically. |

### 3.3 Live input

`--mode live` uses `sounddevice`: `--live-device`, `--live-channels` (default 8), `--live-seconds` (default 10), `--mic-channels` (default `0,1,2,3,4,5`) to select the six perimeter capsules. The capture is saved to `live_capture.wav` in the results folder, so any live run can be replayed offline.

---

## 4. Array and simulation model

| Item | Model |
|---|---|
| Geometry | Circular array, `--mic-radius` (default 0.043 m), capsules evenly spaced. Verify channel order for your device. |
| Propagation | Far-field plane wave, applied as fractional delays in the frequency domain. Mic nearer to the source receives earlier. |
| Target signal | Clean speech as received at **mic 0** (the reference mic). All SI-SDR values are measured against this. |
| Noise | `diffuse_ratio` (default 0.5) of power is independent per-channel noise; the rest is a single directional interferer at an azimuth 60 to 300 degrees away from the talker. |
| SNR | Defined at mic 0 over the whole utterance. Sweep default: 20, 15, 10, 5, 0 dB plus a clean condition. |
| Synthetic noise | Pink HVAC rumble, band-passed suction hiss, ventilator-like cycles, clatter impulses, 1 kHz alarm beeps. |

Limitations: no reverberation, no moving talker, partially unrealistic noise coherence. Real array recordings (multichannel manifest entries or live captures) should complement the simulated sweeps in a paper.

---

## 5. Components under test

### 5.1 Beamformers (`--beamformers`)

| Name | Description |
|---|---|
| `ref_channel` | Reference mic only (no beamforming). Lower-bound baseline. |
| `ref_srp_das` | Reference numpy implementation: SRP-PHAT DOA over 72 angles (5 degree grid), 300 to 4000 Hz band with every 4th FFT bin, followed by frequency-domain delay-and-sum steered to mic 0. |
| `repo_beamformer` | Your implementation via plugin or built-in repo adapter (`src.dsp.source.beamformer`). `.process(x[C,N], sr) -> mono` or `(mono, azimuth_deg)`. DOA metrics only if an azimuth is returned. |

### 5.2 Denoisers (`--denoisers`)

| Name | Description |
|---|---|
| `passthrough` | No denoising. |
| `spectral_gate` | Classical stationary-noise gating (percentile noise floor, Wiener-style gain, floor 0.1). |
| `noisereduce` | `noisereduce` library, non-stationary mode (optional dependency). |
| `deepfilternet` | DeepFilterNet reference (`pip install deepfilternet`), resampled to its internal rate (optional). |
| `repo_denoiser` | Your DuoFilter / DeepFilterNet pipeline via plugin or built-in adapter. `.process(x[N], sr) -> y[N]`. |

### 5.3 ASR engines (`--asrs`)

All engines expose `.reset()`, `.accept(chunk)`, `.finalize() -> text`, which models streaming.

| Name | Description |
|---|---|
| `vosk_grammar` | Vosk with the constrained grammar (`src/asr/vocab.py`, your production configuration). |
| `vosk_open` | Same Vosk model, open vocabulary. |
| `whisper` | faster-whisper (`--whisper-model`, default `tiny.en`, int8, greedy). Non-streaming: buffers audio and decodes at the end. |
| `repo_asr` | Your ASR wrapper via plugin. |

Text is normalised (lowercase, punctuation stripped) before WER.

### 5.4 NLU engines

| Name | Description |
|---|---|
| `repo_setfit` | `VoiceToIntentPipeline(num_threads=--threads)` from `src/nlu/pipeline.py`. |
| `tfidf_logreg` | TF-IDF (1-2 grams) and logistic regression; OOD when max probability < 0.5. Needs `--nlu-train` and scikit-learn. |

### 5.5 Plugin contract

```bash
--plugin repo_beamformer=src.dsp.<module>:<Class>
--plugin repo_denoiser=src.dsp.denoise.<module>:<Class>
--plugin repo_asr=src.asr.<module>:<Class>
```

`attr` is a class or zero-argument factory. Beamformer and denoiser wrappers try `process(x, sr)` and fall back to `process(x)`. Outputs may be numpy arrays or CPU tensors.

---

## 6. Stages

Select isolated stages with `--stages dsp,asr,nlu,ipc` (default all) and `--scope isolated|e2e|both`.

### 6.1 DSP (`bench_dsp`)

**Timing.** A continuous multichannel stream (real file or simulated at 10 dB) is cut into 0.5 s chunks. After `--warmup` chunks, `--reps` chunks (default 100) are timed per component.

| Metric | Definition |
|---|---|
| Latency | Wall-clock per chunk, with `summarize` statistics. |
| RTF | Wall time divided by chunk duration. Budget: p95 RTF <= 0.50. |
| Deadline miss rate | Fraction of chunks with RTF > 1. |
| Budget miss rate | Fraction with RTF > 0.5. |
| CPU/wall | Process CPU time over wall time, a proxy for threading efficiency. |

**Quality** (needs clean mono references; per seed, SNR and utterance, chunked like production):

| Metric | Applies to | Definition |
|---|---|---|
| SI-SDR | both | After integer-lag alignment (+/- 400 samples); capped at 60 dB. |
| dSI-SDR | both | Output SI-SDR minus reference-mic input SI-SDR. Not reported for the clean condition (the input is already near-perfect). |
| STOI | both | Optional (`--stoi`, needs `pystoi`). |
| DOA error | beamformers | Circular absolute error between the median per-chunk azimuth and ground truth. Also fraction within 15 degrees. |

### 6.2 ASR (`bench_asr`)

Audio is the reference-mic signal (no DSP), fed in 0.5 s chunks, for the clean condition and each SNR.

| Metric | Definition |
|---|---|
| WER | Corpus-level (total edits over total reference words) with a 1000-sample bootstrap CI over utterances. |
| Finalize latency | Time of `finalize()` after the last chunk is accepted. Budget: 150 ms. |
| RTF | (accept time + finalize time) over audio duration. |
| Load time, RSS delta | From the registry. |

### 6.3 NLU (`bench_nlu`)

Each utterance is run `--reps-nlu` times (default 5) after warm-up; accuracy comes from the last run.

| Metric | Definition |
|---|---|
| Intent accuracy | In-domain: intent equals ground truth and status is not OOD. OOD item: status is `OUT-OF-DOMAIN`. Wilson 95% CI. |
| OOD false-accept rate | OOD items predicted `EXECUTE`. Safety-critical; should be 0. |
| In-domain false-reject rate | In-domain items predicted OOD. |
| Slot exact match | All ground-truth slots present and equal (numeric tolerance 1e-3). Only for items with slots. |
| Schema validity | Envelope validated against `pipeline.schema` (jsonschema). |
| Latency | Per-call, with percentiles. Budget p95 <= 180 ms. |
| Thread sweep (`--nlu-thread-sweep`) | 1 thread pinned to one core, 2 threads free, 3 threads free. |
| Paired test | SetFit vs TF-IDF correctness, if a baseline was trained. |

The script warns when fewer than 100 NLU utterances are used. For publication, use at least 300 utterances including at least 100 OOD.

### 6.4 IPC and dispatcher (`bench_ipc`)

**Transport microbenchmark.** For ZeroMQ IPC, ZeroMQ TCP loopback and a raw `AF_UNIX` `SOCK_SEQPACKET` pair, at payload sizes 256 B (command envelope), 4 KB, 32 KB (0.5 s float32 audio chunk):
- one-way latency with the sender timestamp embedded in the payload (`--reps-ipc`, default 500);
- throughput over a sustained burst (4 times the repetitions).

Budget: p95 <= 5 ms and throughput > 5000 msg/s on the 256 B IPC case.

**Dispatcher tests** (use `CommandDispatcher`; envelopes are produced by the real NLU so they are schema-valid):

| Test | Method | Pass criterion |
|---|---|---|
| Round trip | PUSH an envelope, wait for the PUB message | p95 <= 5 ms |
| Priority preemption | Queue 45 normal commands and 1 emergency on the priority channel **before** the dispatcher loop starts, then start it and record publish order | Emergency rank 0 |
| HWM burst | 200 commands with non-blocking send; count sent, published, sender-side drops, RSS delta, dispatcher alive | Reported, not auto-judged: interpret against your intended drop policy |
| Malformed payloads | 8 bad payloads (empty, truncated JSON, binary garbage, wrong types, missing keys, oversized) | All answered on the `ERROR` topic, none leak to other topics, dispatcher alive, a valid message still round-trips |

### 6.5 End-to-end (`bench_e2e`)

Per utterance, for each variant, SNR, seed:

```text
for each 0.5 s chunk:  beamform -> denoise -> asr.accept
after last chunk:      asr.finalize -> NLU -> dispatcher round trip (EXECUTE only)
```

**Post-speech latency** is defined as:

```text
last-chunk (beamform + denoise + asr.accept) + asr.finalize + NLU + dispatcher round trip + VAD hangover constant
```

`--vad-hangover-ms` defaults to 0, so the reported number excludes VAD silence hangover. State this explicitly in the paper. Budget: p95 <= 500 ms (stretch 300 ms).

**Variants:**

| Variant | Beamformer | Denoiser | ASR |
|---|---|---|---|
| `ours` | repo (fallback `ref_srp_das`) | repo (fallback `spectral_gate`) | repo (fallback `vosk_grammar`) |
| `ablate_no_dsp` | ref channel | passthrough | ours |
| `ablate_bf_only` | ours | passthrough | ours |
| `ablate_dn_only` | ref channel | ours | ours |
| `base_deepfilternet` (`--baselines`) | ours | DeepFilterNet | ours |
| `base_noisereduce` (`--baselines`) | ours | noisereduce | ours |
| `base_vosk_openvocab` (`--baselines`) | ours | ours | `vosk_open` |
| `base_whisper` (`--baselines`) | ours | ours | `whisper` |

### 6.6 Live (`bench_live`)

Records `--live-seconds` while processing the stream in real time using the `ours` variant.

| Metric | Definition |
|---|---|
| Chunk completion lag | Time from the chunk's capture callback to the end of its processing. |
| Max queue backlog | Largest number of unprocessed chunks waiting. Should stay <= 1. |
| Input overflows | `sounddevice` `input_overflow` count. Should be 0. |
| RTF | Per-chunk processing time over chunk duration. |
| Post-speech latency | Last-chunk lag + finalize + NLU + dispatcher round trip. |

### 6.7 Soak (`--soak-minutes`)

Loops the `ours` pipeline at 10 dB and samples RSS and latency over time. Reports the linear RSS slope (MB/hour, pass if < 5 and peak RSS <= 800 MB) and latency drift (ms/hour). Use 120 minutes to match the 2-hour leak target.

---

## 7. Statistical methodology

| Quantity | Method |
|---|---|
| Latency, RTF, SI-SDR | Mean, std, min, max, p50, p90, p95, p99, and a 95% percentile-bootstrap CI of the mean. |
| WER | Corpus-level ratio, bootstrap over utterances (1000 resamples). |
| Proportions (accuracy, FAR, FRR, slot match) | Wilson 95% score interval. |
| Variant comparison | Paired Wilcoxon signed-rank, Holm-corrected. |
| Repeatability | `--seeds` independent noise and azimuth realisations (default 3), fixed seeds recorded in the report. |
| Warm-up | `--warmup` iterations are discarded before timing. |

---

## 8. Resource and platform instrumentation

`ResourceSampler` runs in a background thread during every stage (250 ms interval) and records:
- per-core CPU percent (mean and peak) and their sum, to compare against the 160% of 400% expectation and the core-partitioning plan;
- RSS of the benchmark process and its children, to compare against the 800 MB budget;
- SoC temperature (`/sys/class/thermal`);
- `vcgencmd get_throttled` flags, current and since boot (under-voltage, frequency cap, throttling).

---

## 9. Outputs

Written to `results/bench_<timestamp>[_tag]/`:

| File | Content |
|---|---|
| `report.json` | Arguments, hardware, per-stage results, resource summaries, component load info, skipped components. |
| `raw_<stage>.csv` | One row per measurement (DSP quality, ASR, NLU, E2E) for external analysis in R or pandas. |
| `tables.md`, `tables.tex` | E2E latency/accuracy table and DSP RTF table (LaTeX uses booktabs). |
| `fig_e2e_latency_cdf.{pdf,png}` | Post-speech latency CDF per variant with the 500 ms budget line. |
| `fig_wer_vs_snr.{pdf,png}` | WER against SNR per variant. |
| `live_capture.wav` | Live mode only. |

The console ends with a PASS/FAIL **budget summary**.

### Budgets (from the architecture document)

| Metric | Budget |
|---|---|
| DSP RTF (p95) | <= 0.50 |
| ASR finalize | <= 150 ms |
| NLU latency (p95) | <= 180 ms |
| IPC transit (p95) | <= 5 ms |
| Dispatcher round trip (p95) | <= 5 ms |
| E2E post-speech (p95) | <= 500 ms (stretch 300 ms) |
| Total RSS | <= 800 MB |
| Emergency preemption | emergency published first |

---

## 10. Running It

```bash
# 1. Harness smoke check (validates harness without needing models or recordings)
python test/benchmark_suite.py --synthetic 6 --stages dsp --scope isolated

# 2. Benchmark isolated NLU with thread sweep (SetFit)
python test/benchmark_suite.py --stages nlu --scope isolated --nlu-thread-sweep

# 3. Benchmark IPC transport & CommandDispatcher
python test/benchmark_suite.py --stages ipc --scope isolated

# 4. Full paper run (offline, labelled manifest, 3 noise seeds)
python test/benchmark_suite.py \
  --manifest data/bench/manifest.jsonl \
  --noise-wav data/noise/or_noise.wav \
  --scope both --baselines --seeds 3 --tag paper_eval

# 5. Live 10-second real-time capture from ReSpeaker array
python test/benchmark_suite.py --mode live --live-seconds 10 --live-device 1 --live-channels 8

# 6. 2-Hour Soak & Memory Leak Test
python test/benchmark_suite.py --synthetic 10 --scope isolated --stages ipc --soak-minutes 120
```

