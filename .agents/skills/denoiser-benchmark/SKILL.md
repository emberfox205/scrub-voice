---
name: denoiser-benchmark
description: >-
  Benchmark and evaluate DSP speech denoisers (DuoFilter, DeepFilterNet, split-band consonant shield,
  and attenuation limiting) against Vosk ASR and downstream NLU intent extraction with side-by-side
  comparisons against Ground Truth. Use whenever testing new denoiser checkpoints, tuning attenuation
  limits, or diagnosing consonant erasure in medical voice commands.
---

# DSP Denoiser & ASR Evaluation Skill

This skill provides standard operating procedures and benchmark workflows for evaluating DSP neural noise suppression models (DuoFilter and DeepFilterNet) in combination with the 6-microphone circular array Delay-and-Sum (DAS) beamformer, the Vosk surgical grammar ASR engine, and the SetFit NLU pipeline.

---

## 1. Acoustic Architecture & Defense Mechanisms

When neural denoisers suppress background noise too aggressively (unconstrained gain masks down to $-\infty\text{ dB}$), two major failure modes occur in speech recognition:

1. **High-Frequency Consonant Erasure:**
   - Unvoiced fricatives (e.g., `/θ/` in *"three"*, `/s/` in *"slice"*, *"six"*, `/t/` in *"stop"*, *"times"*) exhibit high-frequency acoustic turbulence ($3,000\text{ Hz} - 7,000\text{ Hz}$) that resembles white noise hiss.
   - Deep neural masks attenuate these bins, stripping consonants and causing Vosk to misrecognize *"three"* as *"ray"*.
2. **Kaldi CMVN / Zero-Variance Collapse:**
   - When pauses are zeroed out ($0.0$), Kaldi MFCC extraction hits $\ln(0) = -\infty$, leading to CMVN normalization collapse.

### The Two Complementary Solutions
- **Split-Band Consonant Shield (`--split-band-hz 2500.0`):**
  - **Below 2.5 kHz:** Neural denoiser applies deep suppression to kill HVAC, robot motor hum, and low-frequency rumble.
  - **Above 2.5 kHz:** Smoothly cross-fades and preserves the raw/beamformed acoustic spectrum, protecting consonants completely.
- **Minimum Gain Floor Clamping (`--atten-lim-db 20.0` or `10.0`):**
  - Clamps the minimum transmission gain: $\text{min\_gain} = 10^{-\text{atten\_lim\_db} / 20}$.
  - Prevents total spectral extinction and preserves a realistic acoustic room floor.

---

## 2. Automated Comparative Benchmark

To run the unified, single-load benchmark across the 4 standard clinical recordings comparing **Beamformed**, **Old Denoiser (20dB Full-Band)**, **New Denoiser (Split-Band Consonant Shield @ 2.5kHz)**, and **Dialed-Back (10dB)**:

```bash
uv run python test/test_denoiser_eval.py
```

### Key Advantages:
- **Zero Reload Overhead:** Loads Vosk and SetFit NLU into memory once, evaluating all 20 conditions in seconds rather than minutes.
- **Automated Ground Truth Metrics:** Computes Word Error Rate (WER), phonetic keyword hit rate, confidence scores, and NLU intent matches.
- **Markdown Summary:** Generates a ready-to-share side-by-side comparison table.

---

## 3. Testing Custom Audio Files & Exports

### Test Arbitrary Audio Files
You can pass any list of WAV files or directories:
```bash
uv run python test/test_denoiser_eval.py data/asr/recording_1.wav data/asr/test_clip_event.wav
```

### Export Denoised Audio for Human Listening
To save the intermediate audio stages (`_beamformed.wav`, `_old_20db.wav`, `_new_shield_20db.wav`, `_dialed_10db.wav`) to disk:
```bash
uv run python test/test_denoiser_eval.py --export-wavs data/diagnostic/denoiser_test/
```

### Fast ASR-Only Run (Bypassing NLU)
If testing only phonetic transcription and skipping NLU intent classification:
```bash
uv run python test/test_denoiser_eval.py --no-nlu
```

---

## 4. Acoustic Stage-by-Stage Listening (`test_dsp_listen.py`)

To inspect a single audio recording stage-by-stage (Raw $\rightarrow$ Beamformed $\rightarrow$ Neural Denoised) and measure dB energy suppression:

```bash
uv run python test/test_dsp_listen.py data/asr/live_capture.wav \
  --atten-lim-db 20.0 \
  --split-band-hz 2500.0 \
  --out-dir data/diagnostic/listen_test
```

---

## 5. Interpreting Results & Golden Rules

| Metric | Target | Warning / Action |
| :--- | :---: | :--- |
| **Word Error Rate (WER)** | $\le 15\%$ | If $\text{WER} > 30\%$, check for consonant erasure |
| **Consonant Status** | `✅ Preserved` | If *"three"* becomes *"ray"*, ensure `--split-band-hz 2500.0` is active |
| **Phoneme Artifacts** | None | If spurious words appear (e.g. *"m"*), dial back attenuation limit to $10.0\text{ dB}$ |
| **Vosk Dither** | `--dither=0` | Always keep dither at 0 for deterministic benchmarks |

