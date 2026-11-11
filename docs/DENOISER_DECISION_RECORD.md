# Architecture Decision Record (ADR): Retaining vs. Removing the Neural Denoiser

* **Module:** DSP Acoustic Front-End (`src/dsp/`)
* **Status:** Approved / Recommended Architecture
* **Target Hardware:** Raspberry Pi 4 Model B (Cortex-A72 @ 1.5 GHz, 4 GB RAM) + Seeed ReSpeaker 6-Mic Array
* **Downstream Consumers:** Vosk ASR (Kaldi TDNN-HMM grammar) + SetFit NLU Intent Classifier

---

## 1. Context & Problem Statement

In voice-controlled surgical robotics, speech recognition must balance two opposing acoustic objectives:
1. **Aggressive Ambient Noise Rejection:** Operating rooms contain loud, continuous mechanical equipment (HVAC airflow, suction pumps, patient monitoring alarms, and surgical tool motors).
2. **High-Frequency Consonant Preservation:** ASR accuracy critically relies on fragile unvoiced stop and fricative consonants (/s/, /t/, /k/, /f/) with weak acoustic energy.

Earlier prototype iterations introduced aggressive neural denoisers (DeepFilterNet / full-band DuoFilter with 20 dB suppression limits). While effective at producing silence during pauses, these models caused **phoneme mutilation, robotic flanging, and dropped commands** in downstream Vosk ASR.

This document records the empirical evaluation of whether to **keep or remove** the neural denoiser from the edge pipeline.

---

## 2. Empirical Benchmark Evidence

Evaluated using the 20-second clinical ground-truth test suite (`test/test_denoiser_eval.py`) across 5 representative surgical scenarios:

| Metric / Scenario | Non-Denoised (Beamformed) | Old Denoiser (Full-Band @ 20dB) | Consonant Shield (2.5kHz @ 20dB) | Shield + 10dB (Shield @ 10dB) |
| :--- | :---: | :---: | :---: | :---: |
| **truth_01 (DICOM Inspection)** | **0.0% WER** (5/5 intents) | 6.2% WER (5/5 intents) | 25.0% WER (Dropped *"zoom in"*) | **0.0% WER** (5/5 intents) |
| **truth_02 (Tool Delivery)** | **31.6% WER** (4/4 parsed) | 52.6% WER (Severe hallucination) | 42.1% WER (3 unknown intents) | 36.8% WER (2 unknown intents) |
| **truth_03 (Consonant Stress)** | 4.8% WER (**Failed: slice 16**) | 4.8% WER (Slice 33 intact) | 4.8% WER (Slice 33 intact) | **4.8% WER (Slice 33 intact)** |
| **truth_04 (Hemostasis Audit)** | 26.3% WER (4/4 parsed) | 26.3% WER (4/4 parsed) | 21.1% WER (5/5 parsed) | **21.1% WER (5/5 parsed)** |
| **truth_05 (Emergency Recovery)**| **7.7% WER** (5/5 intents) | 7.7% WER (5/5 intents) | 15.4% WER (Dropped *"park arm"*) | **7.7% WER** (5/5 intents) |
| **Macro Average WER** | **~14.08%** | **~19.52%** | **~21.68%** | **~14.08%** |
| **Consonant Preservation** | ✅ Natural physical phase | ❌ Clips /s/, /t/, /k/ | ⚠️ Energy cliff drops words | ✅ Preserved ($>2.5\text{ kHz}$ raw) |
| **Pi 4 Real-Time Factor (RTF)** | **0.110** (11% core) | 0.216 (22% core) | 0.216 (22% core) | **0.216** (22% core) |
| **P95 Latency (Pi 4)** | **67.3 ms** | 108.5 ms | 108.5 ms | **108.5 ms** |

---

## 3. Trade-Off Analysis

### A. Removing the Denoiser (Pure DAS Beamforming)

#### Advantages
1. **100% Linear Phase & Zero Musical Artifacts:** Delay-and-Sum is a linear spatial filter. It introduces no non-linear spectral gating, zero phase jitter, and zero artificial fluttering.
2. **Preserves Kaldi/Vosk Acoustic Features:** Kaldi’s MFCC extraction and Cepstral Mean & Variance Normalization (CMVN) operate reliably on physical acoustics. Pure beamforming avoids zero-energy filterbanks and mathematical singularities ($\ln(0) = -\infty$).
3. **Maximum Edge Headroom:** Shaves off $\sim 108\text{ ms}$ of latency and $22\%$ of CPU core usage on the Raspberry Pi 4, lowering thermal dissipation during sustained surgical sessions.
4. **Natural Acoustic Voice:** Zero frequency seams, zero tonal tilt, and no unnatural background "hiss bed."

#### Vulnerabilities
1. **Physical Limit in Diffuse Noise:** In spherically diffuse noise (HVAC airflow, suction pumps, laminar air hoods), 6 microphones packed in a $43\text{ mm}$ radius have a theoretical array gain limit of:
   $$\text{Array Gain} \le 10 \log_{10}(M) \approx 7.78\text{ dB}$$
   In real clinical testing, measured array gain drops to only **$+1.06\text{ to }+3.35\text{ dB}$**.
2. **Failure Under Low SNR:** If operating room ambient noise reaches $75\text{ dBA}$ and surgeon speech is $65\text{ dBA}$ ($-10\text{ dB SNR}$), beamforming can only raise SNR to $-7\text{ dB}$, causing Vosk ASR failure ($> 70\%$ WER).
3. **Numeric Boundary Confusion:** In File 3, lack of spectral denoising smeared the trailing stop consonant of *"thirty"*, causing Vosk to decode *"thirteen"* and issuing an incorrect slice navigation index (slice 16 instead of 33).

---

### B. Keeping the Denoiser (Always-On Full Suppression)

#### Advantages
1. **Suppresses Isotropic Diffuse Noise:** Attenuates omnidirectional room noise by an extra **$8\text{ to } 14\text{ dB}$** beyond what physical beamforming can achieve.
2. **Resolves Number Ambiguities in Reverb:** Effectively cleaned low-frequency reverberant tails in File 3, allowing Vosk to differentiate *"thirty three"* from *"thirteen three"*.

#### Vulnerabilities
1. **Phoneme Mutilation (The 20 dB Failure):** Unconstrained neural masking mistook weak unvoiced consonants (/s/, /t/, /k/) for background noise, elevating WER to $52.6\%$ on File 2 and dropping whole commands in File 1.
2. **Added Latency & Compute:** Adds $108.5\text{ ms}$ P95 latency per 0.5s chunk, pushing total DSP pipeline latency to $175.8\text{ ms}$.

---

## 4. Architectural Decision

### Decision 1: Default Production Mode is Non-Denoised (`--no-denoise`)
The system will run with **Delay-and-Sum beamforming only** as its default operating state. In typical operating suites ($> 15\text{ dB SNR}$), the neural denoiser introduces more acoustic risk (consonant deletion, spectral ripple) than benefit.

### Decision 2: Retain `Shield + 10dB` in the Codebase
The neural denoiser implementation will **NOT be deleted** from the repository. It will be retained in `src/dsp/denoise/` with its default configuration pinned strictly to:
```bash
--split-band-hz 2500.0 --atten-lim-db 10.0
```
This ensures that if the robot is deployed in an operating suite with active surgical suction or high HVAC rumble ($< 12\text{ dB SNR}$), site administrators can enable noise mitigation without re-introducing consonant mutilation.

---

## 5. Deployment & Configuration Matrix

| Deployment Environment | Ambient Noise Level | Recommended DSP Command Flag | Expected Pipeline RTF |
| :--- | :---: | :--- | :---: |
| **Quiet / Standard OR** | $< 55\text{ dBA}$ ($> 15\text{ dB SNR}$) | `python src/dsp/run_dsp.py --no-denoise` | **0.110** (Ultra-low latency) |
| **Noisy OR / Active Suction** | $55 - 75\text{ dBA}$ ($< 12\text{ dB SNR}$) | `python src/dsp/run_dsp.py --split-band-hz 2500.0 --atten-lim-db 10.0` | **0.326** (Guarded denoise) |
| **Aggressive Denoiser (20dB)**| Any | **DEPRECATED / PROHIBITED** | ❌ Wipes out unvoiced consonants |

---

## 6. Future Upgrade Path: Dynamic Hybrid Bypass

To eliminate manual configuration switches between clean and noisy environments, the production roadmap implements an automatic SNR-based gate in `src/dsp/run_dsp.py`:

```
                  ┌──────────────────────────────┐
                  │ 6-Channel Perimeter Capture  │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  SRP-PHAT + DAS Beamforming  │
                  └──────────────┬───────────────┘
                                 │
                     Compute Live Frame SNR
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
      Live SNR ≥ 12 dB                     Live SNR < 12 dB
     (Clean / Moderate)                     (Heavy Diffuse)
               │                                   │
      BYPASS DENOISER                     ENGAGE DUOFILTER
  (0 ms added, 100% linear)             (Shield @ 2.5kHz, 10dB)
               │                                   │
               └─────────────────┬─────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       Vosk ASR Engine        │
                  └──────────────────────────────┘
```

1. Background noise power $\sigma_N^2$ is estimated during VAD-negative frames using recursive averaging.
2. If real-time speech $\text{SNR} \ge 12\text{ dB}$: Stream raw beamformed audio directly to Vosk (lowest latency, zero distortion).
3. If real-time speech $\text{SNR} < 12\text{ dB}$: Route audio through `Shield + 10dB` DuoFilter to clean diffuse rumble before ASR decoding.
