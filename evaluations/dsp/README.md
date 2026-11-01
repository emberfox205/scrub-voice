# DSP Subsystem: Metrics & Evaluation

Empirical edge performance benchmarks and acoustic evaluation figures for the DSP subsystem on Raspberry Pi 4 Model B (`PI4WURST`, Cortex-A72 @ 1.5 GHz, 4 GB RAM).

For changes from the original prototype code, see **[`CHANGES.md`](CHANGES.md)**.

---

## 1. Measured Hardware Benchmarks (Pi 4)

| Metric | Target Budget | 1 Thread | 2 Threads | 3 Threads | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepFilterNet Latency (Mean)** | $< 500.0\text{ ms}$ | **448.97 ms** | **451.93 ms** | **453.36 ms** | **PASS** |
| **DeepFilterNet Latency (P95)** | $< 500.0\text{ ms}$ | **458.35 ms** | **460.58 ms** | **463.23 ms** | **PASS** |
| **Real-Time Factor (RTF)** | $\le 0.50$ | **0.4490** | **0.4519** | **0.4534** | **PASS** |
| **DeepFilterNet Load Delta** | $< 50.0\text{ MB}$ | $+29.3\text{ MB}$ | $+29.3\text{ MB}$ | $+29.3\text{ MB}$ | **PASS** |
| **Silero VAD Latency (per 32ms)** | $< 10.0\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | **PASS** |
| **Peak Resident RAM (DSP Proc)** | $< 700.0\text{ MB}$ | **629.6 MB** | **630.3 MB** | **629.6 MB** | **PASS** |
| **Denoise Energy Attenuation** | $> 8.0\text{ dB}$ | **$-10.5\text{ to } -14.2\text{ dB}$** (diffuse noise) | **PASS** |

*Note: On an x86_64 host (Intel i7/WSL2), DeepFilterNet achieves an RTF of 0.0485 (48.51 ms per 1.0s audio).*

---

## 2. Metric Evaluation & Key Takeaways

### What is RTF (Real-Time Factor)?
$$\text{RTF} = \frac{\text{Processing Time}}{\text{Audio Duration}}$$
* If $\text{RTF} = 1.0$, the processor takes exactly 1.0 second to process 1.0 second of audio.
* If $\text{RTF} > 1.0$, the algorithm is slower than real time. In a live streaming system, this causes continuous buffer accumulation, audio latency buildup, and eventual memory exhaustion.
* If $\text{RTF} < 0.50$ (our measured **0.4490** on Pi 4), the processor finishes 1.0 second of audio in less than 450 ms. This guarantees zero audio queue accumulation and leaves over 55% of the CPU core free for OS scheduling and I/O.

### What is P95 Latency?
* While *mean* latency measures average execution time, **P95 (95th percentile)** latency represents the upper bound experienced by 95% of audio chunks.
* In surgical operating rooms, worst-case spikes (e.g. caused by memory garbage collection or thread contention) can cause missed commands. Our P95 latency on Pi 4 is **458.35 ms**, comfortably within our 500 ms chunk budget.

### What is Delay-and-Sum Beamforming?
* The Seeed ReSpeaker 8-mic array places 6 microphone capsules in a circle of radius $43\text{ mm}$ around a perimeter.
* Sound arriving from a specific angle $\theta$ reaches closer microphones earlier than farther ones.
* By applying calculated phase delays $\tau_m = -\frac{r \cos(\theta - \phi_m)}{c}$ to each channel and summing them together, the acoustic waves from the target angle interfere **constructively** (amplified), while ambient noise from other directions interferes **destructively** (cancelled).

### What is VAD Window Voting?
* Rather than making an utterance decision on a single momentary audio frame, the VAD evaluates a sliding window of multiple 32 ms frames.
* **Window Voting** requires $\ge 3$ consecutive frames (approx. 100 ms) to exceed the speech probability threshold before declaring speech onset. This prevents transient surgical clicks (e.g., scalpel dropping, tray clatter, throat clearing) from falsely triggering ASR.

---

## 3. Empirical Acoustic Findings & Real-World Caveats

### ⚠️ Acoustic Quality: Neural Denoiser "Vacuum Tube / Pops" Artifact & ASR Phoneme Mutilation
* **Observed Behavior in Live Trials:** DeepFilterNet / DuoFilter unconstrained output can sound "underwhelming" on certain recordings—while stationary noise is removed, the voice exhibits flanging/comb filtering, volume fading, and subtle popping artifacts. Crucially, downstream ASR (Vosk) accuracy drops because the neural denoiser mistakenly suppresses fragile high-frequency unvoiced consonants (/s/, /f/, /t/, /k/) as noise, and creates zero-noise frames that collide with Kaldi's MFCC feature extraction when dither is disabled.
* **Technical Root Causes:**
  1. **ERB/Complex Gain Masking on Unvoiced Consonants:** Unvoiced consonants have noise-like turbulent broadband spectra without pitch harmonics. The neural network's loss function penalizes residual noise, mistaking quiet consonants for background noise and suppressing them by $-30\text{ dB to }-40\text{ dB}$.
  2. **Phase Mismatch & Spectral Gating:** Non-linear frame-by-frame masking creates phase jitter and spectral envelope ripple across adjacent Mel filterbanks, which corrupts Kaldi's Discrete Cosine Transform (DCT) and CMVN variance normalization.
  3. **Linear Beamforming vs. Neural Masking:** Delay-and-Sum (DAS) is a $100\%$ linear spatial filter (zero phase distortion, zero spectral ripple). In quiet-to-moderate environments, pure beamforming (`--no-denoise`) provides pristine, unmodulated formants that Vosk recognizes with high confidence.
* **Implemented Solutions in Pipeline:**
  * **Minimum Gain Floor Clamping (`--atten-lim-db 20.0`):** Clamps over-suppressed frequency bins to the authentic physical microphone input at a $-20\text{ dB}$ floor ($G_{\min} = 0.10$). Prevents dead digital silence, avoids $\ln(0) = -\infty$ Kaldi crashes, and retains the natural acoustic room bed.
  * **Split-Band DuoFilter Consonant Shield (`--split-band-hz 2500.0`):** Applies DuoFilter neural enhancement only in the low-to-mid band ($< 2.5\text{ kHz}$) where $90\%$ of motor, suction, and HVAC noise power lives, while passing the beamformed audio directly in the high band ($> 2.5\text{ kHz}$) with a smooth 8-bin cross-fade. This guarantees $100\%$ preservation of /s/, /f/, /t/, /k/ phonemes for Vosk.
  * **Bypass Option for Pure Beamforming (`--no-denoise`):** For clean/moderate environments, bypasses the neural denoiser completely to stream direct DAS beamformed audio to Vosk, reducing latency and CPU usage to an RTF of $0.002$.

### 💡 Software Delimiter: Explicit "END" Termination Keyword
* **The Clinical Problem:** Relying solely on VAD silence detection in a noisy OR is brittle—HVAC noise or breathing can delay the silence trigger by 400–800 ms, while natural pauses between words can prematurely truncate sentences.
* **Solution Implemented / Recommended:** Support an **explicit trailing command delimiter** (e.g. *"end"*, *"over"*, *"execute"*, or *"now"*), such as:
  > *"Zoom in two times **end**"*  
  > *"Next slice **over**"*
* **Mechanism:** When ASR detects the termination keyword, it immediately flushes the utterance with `{"type": "end"}` over ZeroMQ, completely bypassing the silence hangover timer for zero-latency, unambiguous execution.

---

## 4. Teammate Watch-Outs & Recommendations

1. **ALSA Gain Calibration Across Mic Array:** The Seeed 8-mic array has independent hardware gain sliders in ALSA. If channel 1 is at 80% and channel 3 is at 45%, Delay-and-Sum phase alignment breaks, causing muffled speech and reduced noise suppression. Ensure all perimeter mics are equalized:
   ```bash
   amixer -c seeed8micvoicec sset 'ADC' 80%
   ```
2. **Preserve VAD Recurrent State Reset:** Silero VAD maintains internal LSTM hidden states. In [`src/dsp/source/vad.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/vad.py), `model.reset_states()` must be called upon speech termination to prevent sticky utterance clumping.
3. **Hangover Buffer:** Maintain a trailing silence hangover buffer of at least 300 ms to avoid clipping closing consonants (*"zoom"*, *"times"*, *"reset"*).
4. **Diagnostic Tooling:** Use `--dump-audio <DIR>` to verify raw mic 1 vs beamformed vs denoised WAV files during acoustic trials.
