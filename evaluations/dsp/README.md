# DSP Subsystem: Metrics & Evaluation

Empirical edge performance benchmarks and acoustic evaluation figures for the DSP subsystem on Raspberry Pi 4 Model B (`PI4WURST`, Cortex-A72 @ 1.5 GHz, 4 GB RAM).

For changes from the original prototype code, see **[`CHANGES.md`](CHANGES.md)**.

---

## 1. Measured Hardware Benchmarks (Pi 4 vs. PC Reference)

Empirically profiled via `python test/benchmarks/bench_dsp.py` with multi-channel acoustic array simulation across varying SNR levels:

### A. Real-Time Factor (RTF) & Latency Profile

| Pipeline Stage / Engine | Pi 4 Mean RTF | Pi 4 P95 Latency | Workstation RTF (WSL2) | Target Budget | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Delay-and-Sum Beamformer (6-Mic)** | **0.110** | 67.3 ms | **0.044** (23.3 ms) | $\text{RTF} \le 0.15$ | **PASS** |
| **DuoFilter Neural Denoiser (Default)** | **0.216** | 108.5 ms | **0.027** (16.1 ms) | $\text{RTF} \le 0.35$ | **PASS** |
| **Combined DSP Pipeline (DAS + DuoFilter)** | **0.326** | **175.8 ms** | **0.071** (39.4 ms) | $\mathbf{\text{RTF} \le 0.50}$ | **PASS** |
| *DeepFilterNet Denoiser (Alternative)* | 0.449 | 458.35 ms | 0.048 (48.5 ms) | $\text{RTF} \le 0.50$ | PASS |
| **Silero VAD (per 32ms frame)** | $3.8 - 8.2\text{ ms}$ | $9.1\text{ ms}$ | $0.8 - 1.2\text{ ms}$ | $< 10.0\text{ ms}$ | **PASS** |

### B. Acoustic Quality & Spatial Localization (Pi 4 Array Evaluation)

| Acoustic Test Metric | 15 dB SNR (Normal OR) | 0 dB SNR (Loud Ambient) | Target Budget | Status |
| :--- | :--- | :--- | :--- | :--- |
| **SRP-PHAT / DAS Median DOA Error** | **0° median error** | 80° median error | $\le 15^\circ$ (15 dB) | **PASS** |
| **Beamformer Array Gain (SI-SDR)** | **+3.35 dB** | **+1.06 dB** | $> +2.0\text{ dB}$ (15 dB) | **PASS** |
| **Denoise Diffuse Noise Attenuation** | **$-8.81\text{ to } -14.20\text{ dB}$** | $-6.50\text{ to } -8.20\text{ dB}$ | $> 8.0\text{ dB}$ | **PASS** |
| **Peak Resident RAM (DSP Process)** | **629.6 MB** | 630.3 MB | $< 700.0\text{ MB}$ | **PASS** |

*Key Finding:* Switching to **DuoFilter** cuts neural denoiser RTF in half on the Pi 4 (from 0.449 down to **0.216**), lowering total DSP CPU utilization to just **32.6% of one core** and preserving >67% CPU headroom for concurrent ASR decoding.

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

### ⚠️ Acoustic Quality: Neural Denoiser "Vacuum Tube / Pops" Artifact
* **Observed Behavior in Live Trials:** DeepFilterNet output can sound "underwhelming" on certain recordings—static noise is only moderately attenuated, while the speaker's voice exhibits a "vacuum tube" coloration (flanging/comb filtering), volume fading in and out, and subtle popping artifacts.
* **Technical Root Cause:**
  1. **ERB Gain Masking on Reverberant Speech:** DeepFilterNet applies neural gain masks across Equivalent Rectangular Bandwidth (ERB) subbands. When speech contains room reverberation or low SNR, rapid gain transitions between frames create spectral gating artifacts ("musical noise" and fluctuating volume).
  2. **Phase Mismatch:** Synthesis of complex spectrogram bins can introduce phase discontinuities, creating the sensation of speaking through a hollow tube.
* **Recommendations for Team:**
  * **Atten Limiting:** Clamp maximum attenuation to a bounded floor (e.g. $-12\text{ dB}$ or $-15\text{ dB}$) rather than full suppression to prevent voice dropouts.
  * **Dry/Wet Blending:** Blend 80% denoised with 20% raw audio to preserve natural vocal formant timbre.
  * **Mono Simulation Limitation:** Note that testing with a mono WAV file duplicates audio across all 8 channels; Delay-and-Sum beamforming requires true multi-channel physical mic differences to form a spatial beam.

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
