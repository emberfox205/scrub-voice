# DSP Subsystem: Metrics & Evaluation

Empirical edge performance benchmarks and acoustic evaluation figures for the DSP subsystem on Raspberry Pi 4 Model B (`PI4WURST`, Cortex-A72 @ 1.5 GHz, 4 GB RAM).

For changes from the original prototype code, see **[`CHANGES.md`](CHANGES.md)**.

---

## 1. Measured Hardware Benchmarks (Pi 4)

| Metric | Target Budget | 1 Thread | 2 Threads | 3 Threads | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepFilterNet Mean Latency (1.0s chunk)** | $\le 500.0\text{ ms}$ | **448.97 ms** | 451.93 ms | 453.36 ms | **PASS** |
| **DeepFilterNet P95 Latency** | $\le 500.0\text{ ms}$ | **458.35 ms** | 460.58 ms | 463.23 ms | **PASS** |
| **Real-Time Factor (RTF)** | $\le 0.50$ | **0.4490** | 0.4519 | 0.4534 | **PASS** |
| **DeepFilterNet Load RAM Delta** | $\le 50.0\text{ MB}$ | $+29.3\text{ MB}$ | $+29.3\text{ MB}$ | $+29.3\text{ MB}$ | **PASS** |
| **Silero VAD Latency (per 32ms frame)** | $\le 10.0\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | $3.8 - 8.2\text{ ms}$ | **PASS** |
| **Total DSP Peak Resident RAM** | $\le 700.0\text{ MB}$ | **629.6 MB** | 630.3 MB | 629.6 MB | **PASS** |
| **Denoise Energy Attenuation** | $\ge 8.0\text{ dB}$ | **$-10.5\text{ to } -14.2\text{ dB}$** (diffuse fan / OR noise) | **PASS** |

*Note: On an x86_64 host (Intel i7/WSL2), DeepFilterNet achieves an RTF of 0.0485 (48.51 ms per 1.0s audio).*

---

## 2. Metric Evaluation & Key Takeaways

* **RTF = 0.4490 ($\le 0.50$):** Processing 1.0 second of multi-channel audio takes only 449 ms. This leaves $> 55\%$ of CPU cycles free and guarantees zero audio buffer backlog during continuous live streaming.
* **P95 Latency = 458.35 ms:** Worst-case latency jitter stays well below the 500 ms audio chunk boundary, preventing buffer underruns.
* **VAD Temporal Voting:** Requiring $\ge 3$ consecutive active 32ms windows (approx. 100 ms) successfully filters transient surgical clicks (scalpel drops, tray clatter) without delaying speech onset.

---

## 3. Teammate Watch-Outs & Recommendations

1. **ALSA Gain Calibration Across Mic Array:** The Seeed 8-mic array has independent hardware gain sliders in ALSA. If channel 1 is at 80% and channel 3 is at 45%, Delay-and-Sum phase alignment breaks, causing muffled speech and reduced noise suppression. Ensure all perimeter mics are equalized:
   ```bash
   amixer -c seeed8micvoicec sset 'ADC' 80%
   ```
2. **Preserve VAD Recurrent State Reset:** Silero VAD maintains internal LSTM hidden states. In [`src/dsp/source/vad.py`](file:///mnt/p/AntiGravity/FinalYear/Project_CurrentTopic/scrub-voice/src/dsp/source/vad.py), `model.reset_states()` must be called upon speech termination to prevent sticky utterance clumping.
3. **Hangover Buffer:** Maintain a trailing silence hangover buffer of at least 300 ms to avoid clipping closing consonants (*"zoom"*, *"times"*, *"reset"*).
4. **Diagnostic Tooling:** Use `--dump-audio <DIR>` to verify raw mic 1 vs beamformed vs denoised WAV files during acoustic trials.
