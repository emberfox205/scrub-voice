# Full-Pipeline: End-to-End Metrics & Evaluation

Empirical end-to-end benchmarks, total post-speech latency budgets, and safety invariant validation for the complete **Scrub-Voice** edge pipeline on Raspberry Pi 4 Model B (`PI4WURST`, Cortex-A72 @ 1.5 GHz, 4 GB RAM).

For changes from the original prototype code, see **[`CHANGES.md`](CHANGES.md)**.

---

## 1. End-to-End Latency Waterfall (Pi 4 Hardware)

Measured physical elapsed time from the exact instant vocalization stops until the command executes on the intraoperative display:

| Stage | Subsystem | Hardware Elapsed | P95 Worst-Case | Target Budget | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. VAD Boundary Decision** | DSP Silero VAD | **$15.0\text{ ms}$** | $25.0\text{ ms}$ | $\le 30.0\text{ ms}$ | **PASS** |
| **2. Grammar Lattice Search** | ASR Vosk Kaldi | **$120.0\text{ ms}$** | $140.0\text{ ms}$ | $\le 200.0\text{ ms}$ | **PASS** |
| **3. ASR $\rightarrow$ NLU IPC** | ZeroMQ UNIX Socket | **$2.0\text{ ms}$** | $2.8\text{ ms}$ | $\le 5.0\text{ ms}$ | **PASS** |
| **4. Transformer & Slots** | NLU SetFit (2 Thr) | **$152.85\text{ ms}$** | $155.77\text{ ms}$ | $\le 200.0\text{ ms}$ | **PASS** |
| **5. NLU $\rightarrow$ Dispatcher** | ZeroMQ UNIX Socket | **$2.0\text{ ms}$** | $2.8\text{ ms}$ | $\le 5.0\text{ ms}$ | **PASS** |
| **6. Dispatcher & Schema** | CommandDispatcher | **$1.0\text{ ms}$** | $1.5\text{ ms}$ | $\le 5.0\text{ ms}$ | **PASS** |
| **7. WebSocket Push** | FastAPI / Browser | **$< 1.0\text{ ms}$** | $1.2\text{ ms}$ | $\le 5.0\text{ ms}$ | **PASS** |
| **Total Post-Speech Latency** | **End-to-End** | **$\mathbf{293.85\text{ ms}}$** | **$\mathbf{329.07\text{ ms}}$** | **$\mathbf{\le 500.0\text{ ms}}$** | **PASS** |

---

## 2. Resource Utilization & Platform Comparison

| Resource / Metric | Dev Workstation (Intel i7 / WSL2) | Edge Hardware (Raspberry Pi 4 Model B) | Ratio | Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Post-Speech Latency** | $55.1\text{ ms}$ | **$293.85\text{ ms}$** | $5.33\times$ | $\le 500.0\text{ ms}$ (PASS) |
| **DeepFilterNet RTF** | $0.0485$ ($48.5\text{ ms} / \text{s}$) | **$0.4490$ ($449.0\text{ ms} / \text{s}$)** | $9.25\times$ | $\le 0.50$ (PASS) |
| **Total Resident RAM (RSS)**| $652.8\text{ MB}$ | **$635.5\text{ MB}$** | $0.97\times$ | $\le 1000\text{ MB}$ ($> 3.3\text{ GB}$ free) |
| **Continuous CPU Load** | $\sim 25\%$ (8 cores) | **$\sim 160\%$ (4 cores)** | — | $> 240\%$ CPU capacity free |
| **Cold-Start Pipeline Load**| $\sim 2.5\text{ s}$ | **$\sim 4.5\text{ s}$** | $1.80\times$ | $\le 10.0\text{ s}$ (PASS) |

---

## 3. Clinical Ergonomics & Safety Invariant Validation

* **Conversational Naturalness:** Human conversational turn-taking latency averages **$200 - 300\text{ ms}$**. At **$\sim 293\text{ ms}$**, Scrub-Voice responds within the threshold of natural human perception.
* **Ambient Noise Tolerance:** Beamforming spatial filtering + DeepFilterNet suppresses diffuse operating room noise by **$-10.5\text{ to } -14.2\text{ dB}$**, preserving keyword confidence at $1.00$.
* **Out-of-Domain Filtering:** Background banter and non-command speech are classified as `UNKNOWN` ($100\%$ rejection), preventing unintended tool movement.
* **Emergency Halt Preemption:** High-priority commands (`EMERGENCY_STOP`, `ROBOT_FREEZE`) drain over `scrub_nlu_priority.ipc` with $< 0.2\text{ ms}$ queuing delay.
