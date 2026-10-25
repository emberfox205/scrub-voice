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

## 3. Quick Replication Command

```bash
# Run automated edge profiler on Raspberry Pi 4 (2 threads)
uv run python test/benchmark_edge_profile.py --threads 2
```
