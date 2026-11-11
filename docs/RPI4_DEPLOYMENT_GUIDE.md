# Raspberry Pi 4 Deployment & Testing Guide (SetFit BGE-micro-v2)

This guide provides instructions to run the fine-tuned SetFit intent classification model on a **Raspberry Pi 4 (aarch64 / 64-bit Raspberry Pi OS)**.

---

## 1. System Requirements & Verification

Log into your Raspberry Pi 4 terminal and verify that you are running the 64-bit OS:

```bash
uname -m
```
> **Expected Output:** `aarch64`

Check available RAM (Raspberry Pi 4 has 2GB, 4GB, or 8GB):
```bash
free -h
```

---

## 2. Clone the Repository to the Raspberry Pi

```bash
git clone <YOUR_GITHUB_REPO_URL> scrub-voice
cd scrub-voice
```

Ensure the model directory is present:
```bash
ls -lh src/nlu/models/setfit_bge-micro-v2_8k/
```
You should see `model.safetensors` (~66 MB), `model_head.pkl`, and config JSON files.

---

## 3. Set Up Python Virtual Environment

Modern Raspberry Pi OS (Debian Bookworm) uses managed Python environments (`PEP 668`). Create and activate an isolated virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

---

## 4. Install Dependencies

Install the requirements prepared for Raspberry Pi CPU execution:

```bash
pip install -r requirements-rpi.txt
```

> **Note:** PyPI automatically distributes prebuilt ARM64 wheels for PyTorch CPU, HuggingFace Transformers, and SetFit, so compiling from source is **not** required.

---

## 5. Running Edge Verification & Benchmarks

### Option A: Edge Verification Test (`test/test_edge.py`)
Validates cold-start loading time, resident memory footprint, CPU core affinity, and 100% JSON schema compliance against [`schemas/command_schema.json`](../schemas/command_schema.json):
```bash
python test/test_edge.py
```

Example output:
```text
============================================================
       SCRUB-VOICE EDGE NLU PROFILER (Raspberry Pi 4)
============================================================
[1/4] Measuring Cold-Start Model Load...
  -> Model loaded in 0.78s
[2/4] Measuring Baseline Resident Memory (RSS)...
  -> Memory Delta: +50.0 MB (Total RSS: 254.1 MB)
[3/4] Validating Core Affinity & Latency...
  -> Pinned to CPU Core: 1
  -> Mean Inference Latency: 7.45 ms (Budget: 25.0 ms)
[4/4] Validating 100% Command Schema Compliance across 13 Intents...
  -> Validated 85/85 command envelopes against Draft 2020-12 schema.
  -> Out-of-domain conversational chatter successfully rejected.
============================================================
STATUS: ALL EDGE GUARDS PASSED
============================================================
```

### Option B: Isolated NLU Benchmark Suite (`test/benchmarks/bench_nlu.py`)
Evaluates intent accuracy, slot extraction F1 score, Wilson 95% confidence intervals, and PyTorch CPU thread scaling:
```bash
python test/benchmarks/bench_nlu.py --reps-nlu 5 --nlu-thread-sweep --out results/nlu
```

### Option C: Live Interactive Prompt Console (`test/test_monitor_prompt.py`)
Type natural language commands in real time to verify intent classification and slot extraction against the Intraoperative Monitor:
```bash
python test/test_monitor_prompt.py
```

---

## 6. Performance Expectations on Raspberry Pi 4

| Metric | Target / Expected Value |
| :--- | :--- |
| **Model Size** | ~66.3 MB (`bge-micro-v2` backbone) |
| **RAM Footprint** | ~200 MB – 300 MB RSS |
| **Cold Load Time** | ~1.5 – 3.0 seconds |
| **Inference Latency** | **30 ms – 70 ms** per single utterance on 4x Cortex-A72 CPU |

---

## 7. Optimization Tips for Real-time Operation

1. **Set CPU Governor to Performance (Optional):**
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
2. **Prevent Throttling:** Ensure the Pi has a heatsink or small fan to maintain stable 1.5 GHz - 1.8 GHz clock speed under continuous inference.
