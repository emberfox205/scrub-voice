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
ls -lh models/setfit_bge-micro-v2_8k/
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

## 5. Running the Intent Classifier

### Option A: Full Test (Automated Benchmark + Interactive Testing)
```bash
python test_rpi_intent.py
```

### Option B: Automated Latency & RAM Benchmark Only
Evaluates all 12 surgical commands, calculates mean/min/max/P95 latency, and measures process memory footprint:
```bash
python test_rpi_intent.py --benchmark --runs 5
```

### Option C: Live Interactive Command Prompt Only
Type voice utterances in real-time to inspect intent predictions:
```bash
python test_rpi_intent.py --interactive
```

Example session:
```text
Initializing Edge NLU System on Raspberry Pi 4...
✓ Model loaded in 1.42s
✓ Memory Delta: +210.4 MB (Total RSS: 254.1 MB)

🎙️ INTERACTIVE NLU TEST (Type 'exit' or 'quit' to stop)
=================================================================

Enter voice command > zoom into the patient's wound
 → Intent    : ZOOM_IN
 → Confidence: 99.4%
 → Latency   : 41.20 ms

Enter voice command > switch to next slice
 → Intent    : SLICE_NEXT
 → Confidence: 98.7%
 → Latency   : 38.65 ms

Enter voice command > what time is it
 → Intent    : OUT_OF_SCOPE
 → Confidence: 99.1%
 → Latency   : 39.10 ms
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
