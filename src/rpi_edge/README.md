# Standalone Minimal Edge NLU Pipeline for Raspberry Pi 4

This directory contains the **minimal, self-contained edge Voice-to-Intent pipeline** optimized for real-time execution on the **Raspberry Pi 4 (Quad-Core ARM Cortex-A72, 64-bit OS)**.

---

## 1. System Requirements

- **Target Hardware:** Raspberry Pi 4 Model B (2GB, 4GB, or 8GB RAM).
- **Operating System:** Raspberry Pi OS 64-bit (`aarch64` / Debian Bookworm or Bullseye).
- **Python Version:** Python 3.9, 3.10, 3.11, or 3.12.

Verify 64-bit architecture on your Raspberry Pi terminal:
```bash
uname -m
# Expected: aarch64
```

---

## 2. Directory Structure

```
rpi_edge/
├── run_edge_pipeline.py      # Interactive real-time CLI prompt
├── pipeline.py               # Complete VoiceToIntentPipeline (Single-pass SetFit + Slots)
├── normalizer.py             # Spoken numbers conversion & filler removal
├── slot_extractor.py         # Deterministic regex & safety bounds extractor
├── dictionaries/             # Clinical vocabularies, fillers, constraints
│   ├── command_schema.json
│   ├── fillers.json
│   ├── intent_corpus.json
│   ├── medical_slots.json
│   ├── numbers.json
│   └── slot_constraints.json
├── schemas/                  # Official Draft 2020-12 JSON Schema
│   └── command_schema.json
├── models/                   # Local model weights directory (or use --model-dir)
│   └── setfit_bge-micro-v2_8k/
├── test_edge.py              # Automated self-test verifying schema compliance & latency
├── requirements.txt          # Minimal Python dependencies for ARM64 edge
└── README.md
```

---

## 3. Installation on Raspberry Pi 4

### Step 1: Copy this folder to the Raspberry Pi
You can transfer the `rpi_edge` directory to the Raspberry Pi via `scp` or `rsync`:
```bash
scp -r rpi_edge admin@<PI_IP_ADDRESS>:~/rpi_edge
```

### Step 2: Ensure Model Weights are in place
The fine-tuned SetFit checkpoint (`setfit_bge-micro-v2_8k`) should contain:
- `model.safetensors` (~66 MB)
- `model_head.pkl` (~20 KB)
- `config_setfit.json`, `config.json`, tokenizer configs.

Place the model folder either at:
1. `~/rpi_edge/models/setfit_bge-micro-v2_8k/` (relative local default)
2. `/home/admin/Downloads/setfit_bge-micro-v2_8k/` (system fallback)
3. Or pass any custom path using `--model-dir /path/to/model`.

### Step 3: Set up Python Virtual Environment
On your Raspberry Pi:
```bash
cd ~/rpi_edge
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** PyPI distributes prebuilt ARM64 (`aarch64`) wheels for PyTorch CPU, HuggingFace Transformers, and SetFit, so compiling from source is **not** required.

---

## 4. Running the Interactive Pipeline

### Standard Interactive Mode
Launch the interactive CLI prompt:
```bash
python3 run_edge_pipeline.py
```

### Custom Model Path or Thread Count
```bash
python3 run_edge_pipeline.py --model-dir /home/admin/Downloads/setfit_bge-micro-v2_8k --threads 4
```

### Automated Self-Test & Benchmark
Verify full schema compliance, slot extraction, and inference latency:
```bash
python3 test_edge.py
```

### Raw JSON Stream Mode (For ZeroMQ / ROS2 / Inter-Process Pipes)
Outputs only single-line JSON commands per utterance:
```bash
python3 run_edge_pipeline.py --json
```

---

## 5. Example Interactive Session

```text
=================================================================
 [*] INITIALIZING MEDICAL NLU PIPELINE ON RASPBERRY PI 4
=================================================================
[OK] Pipeline loaded in 1.42s using 4 CPU threads
[OK] Model: /home/admin/rpi_edge/models/setfit_bge-micro-v2_8k
[OK] Classes (13): clear_emergency, close_scan, halt_emergency, log_event, move_camera...
[OK] Memory Delta: +210.4 MB (Total RSS: 254.1 MB)

=================================================================
 🎙️ INTERACTIVE NLU TEST (Type 'exit', 'quit' or 'q' to stop)
=================================================================

Enter voice command > Hand me the scalpel please
 ─────────────────────────────────────────────────────────────
  Command ID  : cmd_1009_094500_001
  Status      : EXECUTE | Intent: pick_tool
  Category    : TOOL_HANDLING | Priority: NORMAL
  Confidence  : 99.4%
  Latency     : 14.80 ms
  Slots       : {"tool_identifier": "scalpel"}
 ─────────────────────────────────────────────────────────────

Enter voice command > Emergency stop immediately
 ─────────────────────────────────────────────────────────────
  Command ID  : cmd_1009_094512_002
  Status      : EXECUTE | Intent: halt_emergency
  Category    : EMERGENCY | Priority: HIGH
  Confidence  : 99.8%
  Latency     : 13.20 ms
  Slots       : {}
 ─────────────────────────────────────────────────────────────

Enter voice command > Zoom in 2.5x
 ─────────────────────────────────────────────────────────────
  Command ID  : cmd_1009_094525_003
  Status      : EXECUTE | Intent: zoom_display
  Category    : MONITOR_DISPLAY | Priority: NORMAL
  Confidence  : 98.7%
  Latency     : 15.10 ms
  Slots       : {"direction": "IN", "factor": 2.5}
 ─────────────────────────────────────────────────────────────
```

---

## 6. Performance Characteristics on Raspberry Pi 4

| Metric | Measured Specification |
| :--- | :--- |
| **Model Size** | ~66 MB FP32 Safetensors |
| **RAM Footprint (RSS)** | ~250 MB total process memory |
| **Cold-Start Load Time** | ~1.4 - 2.1 seconds |
| **Inference Latency** | **12 - 25 ms** per command on ARM Cortex-A72 |
| **Schema Compliance** | 100% adherence to Draft 2020-12 |
| **Single-Pass Inference** | Computes intent and probability in a single encoder pass |