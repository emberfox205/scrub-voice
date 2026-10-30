# Neural Denoiser ONNX Inference & Audio Playback Guide

This guide covers how to execute your exported neural speech enhancement models (**Stateful DuoFilter**, **Stateless DuoFilter**, **GTCRN**, and **DeepFilterNet**) across the restructured folders, and how to listen to and compare results directly from your terminal.

---

## 1. Directory Organization

The denoising subsystem under `src/dsp/denoise/` is organized into three specialized packages:

```
src/dsp/denoise/
├── stateful_onnx/                       # True stateful recurrent streaming (0.5s chunks, O(1) memory)
│   ├── DuoFilterStreaming.py            # PyTorch streaming wrapper with state caches
│   ├── StatefulCausalDepthwiseConv2d.py # Stateful depthwise convolution module
│   ├── export_duofilter_streaming_onnx.py # Exporter to duofilter_stateful.onnx
│   └── duofilter_streaming_denoiser.py  # Runtime inference engine for 0.5s chunks
├── stateless_onnx/                      # Offline sequence / stateless chunking
│   ├── export_duofilter_onnx.py         # Exporter to duofilter.onnx
│   └── stateless_duofilter_denoiser.py  # Static and stateless OLA engine
├── sota_onnx/                           # SOTA reference comparison engines
│   ├── gtcrn_denoiser.py                # GTCRN ONNX streaming engine
│   ├── deepfilternet_denoiser.py        # DeepFilterNet ONNX engine
│   └── export_to_onnx.py                # DeepFilterNet export script
├── onnx/                                # Compiled ONNX model binaries
│   ├── duofilter_stateful.onnx          # Stateful chunk DuoFilter (~628 KB)
│   ├── duofilter.onnx                   # Stateless sequence DuoFilter (~628 KB)
│   ├── gtcrn_simple.onnx                # GTCRN model (~523 KB)
│   └── deepfilternet.onnx               # DeepFilterNet model (~8.9 MB)
└── checkpoint/                          # PyTorch training checkpoints
    └── duofilter_best_model.ckpt
```

---

## 2. Mode 1: Stateful DuoFilter Streaming (Recommended for Edge & Live DSP)

**Use Case:** Real-time microphone streaming, Raspberry Pi 4 edge deployment, and integration with `run_dsp.py`. Operates on **0.5s chunks (32 STFT frames = 8,000 samples @ 16 kHz)**.

* **Zero Boundary Artifacts:** Preserves continuous state caches (`enc0_cache` through `enc3_cache`, `glu_cache`, and `gru_state`) across consecutive chunk calls.
* **Preserved Dynamic Contrast:** Normalizing across $T=32$ frames allows `GroupNorm` to retain the energy contrast of speech over silence, preventing over-suppression.
* **No Lookback Hack Needed:** Temporal context is retained in the state tensors without duplicate audio processing.

### A. Export the Stateful ONNX Model
```bash
python src/dsp/denoise/stateful_onnx/export_duofilter_streaming_onnx.py \
    --checkpoint src/dsp/denoise/checkpoint/duofilter_best_model.ckpt \
    --output src/dsp/denoise/onnx/duofilter_stateful.onnx \
    --chunk-frames 32
```

### B. Run Stateful Chunk Inference CLI
```bash
python src/dsp/denoise/stateful_onnx/duofilter_streaming_denoiser.py \
    --input data/asr/sample_02.wav \
    --output src/dsp/denoise/audio/clean_duofilter_chunk_05s.wav
```

### C. Python API Integration (Used in `run_dsp.py`)
```python
from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import DuoFilterStreamingDenoiser

# 1. Initialize once:
denoiser = DuoFilterStreamingDenoiser(
    model_path="src/dsp/denoise/onnx/duofilter_stateful.onnx",
    chunk_samples=8000,
    threads=1,
)

# 2. In live streaming loop:
# mono_audio is a 1D float32 numpy array of 8,000 samples (0.5s) from beamformer
clean_chunk = denoiser.enhance_chunk(mono_audio)
```

---

## 3. Mode 2: Stateless DuoFilter Processing (Static & Legacy OLA)

**Use Case:** Offline batch evaluation, full-file processing, or testing magnitude-only phase preservation.

### A. Export the Stateless ONNX Model
```bash
python src/dsp/denoise/stateless_onnx/export_duofilter_onnx.py
```

### B. Full-File Static Processing
```bash
# Continuous full-file static pass (default):
python src/dsp/denoise/stateless_onnx/stateless_duofilter_denoiser.py \
    --input data/asr/sample_02.wav \
    --output src/dsp/denoise/audio/clean_static.wav \
    --threads 1

# Optional: Phase preservation for high-reverberation rooms:
python src/dsp/denoise/stateless_onnx/stateless_duofilter_denoiser.py \
    --input data/asr/sample_02.wav \
    --output src/dsp/denoise/audio/clean_static_phase_pres.wav \
    --phase-preservation
```

### C. Stateless Chunk Streaming (with OLA)
```bash
python src/dsp/denoise/stateless_onnx/stateless_duofilter_denoiser.py \
    --input data/asr/sample_02.wav \
    --output src/dsp/denoise/audio/clean_stream_stateless.wav \
    --stream \
    --chunk-size 8000
```

---

## 4. Mode 3: SOTA Reference Denoisers

### A. GTCRN Speech Enhancement
```bash
python src/dsp/denoise/sota_onnx/gtcrn_denoiser.py \
    --input data/asr/sample_02.wav \
    --output src/dsp/denoise/audio/clean_gtcrn_sample_2.wav
```

### B. DeepFilterNet Speech Enhancement
```bash
# Export DeepFilterNet:
python src/dsp/denoise/sota_onnx/export_to_onnx.py \
    src/dsp/denoise/checkpoint/best_model.ckpt \
    --output src/dsp/denoise/onnx/deepfilternet.onnx

# Run DeepFilterNet inference:
python src/dsp/denoise/sota_onnx/deepfilternet_denoiser.py
```

---

## 5. Live Pipeline Integration (`src/dsp/run_dsp.py`)

`run_dsp.py` connects Silero VAD, SRP-PHAT Localizer, DAS Beamformer, and any selected denoiser to stream 0.5s cleaned speech chunks to ASR via ZeroMQ IPC.

```bash
# 1. Stateful DuoFilter (0.5s chunk streaming, default):
python src/dsp/run_dsp.py --wav data/asr/test_clip_event.wav --denoiser-type duofilter

# 2. GTCRN denoiser:
python src/dsp/run_dsp.py --wav data/asr/test_clip_event.wav --denoiser-type gtcrn

# 3. Stateless DuoFilter:
python src/dsp/run_dsp.py --wav data/asr/test_clip_event.wav --denoiser-type duofilter_stateless

# 4. DeepFilterNet:
python src/dsp/run_dsp.py --wav data/asr/test_clip_event.wav --denoiser-type deepfilter

# 5. Live hardware mic array (Seeed 8-mic):
python src/dsp/run_dsp.py --device "plughw:seeed8micvoicec,0" --denoiser-type duofilter
```

---

## 6. Real-Time Performance Benchmarks (Single CPU Thread)

Measured on 16 kHz audio (sample_02.wav) using a single CPU core:

| Model & Mode | Script Path | Chunk Size | Real-Time Factor (RTF) | Speedup vs Real-Time |
| :--- | :--- | :--- | :--- | :--- |
| **Stateful DuoFilter** | `stateful_onnx/duofilter_streaming_denoiser.py` | **0.5 s (8,000 samples)** | **0.0380** | **~26x faster** |
| **Stateless DuoFilter** | `stateless_onnx/stateless_duofilter_denoiser.py` | Full-file (static) | **0.0270** | **~37x faster** |
| **GTCRN Simple** | `sota_onnx/gtcrn_denoiser.py` | Frame-by-frame (16 ms) | **0.0316** | **~31x faster** |

---

## 7. Listening & A/B Comparison (Terminal Playback)

### On macOS (`afplay`)
```bash
# 1. Listen to raw noisy input:
afplay data/asr/sample_02.wav

# 2. Listen to Stateful 0.5s Chunk Clean Output:
afplay src/dsp/denoise/audio/clean_duofilter_chunk_05s.wav

# 3. Listen to GTCRN Clean Output:
afplay src/dsp/denoise/audio/clean_gtcrn_sample_2.wav

# 4. Quick A/B Comparison:
afplay data/asr/sample_02.wav && afplay src/dsp/denoise/audio/clean_duofilter_chunk_05s.wav
```
*(Press `Ctrl + C` anytime to halt playback).*

### On Linux / Raspberry Pi (`aplay`)
```bash
aplay -D default -r 16000 -f S16_LE src/dsp/denoise/audio/clean_duofilter_chunk_05s.wav
```

---

## 8. Summary of Renamed Files

| Old Location | New Location |
| :--- | :--- |
| `src/dsp/denoise/gtcrn_denoiser.py` | `src/dsp/denoise/sota_onnx/gtcrn_denoiser.py` |
| `src/dsp/denoise/pi_inference.py` | `src/dsp/denoise/sota_onnx/deepfilternet_denoiser.py` |
| `src/dsp/denoise/export_to_onnx.py` | `src/dsp/denoise/sota_onnx/export_to_onnx.py` |
| `src/dsp/denoise/pi_reference_duofilter.py` | `src/dsp/denoise/stateless_onnx/stateless_duofilter_denoiser.py` |
| `src/dsp/denoise/export_duofilter_onnx.py` | `src/dsp/denoise/stateless_onnx/export_duofilter_onnx.py` |
| *(New)* | `src/dsp/denoise/stateful_onnx/duofilter_streaming_denoiser.py` |
| *(New)* | `src/dsp/denoise/stateful_onnx/export_duofilter_streaming_onnx.py` |
| *(New)* | `src/dsp/denoise/stateful_onnx/DuoFilterStreaming.py` |
| *(New)* | `src/dsp/denoise/stateful_onnx/StatefulCausalDepthwiseConv2d.py` |
