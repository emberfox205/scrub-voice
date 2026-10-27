# Neural Denoiser ONNX Inference & Audio Playback Guide

This guide covers how to execute your exported neural speech enhancement models (**DuoFilter**, **DeepFilterNet**, and **GTCRN**) in **both Static (Full-File)** and **Real-Time Streaming (Chunk-by-Chunk)** modes, and how to instantly listen to and compare the results directly from your terminal.

---

## 1. Prerequisites & Environment

Activate your conda environment (`scrub`) and ensure the core files and models are available:

### Core Files
* **Inference Scripts:**
  * `src/dsp/denoise/pi_reference_duofilter.py` (DuoFilter engine - supports both Static and Streaming OLA)
  * `src/dsp/denoise/gtcrn_denoiser.py` (GTCRN engine)
  * `src/dsp/denoise/pi_inference.py` (DeepFilterNet engine)
* **Exported ONNX Models:**
  * `src/dsp/denoise/onnx/duofilter.onnx`
  * `src/dsp/denoise/onnx/gtcrn_simple.onnx`
* **Test Audio:**
  * `data/asr/test_clip_event.wav` (or `src/dsp/denoise/audio/noise_audio.wav`)

---

## 2. Mode 1: Static Audio Processing (Offline / Full File)

**Use Case:** Offline batch processing, benchmark evaluation, or processing complete pre-recorded audio files. The entire waveform is analyzed and synthesized with continuous STFT -> ONNX -> iSTFT, completely eliminating chunk boundary artifacts and window tapering dips.

### CLI Execution

To run in static mode, simply omit the `--stream` argument (or leave default):

```bash
# Run DuoFilter on a static audio file
python src/dsp/denoise/pi_reference_duofilter.py \
    --input data/asr/test_clip_event.wav \
    --output src/dsp/denoise/audio/clean_static.wav \
    --threads 1

# Optional: Enable Phase Preservation (Magnitude-only masking) for heavy room reverb:
python src/dsp/denoise/pi_reference_duofilter.py \
    --input data/asr/test_clip_event.wav \
    --output src/dsp/denoise/audio/clean_static_phase_pres.wav \
    --phase-preservation
```

### Python API Example (Static Mode)

```python
import soundfile as sf
from src.dsp.denoise.pi_reference_duofilter import DuoFilterDenoiser

# 1. Initialize engine
denoiser = DuoFilterDenoiser(threads=1)

# 2. Load complete audio
audio, sr = sf.read("data/asr/test_clip_event.wav", dtype="float32")

# 3. Enhance in a single continuous pass (streaming=False)
clean_audio = denoiser.enhance(audio, sample_rate=sr, streaming=False)

# 4. Save enhanced audio
sf.write("src/dsp/denoise/audio/clean_static.wav", clean_audio, 16000)
print("Static processing completed successfully!")
```

---

## 3. Mode 2: Real-Time Streaming Audio Processing (Chunk-by-Chunk)

**Use Case:** Simulating live microphone streaming, Raspberry Pi edge deployment, or real-time DSP pipelines where audio arrives incrementally in chunks (e.g., 500 ms = 8,000 samples).

The streaming mode uses an internal **FIFO input buffer** and **Overlap-Add (OLA) synthesis buffer** to preserve the Constant Overlap-Add (COLA) condition across consecutive chunk boundaries without clicks, dips, or comb filtering.

### CLI Execution (Just add `--stream`)

To run in streaming mode, simply pass the `--stream` flag!

```bash
# Process in real-time streaming mode (500 ms chunks with OLA)
python src/dsp/denoise/pi_reference_duofilter.py \
    --input data/asr/test_clip_event.wav \
    --output src/dsp/denoise/audio/clean_stream.wav \
    --stream

# Optional: Customize the chunk size (e.g. 4000 samples = 250 ms)
python src/dsp/denoise/pi_reference_duofilter.py \
    --input data/asr/test_clip_event.wav \
    --output src/dsp/denoise/audio/clean_stream_250ms.wav \
    --stream \
    --chunk-size 4000
```

### Live Pipeline Integration (`run_dsp.py`)

To run the denoiser as part of the complete scrub-voice live DSP service (Localizer + Beamformer + VAD + Denoiser):

```bash
python src/dsp/run_dsp.py --wav data/asr/test_clip_event.wav --denoiser duofilter
```

### Real-Time Performance Benchmarks (Single CPU Thread)

| Audio Chunk Size | Budget Time | Processing Time | Headroom Margin | Real-Time Factor (RTF) |
| :--- | :--- | :--- | :--- | :--- |
| **500 ms (8,000 samples)** | 500.0 ms | **~21.6 ms** | **~478.4 ms (95.7% idle)** | **0.043 (23x faster than real-time)** |
| **250 ms (4,000 samples)** | 250.0 ms | **~11.2 ms** | **~238.8 ms (95.5% idle)** | **0.045 (22x faster than real-time)** |

---

## 4. Listening & A/B Comparison (Command Line)

You can listen to audio files directly from the command line without opening a desktop GUI.

### On macOS (Apple Silicon / Intel)
macOS comes with the built-in `afplay` (Audio File Play) utility:

**1. Listen to raw noisy input:**
```bash
afplay data/asr/test_clip_event.wav
```

**2. Listen to Static (Full-File) Clean Output:**
```bash
afplay src/dsp/denoise/audio/clean_static.wav
```

**3. Listen to Streamed (Chunk-by-Chunk) Clean Output:**
```bash
afplay src/dsp/denoise/audio/clean_stream.wav
```

**4. A/B Toggle Comparison:**
Run them sequentially to immediately hear the background noise drop from SNR ~26 dB to ~52 dB:
```bash
afplay data/asr/test_clip_event.wav && afplay src/dsp/denoise/audio/clean_stream.wav
```
*(Tip: Press `Ctrl + C` anytime to stop playback).*

---

### On Linux (Raspberry Pi / Edge Device)
Linux uses ALSA's `aplay`:

```bash
# Play clean static output
aplay src/dsp/denoise/audio/clean_static.wav

# Play clean streamed output
aplay src/dsp/denoise/audio/clean_stream.wav

# Explicit sample format if playing raw PCM:
aplay -D default -r 16000 -f S16_LE src/dsp/denoise/audio/clean_stream.wav
```

---

## 5. Troubleshooting & Tips

* **Comb Filtering / Hollow Sound:**
  * If processing multi-channel microphone array audio, always specify `--channel 0` to select the reference microphone. Never average multi-mic audio with `.mean(axis=1)` prior to delay-alignment beamforming, as spatial delays cause acoustic comb filtering.
  * For rooms with severe wall reflections, use `--phase-preservation` to retain the pristine input phase.
* **Buffer Continuity:**
  * When using streaming in custom loops, always call `denoiser.reset_states()` before starting a new speech utterance to clear the internal FIFO and OLA buffers.
