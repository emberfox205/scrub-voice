# DeepFilterNet ONNX Inference & Audio Playback Guide

This guide covers how to execute your exported DeepFilterNet ONNX model to denoise an audio file, and how to instantly listen to the results directly from your terminal.

---

## 1. Prerequisites
You should also ensure that these three files are ready:
* `pi_inference.py` (The main inference script)
* `deepfilternet.onnx` (Your exported model)
* `test.wav` (Your noisy input audio)

---

## 2. Running the Inference

To denoise your audio file, simply execute the Python script. Ensure that the `INPUT_WAV` variable inside the script is pointing to your `test.wav` file.

```bash
python pi_inference.py
```

**Expected Output:**
If successful, the script will process the file through the ONNX Runtime and save the denoised result to your disk. You should see a message similar to:
> `Saved enhanced audio to clean_audio.wav`

---

## 3. Listening to the Audio (Command Line)

You don't need to open a media player application to listen to your `.wav` files! Both macOS and Linux (Raspberry Pi) have built-in terminal audio players.

### On macOS (Apple Silicon / Intel)
Use the built-in `afplay` (Audio File Play) command:

**Listen to the noisy input:**
```bash
afplay test.wav
```

**Listen to the clean output:**
```bash
afplay clean_audio.wav
```
*(Tip: If you want to stop playback early, press `Ctrl + C` in your terminal).*

### On Linux (Raspberry Pi)
Linux uses the Advanced Linux Sound Architecture (ALSA) player `aplay`:

```bash
aplay clean_audio.wav
```