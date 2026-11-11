# ReSpeaker 6-Mic Circular Array Guide

Quick reference for hardware audio recording, mixer calibration, and playback testing on the Raspberry Pi 4.

---

## 1. Quick In-Repo Recording Tool (`test/record_audio.py`)

A safe, DMA-aligned hardware recording utility conforming strictly to `AGENTS.md` rules:
- **RAM-first staging:** Captures into `/dev/shm` in RAM first before writing to disk, protecting the Raspberry Pi microSD card from controller freezes.
- **DMA Ring Buffer:** Configures `--buffer-size=65536 --period-size=8192` to eliminate ALSA XRUNs.
- **ALSA Driver Cleanup:** Intercepts `Ctrl+C` (SIGINT) to cleanly call `snd_pcm_close()`.
- **Live VU Meter & Tap Test:** Renders live dB meter and prints per-mic channel energy upon completion.

### Usage
```bash
# 1. Quick 5-second recording (saves to data/wav_files/test_recording_<timestamp>.wav):
python test/record_audio.py

# 2. Custom duration (e.g. 10s) and custom destination:
python test/record_audio.py -d 10 -o data/wav_files/surgical_command_01.wav

# 3. Manual interactive mode (press Enter to stop):
python test/record_audio.py --manual

# 4. Save single-channel mono (Mic 1) instead of 6-mic array:
python test/record_audio.py -d 5 --mono

# 5. Record and immediately transcribe/evaluate with ASR + NLU:
python test/record_audio.py -d 5 --eval
```

---

## 2. Legacy Standalone Script (`~/Music/record_6mics.py`)

A synchronized recording script is located in `~/Music/record_6mics.py` on the Pi. It captures the raw 8-channel hardware stream, drops dummy channels (6 & 7), and splits the active microphones into 6 individual WAV files (`mic_01.wav` – `mic_06.wav`).

### Usage

```bash
cd ~/Music

# Record for 5 seconds (default)
python3 record_6mics.py 5

# Record for custom duration (e.g., 10 seconds)
python3 record_6mics.py 10
```

### Output & Mic Health Check

The script outputs:

* `mic_01.wav` to `mic_06.wav` (16 kHz, 16-bit mono PCM).
* A console table printing peak energy levels per channel. Use this for a **tap test**: lightly tap near each microphone to confirm all 6 sensors register activity.

---

## 3. Earphone Playback on the Pi

To listen to recordings directly from the Pi with earphones:

```bash
# Playback through Pi 4 onboard 3.5mm jack
aplay -D plughw:Headphones,0 mic_01.wav

# Playback through ReSpeaker HAT 3.5mm jack
aplay -D plughw:seeed8micvoicec,0 mic_01.wav

# Play all 6 channels sequentially
for f in mic_0{1..6}.wav; do
    echo "Playing $f..."
    aplay -D plughw:Headphones,0 "$f"
done
```

---

## 3. AlsaMixer Calibration

Improper gain staging causes harsh digital clipping or excessive background hiss.

### Opening the Mixer

```bash
alsamixer
```

Press **`F6`** $\rightarrow$ select **`seeed-8mic-voicecard`**.

> **Crucial for DSP:** Always keep channels 1 through 6 set to identical gain values to preserve array phase and amplitude balance for beamforming.

---

## 4. Saving Mixer Settings Permanently

Mixer changes reset upon reboot unless explicitly stored in both standard ALSA and the Seeed service state file:

```bash
sudo alsactl store && sudo alsactl store -f /etc/voicecard/ac108_6mic.state
```
