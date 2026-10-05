# ReSpeaker 6-Mic Circular Array Guide

Quick reference for hardware audio recording, mixer calibration, and playback testing on the Raspberry Pi 4.

---

## 1. Multi-Mic Recording Script (`~/Music/record_6mics.py`)

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

## 2. Earphone Playback on the Pi

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
