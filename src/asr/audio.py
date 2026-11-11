"""Load audio and convert it to 16 kHz mono float32 - the format every
engine expects (see src/asr/CLAUDE.md "Input audio"). The DSP team may deliver
44.1 kHz and/or multi-channel audio, so every file goes through downmix +
resample here even if it's already 16 kHz mono.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = downmix(audio)
    audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def downmix(audio: np.ndarray) -> np.ndarray:
    """Average all channels into one. audio is shaped (frames, channels)."""
    return audio.mean(axis=1)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    gcd = np.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)
