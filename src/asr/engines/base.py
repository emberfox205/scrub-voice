"""Common interface every ASR engine implements.

A Protocol (from the typing module) is a structural interface: any class
with a matching `name` attribute and `transcribe` method satisfies this,
without needing to inherit from it. This lets benchmark.py treat
faster-whisper and Vosk the same way.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from asr.schema import ASRResult


class ASREngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRResult: ...
