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


class StreamingASREngine(ASREngine, Protocol):
    """An engine that can also take an utterance chunk by chunk and say
    what it has heard so far (Vosk). asr.stream uses this to react to a
    spoken delimiter; other engines (Whisper) get the whole utterance."""

    def accept_chunk(self, chunk: np.ndarray) -> list[str]: ...

    def finish(self, audio: np.ndarray) -> ASRResult: ...
