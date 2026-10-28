"""Streaming entry point: 0.5 s audio chunks from the DSP front end in, one
ASR result (the JSON contract in src/asr/CLAUDE.md) per utterance out.

The DSP front end only passes on chunks in which its voice activity
detector heard speech, already beamformed and cleaned. So ASR does not run
its own speech detection: a run of speech chunks is one utterance. The
utterance ends when (`end_reason` in the result)
- the speaker says a delimiter after a command, "next slice now"
  ("keyword"; vocab.DELIMITERS) - sent at once, without waiting for the
  DSP to notice the silence; the next chunk starts a new utterance,
- the DSP says speech has stopped (`end_utterance()`, "dsp_end"),
- the next chunk does not follow on directly in time ("gap": chunks without
  speech in between were not sent),
- the service waited too long for the next chunk (run_asr.py, "timeout"), or
- it gets longer than `max_utterance_s` ("max_length", safety net for a
  stuck detector - also where the delimiter helps: noise can keep the
  detector on).
The Vosk engine gets every chunk as it arrives, so its partial result can
be checked for the delimiter; its result equals transcribing the whole
utterance in one go - exactly how it was benchmarked (docs/ASR_FINDINGS.md),
so the measured accuracy holds.

Usage:
    stream = UtteranceStream(CONFIGS[PRODUCTION_CONFIG]())
    for chunk, capture_time in dsp_chunks:
        result = stream.feed(chunk, capture_time)
        if result: send(result.to_json())
    result = stream.end_utterance()   # when the DSP reports end of speech
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from asr.audio import TARGET_SR
from asr.schema import ASRResult, WordInfo
from asr.vocab import ends_with_delimiter

if TYPE_CHECKING:  # only for the type hint; the tests pass a fake engine
    from asr.engines.vosk_engine import VoskEngine


def chunk_to_mono(chunk: np.ndarray) -> np.ndarray:
    """DSP chunk -> 1-D float32 in [-1, 1]. Accepts the DSP's (channels,
    samples) layout (averaged to one channel) and int16 samples."""
    chunk = np.asarray(chunk)
    if np.issubdtype(chunk.dtype, np.integer):
        chunk = chunk.astype(np.float32) / 32768.0
    if chunk.ndim == 2:
        chunk = chunk.mean(axis=0)
    return chunk.astype(np.float32)


class UtteranceStream:
    def __init__(self, engine: VoskEngine, max_utterance_s: float = 15.0, gap_tolerance_s: float = 0.25) -> None:
        """engine: the Vosk engine (bench_configs.PRODUCTION_CONFIG), 16 kHz input.
        max_utterance_s: a log_event with a comment takes ~10 s.
        gap_tolerance_s: timestamp jitter allowed between two chunks; a real
        gap is at least one missing 0.5 s chunk."""
        self.engine = engine
        self.max_utterance_s = max_utterance_s
        self.gap_tolerance_s = gap_tolerance_s
        self._chunks: list[np.ndarray] = []
        self._start_time = 0.0      # capture time of the utterance's first chunk
        self._expected_next = 0.0   # capture time the next chunk should have

    def has_audio(self) -> bool:
        return bool(self._chunks)

    def feed(self, chunk: np.ndarray, capture_time: float) -> Optional[ASRResult]:
        """Add one speech chunk (capture_time = epoch seconds of its first
        sample). Returns the finished previous utterance if this chunk
        starts a new one, or the current one if it ended with a delimiter
        or reached the maximum length; otherwise None."""
        chunk = chunk_to_mono(chunk)
        result = None
        if self._chunks and capture_time - self._expected_next > self.gap_tolerance_s:
            result = self.end_utterance("gap")
        if not self._chunks:
            self._start_time = capture_time
        self._chunks.append(chunk)
        self._expected_next = capture_time + len(chunk) / TARGET_SR
        heard = self.engine.accept_chunk(chunk)
        # Only one result can be returned: a delimiter in the very chunk
        # after a gap is handled with the next chunk or at the end.
        if result is None and ends_with_delimiter(heard):
            return self.end_utterance("keyword")
        if result is None and self._expected_next - self._start_time >= self.max_utterance_s:
            result = self.end_utterance("max_length")
        return result

    def end_utterance(self, reason: str = "dsp_end") -> Optional[ASRResult]:
        """Transcribe what was collected (None if nothing was). reason: see
        the module notes ("dsp_end" when the DSP reports end of speech)."""
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        start_time = self._start_time
        self._chunks = []

        result = self.engine.finish(audio)
        result.end_reason = reason
        # The engine's word times count from the first chunk's start. The
        # contract wants segment_start = where the speech starts, and word
        # times relative to it.
        offset = result.words[0].start if result.words else 0.0
        result.segment_start = round(start_time + offset, 3)
        result.words = [
            WordInfo(word=w.word, start=round(w.start - offset, 3), end=round(w.end - offset, 3), confidence=w.confidence)
            for w in result.words
        ]
        return result
