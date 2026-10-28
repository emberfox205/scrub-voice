"""Streaming entry point: 0.5 s audio chunks from the DSP front end in, one
ASR result (the JSON contract in src/asr/CLAUDE.md) per utterance out -
plus an early copy for halt commands (below).

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

Emergency fast path: when the partial result first contains a halt command
(vocab.HALT_PHRASES: "halt", "emergency stop", "hard stop"), an
early copy is sent at once ("emergency_early": the words heard so far, no
word times or confidence - partial results have none), and the utterance
goes on and ends as usual. NLU gets the halt twice; halting twice is
harmless, and cutting the utterance instead would send the rest ("robot"
of "halt robot") as a command of its own.

Usage:
    stream = UtteranceStream(CONFIGS[PRODUCTION_CONFIG]())
    for chunk, capture_time in dsp_chunks:
        for result in stream.feed(chunk, capture_time):
            send(result.to_json())
    result = stream.end_utterance()   # when the DSP reports end of speech
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from asr.audio import TARGET_SR
from asr.schema import ASRResult, WordInfo
from asr.vocab import ends_with_delimiter, heard_halt

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
        self._halt_sent = False     # early copy of a halt already sent for this utterance

    def has_audio(self) -> bool:
        return bool(self._chunks)

    def feed(self, chunk: np.ndarray, capture_time: float) -> list[ASRResult]:
        """Add one speech chunk (capture_time = epoch seconds of its first
        sample). Returns what to send now, in order - usually nothing: the
        previous utterance if this chunk starts a new one (gap); then the
        current one if it ended with a delimiter or reached the maximum
        length, or else the early copy of a halt command first heard in it."""
        chunk = chunk_to_mono(chunk)
        results: list[ASRResult] = []
        if self._chunks and capture_time - self._expected_next > self.gap_tolerance_s:
            results.append(self.end_utterance("gap"))
        if not self._chunks:
            self._start_time = capture_time
            self._halt_sent = False
        self._chunks.append(chunk)
        self._expected_next = capture_time + len(chunk) / TARGET_SR
        heard = self.engine.accept_chunk(chunk)
        if ends_with_delimiter(heard):
            results.append(self.end_utterance("keyword"))
        elif self._expected_next - self._start_time >= self.max_utterance_s:
            results.append(self.end_utterance("max_length"))
        elif not self._halt_sent and heard_halt(heard):
            # The utterance goes on; its full result follows when it ends.
            self._halt_sent = True
            results.append(ASRResult(text=self.engine.heard_so_far(), confidence=None, language="en",
                                     words=[], segment_start=round(self._start_time, 3),
                                     end_reason="emergency_early"))
        return results

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
