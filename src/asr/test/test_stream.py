"""Tests for asr.stream (a fake engine, no model needed). Run from the
repository root:
    python -m unittest discover -s src/asr/test -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path, like the scripts

from asr.schema import ASRResult, WordInfo
from asr.stream import UtteranceStream, chunk_to_mono

SR = 16000
CHUNK = 8000  # 0.5 s


class FakeEngine:
    """Records what it was given; "hears" one word starting at 0.3 s."""
    name = "fake"

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def transcribe(self, audio, sr=16000):
        self.calls.append(audio)
        words = [WordInfo("zoom", 0.3, 0.6, 0.9), WordInfo("in", 0.6, 0.8, 0.8)]
        return ASRResult("zoom in", 0.85, "en", words, segment_start=0.0)


def chunk(value: float = 0.1) -> np.ndarray:
    return np.full(CHUNK, value, dtype=np.float32)


class UtteranceStreamTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.stream = UtteranceStream(self.engine)

    def test_one_utterance_from_consecutive_chunks(self):
        self.assertIsNone(self.stream.feed(chunk(), 100.0))
        self.assertIsNone(self.stream.feed(chunk(), 100.5))
        result = self.stream.end_utterance()
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(self.engine.calls[0]), 2 * CHUNK)
        # segment_start = first chunk's capture time + where the speech starts
        self.assertAlmostEqual(result.segment_start, 100.3)
        # word times relative to segment_start
        self.assertEqual([(w.start, w.end) for w in result.words], [(0.0, 0.3), (0.3, 0.5)])

    def test_gap_starts_a_new_utterance(self):
        self.stream.feed(chunk(), 100.0)
        self.stream.feed(chunk(), 100.5)
        first = self.stream.feed(chunk(), 102.0)  # chunks 101.0 and 101.5 had no speech
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first.segment_start, 100.3)
        second = self.stream.end_utterance()
        self.assertAlmostEqual(second.segment_start, 102.3)
        self.assertEqual([len(a) for a in self.engine.calls], [2 * CHUNK, CHUNK])

    def test_timestamp_jitter_is_not_a_gap(self):
        self.stream.feed(chunk(), 100.0)
        self.assertIsNone(self.stream.feed(chunk(), 100.6))
        self.assertEqual(len(self.engine.calls), 0)

    def test_maximum_length(self):
        stream = UtteranceStream(self.engine, max_utterance_s=1.0)
        self.assertIsNone(stream.feed(chunk(), 100.0))
        self.assertIsNotNone(stream.feed(chunk(), 100.5))
        self.assertFalse(stream.has_audio())

    def test_end_without_audio(self):
        self.assertIsNone(self.stream.end_utterance())
        self.assertEqual(self.engine.calls, [])


class ChunkToMonoTest(unittest.TestCase):
    def test_dsp_layout_and_int16(self):
        multi = np.stack([np.full(CHUNK, 0.2), np.full(CHUNK, 0.4)]).astype(np.float32)  # (channels, samples)
        mono = chunk_to_mono(multi)
        self.assertEqual(mono.shape, (CHUNK,))
        self.assertAlmostEqual(float(mono[0]), 0.3, places=6)
        self.assertAlmostEqual(float(chunk_to_mono(np.full(4, 16384, dtype=np.int16))[0]), 0.5)
        self.assertEqual(chunk_to_mono(np.zeros(4, dtype=np.float64)).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
