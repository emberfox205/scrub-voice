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
    """Like the Vosk engine, fed chunk by chunk (no delimiter heard);
    records each finished utterance and "hears" one word at 0.3 s."""
    name = "fake"

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def accept_chunk(self, chunk):
        return ["zoom"]

    def finish(self, audio):
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
        self.assertEqual(self.stream.feed(chunk(), 100.0), [])
        self.assertEqual(self.stream.feed(chunk(), 100.5), [])
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
        [first] = self.stream.feed(chunk(), 102.0)  # chunks 101.0 and 101.5 had no speech
        self.assertAlmostEqual(first.segment_start, 100.3)
        second = self.stream.end_utterance()
        self.assertAlmostEqual(second.segment_start, 102.3)
        self.assertEqual([len(a) for a in self.engine.calls], [2 * CHUNK, CHUNK])

    def test_timestamp_jitter_is_not_a_gap(self):
        self.stream.feed(chunk(), 100.0)
        self.assertEqual(self.stream.feed(chunk(), 100.6), [])
        self.assertEqual(len(self.engine.calls), 0)

    def test_maximum_length(self):
        stream = UtteranceStream(self.engine, max_utterance_s=1.0)
        self.assertEqual(stream.feed(chunk(), 100.0), [])
        self.assertEqual(len(stream.feed(chunk(), 100.5)), 1)
        self.assertFalse(stream.has_audio())

    def test_end_without_audio(self):
        self.assertIsNone(self.stream.end_utterance())
        self.assertEqual(self.engine.calls, [])

    def test_end_reasons(self):
        self.stream.feed(chunk(), 100.0)
        self.assertEqual(self.stream.end_utterance().end_reason, "dsp_end")
        self.stream.feed(chunk(), 101.0)
        [gap] = self.stream.feed(chunk(), 103.0)
        self.assertEqual(gap.end_reason, "gap")
        self.assertEqual(self.stream.end_utterance("timeout").end_reason, "timeout")
        stream = UtteranceStream(self.engine, max_utterance_s=1.0)
        stream.feed(chunk(), 100.0)
        [longest] = stream.feed(chunk(), 100.5)
        self.assertEqual(longest.end_reason, "max_length")


class ScriptedEngine:
    """Like the Vosk engine; "hears" the scripted partial results, one per
    chunk."""
    name = "scripted"

    def __init__(self, partials: list[str]):
        self.partials = [p.split() for p in partials]
        self.fed = 0
        self.heard: list[str] = []
        self.finished: list[int] = []  # number of samples per finished utterance

    def accept_chunk(self, chunk):
        self.heard = self.partials[min(self.fed, len(self.partials) - 1)]
        self.fed += 1
        return self.heard

    def heard_so_far(self):
        return " ".join(w for w in self.heard if w != "[unk]")

    def finish(self, audio):
        self.finished.append(len(audio))
        return ASRResult("next slice", 0.9, "en", [WordInfo("next", 0.1, 0.4, 0.9)], segment_start=0.0)


class DelimiterTest(unittest.TestCase):
    def test_delimiter_after_a_command_sends_at_once(self):
        engine = ScriptedEngine(["next", "next slice now", "zoom"])
        stream = UtteranceStream(engine)
        self.assertEqual(stream.feed(chunk(), 100.0), [])
        [result] = stream.feed(chunk(), 100.5)  # no DSP "end" needed
        self.assertEqual(result.end_reason, "keyword")
        self.assertEqual(engine.finished, [2 * CHUNK])
        # The next chunk starts a new utterance; the DSP "end" for the old one finds nothing.
        self.assertFalse(stream.has_audio())
        self.assertEqual(stream.feed(chunk(), 101.0), [])
        self.assertEqual(stream.end_utterance().end_reason, "dsp_end")
        self.assertEqual(engine.finished, [2 * CHUNK, CHUNK])

    def test_delimiter_without_command_does_not_end(self):
        for partials in (["now"], ["[unk] over"], ["now zoom"], ["[unk] five now"]):
            with self.subTest(partials=partials):
                stream = UtteranceStream(ScriptedEngine(partials))
                self.assertEqual(stream.feed(chunk(), 100.0), [])
                self.assertTrue(stream.has_audio())


class EmergencyFastPathTest(unittest.TestCase):
    def test_halt_is_sent_early_and_the_utterance_goes_on(self):
        engine = ScriptedEngine(["[unk]", "[unk] halt", "[unk] halt robot", "[unk] halt robot"])
        stream = UtteranceStream(engine)
        self.assertEqual(stream.feed(chunk(), 100.0), [])
        [early] = stream.feed(chunk(), 100.5)
        self.assertEqual((early.text, early.end_reason, early.words, early.confidence),
                         ("halt", "emergency_early", [], None))
        self.assertAlmostEqual(early.segment_start, 100.0)
        self.assertEqual(engine.finished, [])          # nothing was cut off
        self.assertEqual(stream.feed(chunk(), 101.0), [])  # only one early copy per utterance
        self.assertEqual(stream.end_utterance().end_reason, "dsp_end")
        self.assertEqual(engine.finished, [3 * CHUNK])

    def test_next_utterance_can_send_again(self):
        stream = UtteranceStream(ScriptedEngine(["emergency stop"]))
        self.assertEqual(len(stream.feed(chunk(), 100.0)), 1)
        stream.end_utterance()
        self.assertEqual(len(stream.feed(chunk(), 102.0)), 1)

    def test_no_early_copy_for_other_commands(self):
        for partials in (["stop"], ["[unk] stop now"], ["back off"], ["clear area"], ["emergency"]):
            with self.subTest(partials=partials):
                results = UtteranceStream(ScriptedEngine(partials)).feed(chunk(), 100.0)
                self.assertNotIn("emergency_early", [r.end_reason for r in results])

    def test_halt_with_delimiter_sends_the_full_utterance_only(self):
        [result] = UtteranceStream(ScriptedEngine(["halt robot now"])).feed(chunk(), 100.0)
        self.assertEqual(result.end_reason, "keyword")


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
