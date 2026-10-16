"""Tests for the Vosk engine: the log_event splice (no model needed) and
word times on a real clip (needs the downloaded model, skipped otherwise).
Run from the repository root:
    python -m unittest discover -s src/asr/test -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path, like the scripts

from asr.audio import load_audio
from asr.engines.vosk_engine import MODEL_NAME, MODELS_DIR, VoskEngine, _starts_log_event, splice_log_event

REPO_ROOT = Path(__file__).resolve().parents[3]


def words(*items: tuple[str, float, float]) -> list[dict]:
    return [{"word": w, "start": s, "end": e, "conf": 1.0} for w, s, e in items]


class SpliceTest(unittest.TestCase):
    def test_header_from_grammar_text_from_freeform(self):
        grammar = words(("log", 0.1, 0.3), ("event", 0.3, 0.7), ("[unk]", 0.8, 1.5), ("note", 1.6, 1.8))
        free = words(("blog", 0.1, 0.3), ("event", 0.3, 0.7), ("first", 0.8, 1.0),
                     ("incision", 1.0, 1.5), ("note", 1.6, 1.8), ("stable", 1.9, 2.2))
        spliced = [w["word"] for w in splice_log_event(grammar, free)]
        self.assertEqual(spliced, ["log", "event", "first", "incision", "note", "stable"])

    def test_leading_unk_is_skipped(self):
        grammar = words(("[unk]", 0.0, 0.2), ("record", 0.3, 0.6), ("milestone", 0.6, 1.1), ("[unk]", 1.2, 1.6))
        free = words(("um", 0.0, 0.2), ("record", 0.3, 0.6), ("milestone", 0.6, 1.1), ("biopsy", 1.2, 1.6))
        spliced = [w["word"] for w in splice_log_event(grammar, free)]
        self.assertEqual(spliced, ["record", "milestone", "biopsy"])

    def test_trigger_alone_falls_back_to_freeform(self):
        # Dev set: "Hello" came out of the grammar as "log".
        grammar = words(("log", 0.2, 0.6))
        free = words(("hello", 0.2, 0.6))
        self.assertEqual(splice_log_event(grammar, free), free)

    def test_no_header_falls_back_to_freeform(self):
        grammar = words(("please", 0.0, 0.3), ("log", 0.3, 0.5))
        free = words(("please", 0.0, 0.3), ("log", 0.3, 0.5), ("this", 0.5, 0.8))
        self.assertEqual(splice_log_event(grammar, free), free)

    def test_trigger(self):
        self.assertTrue(_starts_log_event(words(("[unk]", 0, 1), ("log", 1, 2))))
        self.assertTrue(_starts_log_event(words(("could", 0, 1), ("you", 1, 2), ("record", 2, 3))))
        self.assertFalse(_starts_log_event(words(("next", 0, 1), ("slice", 1, 2))))
        self.assertFalse(_starts_log_event(words(("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("log", 3, 4))))


@unittest.skipUnless((MODELS_DIR / MODEL_NAME).exists(), "Vosk model not downloaded")
class RealClipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = load_audio(REPO_ROOT / "data" / "asr" / "sample_01.wav")[: 16000 * 4]
        cls.engine = VoskEngine()

    def test_same_audio_same_words_and_times(self):
        # Vosk's own times keep counting across Reset(); the engine shifts
        # them back, so the second run must match the first exactly.
        first = self.engine.transcribe(self.audio)
        second = self.engine.transcribe(self.audio)
        self.assertTrue(first.words)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertLess(first.words[-1].end, 4.0 + 0.01)


if __name__ == "__main__":
    unittest.main()
