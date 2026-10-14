"""Unit tests for asr.normalize. Run from the repository root:
    python -m unittest discover -s src/asr/test -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path, like the scripts

from asr.normalize import normalize, words_to_digits


def digits(text: str) -> str:
    return " ".join(words_to_digits(text.split()))


class WordsToDigitsTest(unittest.TestCase):
    def test_below_hundred(self):
        self.assertEqual(digits("zero seven nineteen"), "0 7 19")
        self.assertEqual(digits("forty five"), "45")
        self.assertEqual(digits("twenty"), "20")

    def test_hundreds(self):
        self.assertEqual(digits("one hundred twenty"), "120")
        self.assertEqual(digits("one hundred and twenty"), "120")
        self.assertEqual(digits("two hundred five"), "205")
        self.assertEqual(digits("a hundred"), "100")
        self.assertEqual(digits("2 hundred"), "200")
        self.assertEqual(digits("one hundred twenty three"), "123")

    def test_decimals(self):
        self.assertEqual(digits("one point five"), "1.5")
        self.assertEqual(digits("two point two five"), "2.25")

    def test_words_around_numbers_are_kept(self):
        self.assertEqual(digits("one hundred and then"), "100 and then")
        self.assertEqual(digits("five point the camera"), "5 point the camera")
        self.assertEqual(digits("a slice"), "a slice")

    def test_separate_numbers_stay_separate(self):
        self.assertEqual(digits("one two three"), "1 2 3")
        self.assertEqual(digits("twenty thirty"), "20 30")

    def test_trailing_zeros(self):
        self.assertEqual(digits("2.0 2.50"), "2 2.5")


class NormalizeTest(unittest.TestCase):
    def test_reference_and_whisper_style_agree(self):
        # The test lists write numbers as words, Whisper writes digits.
        pairs = [
            ("Zoom in two x", "Zoom in 2x."),
            ("Zoom out one point five x", "Zoom out 1.5x."),
            ("Go to slice forty five", "Go to slice 45."),
            ("Go to slice one hundred twenty", "Go to slice 120."),
            ("Camera left five centimeters", "Camera left 5 cm."),
            ("Camera left five centimeters", "Camera left 5cm"),
            ("Tilt down fifteen degrees", "Tilt down 15°."),
            ("Display X-ray", "Display x-ray"),
            ("Display X-ray", "Display xray."),
            ("Open CT scan", "Open C.T. scan."),
            ("Show MRI", "Show M R I"),
            ("Give me a ten blade", "Give me a 10 blade."),
            ("Standby robot", "Stand by, robot."),
            ("Zoom in two x", "Zoom in 2×"),
        ]
        for reference, hypothesis in pairs:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(normalize(reference), normalize(hypothesis))

    def test_examples(self):
        self.assertEqual(normalize("Display X-ray"), "display xray")
        self.assertEqual(normalize("hand me the hemo stat"), "hand me the hemostat")
        self.assertEqual(normalize("Um, hand me the scalpel please"), "hand me the scalpel please")
        self.assertEqual(normalize("log event [unk] note [unk]"), "log event note")

    def test_json_field(self):
        # The `normalized` field of the output JSON (src/asr/CLAUDE.md).
        self.assertEqual(normalize("Zoom in two x, please.", corrections=True, drop_politeness=True), "zoom in 2 x")
        self.assertEqual(normalize("Okay, next slice please", corrections=True, drop_politeness=True), "okay next slice")

    def test_sentence_end_dot_is_removed_but_decimal_kept(self):
        self.assertEqual(normalize("Slice 45."), "slice 45")
        self.assertEqual(normalize("Zoom 2.5x."), "zoom 2.5 x")

    def test_synonyms_are_not_mapped(self):
        # Mapping "knife" -> "scalpel" is the language understanding stage's job.
        self.assertEqual(normalize("Knife please"), "knife please")


if __name__ == "__main__":
    unittest.main()
