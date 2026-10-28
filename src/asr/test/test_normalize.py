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

    def test_sentence_end_dot_is_removed_but_decimal_kept(self):
        self.assertEqual(normalize("Slice 45."), "slice 45")
        self.assertEqual(normalize("Zoom 2.5x."), "zoom 2.5 x")

    def test_synonyms_are_not_mapped(self):
        # Mapping "knife" -> "scalpel" is the language understanding stage's job.
        self.assertEqual(normalize("Knife please"), "knife please")


class CorrectionsTest(unittest.TestCase):
    def fixed(self, text: str) -> str:
        return normalize(text, corrections=True)

    def test_dev_errors_are_fixed(self):
        cases = [
            ("Soon.", "zoom"),
            ("Soon out.", "zoom out"),
            ("Soom in to X.", "zoom in 2 x"),
            ("magnify 3 eggs", "magnify 3 x"),
            ("Big off!", "back off"),
            ("Hot robot.", "halt robot"),
            ("hold rowboat", "hold robot"),
            ("Begg 5 slices.", "back 5 slices"),
            ("Okay, next slide please.", "okay next slice please"),
            ("Scampel.", "scalpel"),
            ("Pass the hemmostat.", "pass the hemostat"),
            ("And meet the Mitzen bow.", "and meet the metzenbaum"),
            ("Open the MRI skin.", "open the mri scan"),
            ("close the city skin", "close the ct scan"),
            ("show em or i", "show mri"),
            ("This means MRI.", "dismiss mri"),
            ("Center review.", "center view"),
            ("cameron forward", "camera forward"),
            ("Knock event, first incision.", "log event first incision"),
        ]
        for heard, expected in cases:
            with self.subTest(heard=heard):
                self.assertEqual(self.fixed(heard), expected)

    def test_real_phrases_are_left_alone(self):
        for text in ["Close skin", "Hand me the scope", "We will be done soon",
                     "It's not a big of a deal", "Cameron, can you help", "This means we wait"]:
            with self.subTest(text=text):
                self.assertEqual(self.fixed(text), normalize(text))

    def test_no_test_phrase_is_changed(self):
        # The map must never change something that was really said.
        prompts = Path(__file__).resolve().parents[3] / "data" / "asr" / "prompts"
        for name in ["commands_dev.txt", "commands_heldout.txt", "non_command.txt"]:
            for line in (prompts / name).read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    with self.subTest(phrase=line):
                        self.assertEqual(self.fixed(line), normalize(line))


if __name__ == "__main__":
    unittest.main()
