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


class NormalizeTest(unittest.TestCase):
    def test_reference_and_vosk_agree(self):
        # The phrase lists are written with capitals and punctuation; Vosk
        # writes lowercase words (sound-alike spellings in partial results).
        pairs = [
            ("Zoom out one point five x", "zoom out one point five x"),
            ("Go to slice one hundred twenty", "go to slice one hundred twenty"),
            ("Display X-ray", "display x-ray"),
            ("Display X-ray", "display x ray"),
            ("Open CT scan", "open c t scan"),
            ("Show MRI", "show m r i"),
            ("Pass the hemostat", "pass the hemo stat"),
            ("Log event, first incision", "log event first incision"),
            ("Um, hand me the scalpel please", "hand me the scalpel please"),
        ]
        for reference, hypothesis in pairs:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(normalize(reference), normalize(hypothesis))

    def test_examples(self):
        self.assertEqual(normalize("Display X-ray"), "display xray")
        self.assertEqual(normalize("hand me the hemo stat"), "hand me the hemostat")
        self.assertEqual(normalize("Um, hand me the scalpel please"), "hand me the scalpel please")
        self.assertEqual(normalize("log event [unk] note [unk]"), "log event note")

    def test_punctuation_is_removed(self):
        self.assertEqual(normalize("Don't stop now, we are almost done."), "dont stop now we are almost done")

    def test_synonyms_are_not_mapped(self):
        # Mapping "knife" -> "scalpel" is the language understanding stage's job.
        self.assertEqual(normalize("Knife please"), "knife please")


if __name__ == "__main__":
    unittest.main()
