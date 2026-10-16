"""Unit tests for asr.vocab. Run from the repository root:
    python -m unittest discover -s src/asr/test -v
"""

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))  # src/ on sys.path, like the scripts

from asr.normalize import normalize
from asr.vocab import (
    COMMAND_VOCAB, INTENT_TERMS, SPOKEN_FORMS, WHISPER_PROMPT, WHISPER_PROMPT_SHORT,
    check_all_biasing_leakage, check_leakage, cluster_key, find_terms, term_counts,
    vosk_grammar_phrases,
)

PROMPTS_DIR = SRC_DIR.parent / "data" / "asr" / "prompts"
SCHEMA_INTENTS = {
    "pick_tool", "park_arm", "clear_emergency", "halt_emergency", "open_scan", "close_scan",
    "zoom_display", "navigate_slice", "log_event", "move_camera", "tilt_camera", "reset_camera",
}


def phrase_list(name: str) -> list[str]:
    lines = (PROMPTS_DIR / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


ALL_TEST_PHRASES = [p for f in ("commands_dev.txt", "commands_heldout.txt", "non_command.txt") for p in phrase_list(f)]


class VocabularyTest(unittest.TestCase):
    def test_every_schema_intent_has_terms(self):
        self.assertEqual(set(INTENT_TERMS), SCHEMA_INTENTS)

    def test_spoken_forms_normalize_back_to_the_term(self):
        for term, spoken in SPOKEN_FORMS.items():
            with self.subTest(term=term):
                self.assertEqual(normalize(spoken), normalize(term))

    def test_grammar_entries(self):
        entries = vosk_grammar_phrases()
        self.assertTrue(all(e == e.lower() for e in entries))
        for needed in ["go to", "back off", "hemo stat", "x ray", "hundred", "point", "forty", "metzenbaum"]:
            self.assertIn(needed, entries)
        self.assertNotIn("hemostat", entries)  # not in the Vosk model's dictionary
        # Words that only occur inside a phrase are not entries of their own
        # ("to" next to "two" gave "go two slice")...
        for phrase_only in ["to", "go", "off", "ray", "hemo"]:
            self.assertNotIn(phrase_only, entries)
        # ...but words needed alone stay: "back 5 slices", "ten", "x" in "2 x".
        for needed_alone in ["back", "ten", "x", "zoom"]:
            self.assertIn(needed_alone, entries)

    def test_no_test_sentence_in_any_biasing_text(self):
        # Stricter than the benchmark (dev only): held-out and non-command too.
        check_all_biasing_leakage(ALL_TEST_PHRASES)

    def test_leakage_check_catches_a_leak(self):
        with self.assertRaises(AssertionError):
            check_leakage(["we go to slice forty five now"], ["Go to slice 45"], "test")
        check_leakage(["freeze"], ["Freeze"], "test")  # a single term is allowed

    def test_prompts_contain_the_hard_terms(self):
        for prompt in (WHISPER_PROMPT, WHISPER_PROMPT_SHORT):
            for term in ["scalpel", "Metzenbaum", "hemostat", "forceps", "CT", "MRI", "X-ray"]:
                with self.subTest(term=term):
                    self.assertIn(term, prompt)


class TermMetricsTest(unittest.TestCase):
    def test_longest_term_first(self):
        self.assertEqual(find_terms(normalize("Back off")), ["back off"])
        self.assertEqual(find_terms(normalize("Back five slices")), ["back", "5", "slices"])
        self.assertEqual(find_terms(normalize("Give me a ten blade")), ["give me", "10 blade"])
        self.assertEqual(find_terms(normalize("Zoom out one point five x")), ["zoom out", "1.5"])

    def test_numbers_are_terms(self):
        self.assertEqual(find_terms(normalize("Go to slice one hundred twenty")), ["go to", "slice", "120"])

    def test_everyday_in_out_are_not_terms(self):
        self.assertEqual(find_terms(normalize("The family is waiting outside")), [])
        self.assertEqual(find_terms(normalize("How was your weekend")), [])

    def test_wrong_number_counts_as_missed_and_invented(self):
        ref, hyp = normalize("Zoom in two x"), normalize("zoom in 3x")
        self.assertEqual(term_counts(ref, hyp), (2, 2, 1))

    def test_every_vocab_term_is_found_in_itself(self):
        for term in COMMAND_VOCAB:
            with self.subTest(term=term):
                self.assertEqual(find_terms(normalize(term)), [normalize(term)])

    def test_cluster_key_prefers_notes(self):
        self.assertEqual(cluster_key("pick_tool", "hand scalpel"), "pick_tool")
        self.assertEqual(cluster_key("", "hand scalpel"), "hand")


if __name__ == "__main__":
    unittest.main()
