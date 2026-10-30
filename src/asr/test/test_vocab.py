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
    COMMAND_VOCAB, INTENT_TERMS, SPOKEN_FORMS,
    check_grammar_leakage, check_leakage, cluster_key, ends_with_delimiter, find_terms, heard_halt,
    term_counts, vosk_grammar_phrases,
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

    def test_synonyms_from_the_evaluation(self):
        entries = vosk_grammar_phrases()
        for needed in ["bring up", "pull up", "hard stop", "make it bigger", "go back",
                       "advance", "enlarge", "evacuate", "patient", "safe height"]:
            self.assertIn(needed, entries)
        for phrase_only in ["pull", "make", "it", "hard", "surgical", "safe"]:
            self.assertNotIn(phrase_only, entries)
        # Left out after measuring (see INTENT_TERMS): "hold" turned "halt"
        # into "hold", "to" turned "two" into "to", the short words caught
        # everyday speech. "wait" sounds like "eight".
        for left_out in ["hold", "on", "jump", "to", "load", "hide", "closer", "wait"]:
            self.assertNotIn(left_out, entries)

    def test_delimiters(self):
        entries = vosk_grammar_phrases()
        for delimiter in ["execute", "over", "now"]:
            self.assertIn(delimiter, entries)
        self.assertNotIn("end", entries)  # sounds like "and"
        self.assertTrue(ends_with_delimiter("next slice now".split()))
        self.assertTrue(ends_with_delimiter("zoom in two x execute".split()))
        self.assertTrue(ends_with_delimiter("[unk] open c t scan [unk] over".split()))
        self.assertFalse(ends_with_delimiter(["now"]))                       # no command before it
        self.assertFalse(ends_with_delimiter("now zoom in".split()))         # not the last word
        self.assertFalse(ends_with_delimiter("[unk] five now".split()))      # a number alone is no command
        self.assertFalse(ends_with_delimiter("go to slice one hundred and".split()))
        # Known: "Don't stop now" ends after "stop" - a false halt is the safe side.
        self.assertTrue(ends_with_delimiter("[unk] stop now".split()))

    def test_halt_phrases(self):
        for heard in ("halt", "[unk] halt robot", "emergency stop", "emergency stop emergency stop", "hard stop robot"):
            with self.subTest(heard=heard):
                self.assertTrue(heard_halt(heard.split()))
        # "freeze": "next three slices" was heard as "next freeze" for a moment.
        for heard in ("stop", "[unk] stop now", "emergency", "emergency clear", "back off", "next freeze", ""):
            with self.subTest(heard=heard):
                self.assertFalse(heard_halt(heard.split()))

    def test_context_words_are_not_terms(self):
        # In the grammar, but "The patient is stable" is not a command.
        self.assertEqual(find_terms(normalize("The patient is stable")), [])
        self.assertEqual(find_terms(normalize("Bring up the patient CT")), ["bring up", "ct"])

    def test_no_test_sentence_in_the_grammar(self):
        # Stricter than the benchmark (dev only): held-out and non-command too.
        check_grammar_leakage(ALL_TEST_PHRASES)

    def test_leakage_check_catches_a_leak(self):
        with self.assertRaises(AssertionError):
            check_leakage(["we go to slice forty five now"], ["Go to slice 45"], "test")
        check_leakage(["freeze"], ["Freeze"], "test")  # a single term is allowed


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
        ref, hyp = normalize("Zoom in two x"), normalize("zoom in three x")
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
