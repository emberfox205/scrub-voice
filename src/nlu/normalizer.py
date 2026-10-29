"""
Lexical Normalizer & Preprocessor for Medical Voice Commands
Target: Sub-0.5ms edge execution on ARM Cortex-A72 (Raspberry Pi 4) / x86.
Loads configurable dictionaries from NLP/dictionaries/ with embedded fallbacks.
"""

import os
import json
import re
import time
from typing import Dict, Tuple, Optional, List

# ---------------------------------------------------------------------------
# Default Embedded Fallbacks (used if JSON files are missing)
# ---------------------------------------------------------------------------

DEFAULT_UNITS: Dict[str, int] = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
}

DEFAULT_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

DEFAULT_SCALES: Dict[str, int] = {
    "hundred": 100,
    "thousand": 1000
}

DEFAULT_ORDINALS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30, "fortieth": 40, "fiftieth": 50
}

DEFAULT_FRACTIONS: Dict[str, str] = {
    "half": "0.5",
    "a half": "0.5",
    "one half": "0.5",
    "quarter": "0.25",
    "a quarter": "0.25",
    "three quarters": "0.75"
}

DEFAULT_IDIOMS: Dict[str, str] = {
    r"\btwice\b": "2x",
    r"\bthrice\b": "3x",
    r"\bhalf\s*(?:a\s*)?(?:times|x)?\b": "0.5x"
}

DEFAULT_FILLERS: List[str] = [
    r"\bplease\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bassistant\b",
    r"\bhey assistant\b",
    r"\bhey\b",
    r"\bnurse\b",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bnow\b",
    r"\bokay\b",
    r"\boh\b"
]

# ---------------------------------------------------------------------------
# LexicalNormalizer Engine
# ---------------------------------------------------------------------------

class LexicalNormalizer:
    """
    High-performance Lexical Normalizer optimized for edge deployment.
    Loads lookup dictionaries dynamically from NLP/dictionaries/.
    """

    def __init__(self, dict_dir: Optional[str] = None):
        if dict_dir is None:
            dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionaries")
        self.dict_dir = dict_dir

        # Load numbers & fillers from JSON, falling back to embedded defaults
        self._load_dictionaries()

        # Build tokens set for fast lookup
        self.all_num_tokens = set(self.units.keys()) | set(self.tens.keys()) | set(self.scales.keys()) | {"point"}

        # Compile regular expressions
        self.re_spaces = re.compile(r"\s+")
        self.re_punctuation = re.compile(r"[^\w\s\.\-]")
        self.re_multiplier = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:times|x)\b", re.IGNORECASE)
        self.compiled_fillers = [re.compile(p, re.IGNORECASE) for p in self.filler_patterns]
        self.compiled_idioms = [(re.compile(p, re.IGNORECASE), repl) for p, repl in self.idioms.items()]

    def _load_dictionaries(self):
        """Loads numbers.json and fillers.json with error tolerance."""
        numbers_path = os.path.join(self.dict_dir, "numbers.json")
        fillers_path = os.path.join(self.dict_dir, "fillers.json")

        # Load numbers
        if os.path.exists(numbers_path):
            try:
                with open(numbers_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.units = data.get("units", DEFAULT_UNITS)
                    self.tens = data.get("tens", DEFAULT_TENS)
                    self.scales = data.get("scales", DEFAULT_SCALES)
                    self.ordinals = data.get("ordinals", DEFAULT_ORDINALS)
                    self.fractions = data.get("fractions", DEFAULT_FRACTIONS)
                    self.idioms = data.get("idioms", DEFAULT_IDIOMS)
            except Exception as e:
                self._use_default_numbers()
        else:
            self._use_default_numbers()

        # Load fillers
        if os.path.exists(fillers_path):
            try:
                with open(fillers_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.filler_patterns = data.get("filler_patterns", DEFAULT_FILLERS)
            except Exception as e:
                self.filler_patterns = DEFAULT_FILLERS
        else:
            self.filler_patterns = DEFAULT_FILLERS

    def _use_default_numbers(self):
        self.units = DEFAULT_UNITS
        self.tens = DEFAULT_TENS
        self.scales = DEFAULT_SCALES
        self.ordinals = DEFAULT_ORDINALS
        self.fractions = DEFAULT_FRACTIONS
        self.idioms = DEFAULT_IDIOMS

    def strip_fillers(self, text: str) -> str:
        """Strips conversational noise and polite filler phrases."""
        for pattern in self.compiled_fillers:
            text = pattern.sub(" ", text)
        return text

    def replace_idioms_and_ordinals(self, text: str) -> str:
        """Replaces common spoken idioms, multipliers, and ordinals."""
        for pattern, repl in self.compiled_idioms:
            text = pattern.sub(repl, text)

        for ord_word, ord_val in self.ordinals.items():
            pattern = r"\b" + re.escape(ord_word) + r"\b"
            text = re.sub(pattern, str(ord_val), text, flags=re.IGNORECASE)

        return text

    def _parse_words_to_integer(self, words: list[str]) -> Optional[int]:
        if not words:
            return None

        total = 0
        current = 0

        for word in words:
            if word in self.units:
                current += self.units[word]
            elif word in self.tens:
                current += self.tens[word]
            elif word in self.scales:
                scale = self.scales[word]
                if current == 0:
                    current = 1
                current *= scale
                if scale >= 1000:
                    total += current
                    current = 0
            else:
                return None

        return total + current

    def _parse_decimal_part(self, words: list[str]) -> Optional[str]:
        digits = []
        for w in words:
            if w in self.units:
                digits.append(str(self.units[w]))
            elif w in self.tens:
                digits.append(str(self.tens[w]))
            else:
                return None
        return "".join(digits) if digits else None

    def _parse_numeric_sequence(self, seq: list[str]) -> Optional[str]:
        if not seq:
            return None

        if "point" in seq:
            idx = seq.index("point")
            int_words = seq[:idx]
            dec_words = seq[idx + 1:]

            if not int_words:
                int_part = 0
            else:
                int_part = self._parse_words_to_integer(int_words)
                if int_part is None:
                    return None

            if not dec_words:
                return str(int_part)

            dec_part = self._parse_decimal_part(dec_words)
            if dec_part is None:
                return None

            return f"{int_part}.{dec_part}"
        else:
            int_val = self._parse_words_to_integer(seq)
            if int_val is not None:
                return str(int_val)
            return None

    def convert_spoken_numbers(self, text: str) -> str:
        tokens = text.split()
        new_tokens = []
        i = 0
        n = len(tokens)

        while i < n:
            token_lower = tokens[i].lower()

            if token_lower in self.all_num_tokens:
                j = i
                seq = []
                while j < n and tokens[j].lower() in self.all_num_tokens:
                    seq.append(tokens[j].lower())
                    j += 1

                converted = self._parse_numeric_sequence(seq)
                if converted is not None:
                    new_tokens.append(converted)
                    i = j
                    continue

            new_tokens.append(tokens[i])
            i += 1

        return " ".join(new_tokens)

    def normalize(self, raw_text: str) -> str:
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text.lower()
        text = self.re_punctuation.sub(" ", text)
        text = self.replace_idioms_and_ordinals(text)
        text = self.convert_spoken_numbers(text)
        text = self.strip_fillers(text)
        text = self.re_multiplier.sub(r"\1x", text)
        text = self.re_spaces.sub(" ", text).strip()
        return text

    def process(self, raw_transcript: str) -> Tuple[str, str]:
        return raw_transcript, self.normalize(raw_transcript)


# ---------------------------------------------------------------------------
# Verification Tests & Benchmark
# ---------------------------------------------------------------------------

def run_tests_and_benchmark():
    normalizer = LexicalNormalizer()
    print(f"Dictionaries loaded from: {normalizer.dict_dir}")

    test_cases = [
        ("Please zoom in two point five times on coronal view", "zoom in 2.5x on coronal view"),
        ("Zoom in point five times", "zoom in 0.5x"),
        ("Zoom in zero point seven five times please", "zoom in 0.75x"),
        ("Could you zoom out two times assistant", "zoom out 2x"),
        ("Zoom in three point five x on axial view", "zoom in 3.5x on axial view"),
        ("Zoom in twice please", "zoom in 2x"),
        ("Magnify image thrice", "magnify image 3x"),
        ("Zoom in half a times", "zoom in 0.5x"),
        ("Can you move view up by twenty five pixels assistant", "move view up by 25 pixels"),
        ("Move display left one hundred fifty pixels", "move display left 150 pixels"),
        ("Jump to slice number forty five please nurse", "jump to slice number 45"),
        ("Advance forward three slices", "advance forward 3 slices"),
        ("Go back twelve slices", "go back 12 slices"),
        ("Go to the first slice please", "go to the 1 slice"),
        ("Select third series", "select 3 series"),
        ("Please switch to bone window thank you", "switch to bone window"),
        ("Nurse could you invert image contrast now", "invert image contrast"),
        ("Assistant reset view please", "reset view"),
        ("Vitals are stabilizing nicely", "vitals are stabilizing nicely")
    ]

    all_passed = True
    for idx, (raw, expected) in enumerate(test_cases, 1):
        norm = normalizer.normalize(raw)
        if norm != expected:
            print(f"Test {idx:02d} [FAIL]: Raw='{raw}', Got='{norm}', Expected='{expected}'")
            all_passed = False
        else:
            print(f"Test {idx:02d} [PASS] '{raw}' -> '{norm}'")

    if not all_passed:
        return

    # Benchmark
    sample = "Please zoom in two point five times on coronal view thank you assistant"
    N = 10000
    t0 = time.perf_counter()
    for _ in range(N):
        _ = normalizer.normalize(sample)
    total_time = time.perf_counter() - t0
    avg_ms = (total_time / N) * 1000

    print(f"\nAverage Latency: {avg_ms:.4f} ms per utterance ({int(N / total_time):,} calls/sec)")
    print(f"Budget Verification: {avg_ms:.4f} ms <= 0.5000 ms -> {'PASS' if avg_ms < 0.5 else 'FAIL'}")

if __name__ == "__main__":
    run_tests_and_benchmark()
