"""Text normalization for scoring only (benchmark), used identically on
reference and hypothesis. It is not applied to what ASR sends: NLU gets the
transcript as heard and is the only stage that normalizes (decision
2026-10-28, evaluations/asr/README.md "Priority 3").

Steps: lowercase, drop Vosk's "[unk]" marker, split hyphens, strip
punctuation, number words -> digits, one written form per word (spellings),
drop fillers. Politeness words ("please", "thanks") are kept - a dropped
"please" counts as an error.

Synonyms are NOT mapped here ("knife" stays "knife"): that is the language
understanding stage's job (docs/COMMAND_SCHEMA_GUIDE.md, section 3A).
"""

from __future__ import annotations

import re
from typing import Optional

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# Different spellings of the same word -> one written form: "X-ray" (split
# at the hyphen) and the sound-alike spellings of the Vosk grammar
# (vocab.SPOKEN_FORMS), which still appear in Vosk's partial results.
SPELLINGS = {
    "c t": "ct",
    "m r i": "mri",
    "x ray": "xray",
    "hemo stat": "hemostat",
}
FILLERS = {"uh", "uhh", "um", "umm", "uhm", "er", "erm", "hmm", "hm", "mm"}


def strip_punctuation(text: str) -> str:
    text = text.lower().replace("[unk]", " ").replace("-", " ")
    return re.sub(r"[^\w\s]", "", text)


def _read_below_100(tokens: list[str], i: int) -> Optional[tuple[int, int]]:
    """"seven", "forty", "forty five" at tokens[i] -> (value, index after it)."""
    if i >= len(tokens):
        return None
    tok = tokens[i]
    if tok in TENS:
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt in NUMBER_WORDS and 1 <= NUMBER_WORDS[nxt] <= 9:
            return TENS[tok] + NUMBER_WORDS[nxt], i + 2
        return TENS[tok], i + 1
    if tok in NUMBER_WORDS:
        return NUMBER_WORDS[tok], i + 1
    return None


def _read_integer(tokens: list[str], i: int) -> Optional[tuple[int, int]]:
    """Adds hundreds: "one hundred (and) twenty", "a hundred"."""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
    if nxt == "hundred" and tokens[i] == "a":
        start: Optional[tuple[int, int]] = (1, i + 1)
    else:
        start = _read_below_100(tokens, i)
    if start is None:
        return None
    value, j = start
    if j < len(tokens) and tokens[j] == "hundred" and value >= 1:
        value, j = value * 100, j + 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest = _read_below_100(tokens, k)
        if rest is not None and rest[0] > 0:
            value, j = value + rest[0], rest[1]
    return value, j


def _read_number(tokens: list[str], i: int) -> Optional[tuple[str, int]]:
    """Adds decimals: "one point five" -> "1.5". Returns (digits, next index)."""
    integer = _read_integer(tokens, i)
    if integer is None:
        return None
    value, j = integer
    decimals = ""
    if j < len(tokens) and tokens[j] == "point":
        k = j + 1
        while k < len(tokens) and NUMBER_WORDS.get(tokens[k], 10) <= 9:
            decimals += str(NUMBER_WORDS[tokens[k]])
            k += 1
        if decimals:
            j = k
    return (f"{value}.{decimals}" if decimals else str(value)), j


def words_to_digits(tokens: list[str]) -> list[str]:
    """Spoken numbers -> digits: zero to the hundreds ("one hundred and
    twenty" -> 120) and decimals ("one point five" -> 1.5). Each number is
    read greedily, so "one two three" stays three numbers: 1 2 3."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        number = _read_number(tokens, i)
        if number is not None:
            out.append(number[0])
            i = number[1]
        else:
            out.append(tokens[i])
            i += 1
    return out


def canonicalize_spellings(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    for spelling, canonical in SPELLINGS.items():
        joined = re.sub(rf"\b{spelling}\b", canonical, joined)
    return joined.split()


def drop_fillers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FILLERS]


def normalize(text: str) -> str:
    """Applied to reference and hypothesis alike."""
    tokens = strip_punctuation(text).split()
    tokens = words_to_digits(tokens)
    tokens = canonicalize_spellings(tokens)
    tokens = drop_fillers(tokens)
    return " ".join(tokens)
