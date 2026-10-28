"""Text normalization for scoring only (benchmark), used identically on
reference and hypothesis. It is not applied to what ASR sends: NLU gets the
transcript as heard and is the only stage that normalizes (decision
2026-10-28, evaluations/asr/README.md "Priority 3").

Steps: lowercase, drop Vosk's "[unk]" marker, split hyphens, strip
punctuation (decimal points between digits are kept), split units off
numbers ("2x" -> "2 x"), number words -> digits, one written form per word
(spellings), drop fillers. Politeness words ("please", "thanks") are kept -
a dropped "please" counts as an error.

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
# Different spellings of the same word -> one written form, so "X-ray",
# "x ray" and "xray" all score alike. Dotted forms like "C.T." already
# collapse to "ct" when punctuation is stripped. "hemo stat" is how the Vosk
# grammar spells hemostat (the word is not in the model's dictionary).
SPELLINGS = {
    "c t": "ct", "see tea": "ct", "see t": "ct",
    "m r i": "mri",
    "x ray": "xray",
    "hemo stat": "hemostat",
    "stand by": "standby",
    "pick ups": "pickups",
    "cm": "centimeters", "centimetres": "centimeters", "centimetre": "centimeter",
}
# Number followed directly by a unit, as Whisper writes it: "2x", "1.5x", "5cm".
NUMBER_UNIT = re.compile(r"\b(\d+(?:\.\d+)?)(x|cm)\b")
FILLERS = {"uh", "uhh", "um", "umm", "uhm", "er", "erm", "hmm", "hm", "mm"}

# Post-correction map: (regex on normalized text, replacement). Benchmark
# only (the "+corr" rows): it shows what fixing these mishearings would gain,
# but nothing sent to NLU is corrected. The errors are Whisper's; the Vosk
# grammar cannot produce them outside log_event free text (2026-10-28: the
# "+corr" rows of the production config equal the plain ones). Built from
# errors on the dev set only (Phase B, 2026-10-16; which engine made each
# error: docs/ASR_FINDINGS.md). Only mishearings of command words, and only
# phrases that would almost never be meant literally in an operating room,
# so the map can't create false commands. Left out on purpose:
# "close skin" (a real surgical phrase - Whisper's "close scan"), "scope"
# (a real instrument - Whisper's "scalpel"), "soon" inside a sentence, and
# free-text words of log_event ("troll car" for trocar).
# A whole utterance is matched with ^...$.
CORRECTIONS: list[tuple[str, str]] = [
    # zoom_display
    (r"^soon$", "zoom"),
    (r"\b(soon|soom) (in|out)\b", r"zoom \2"),
    (r"\bzoom (in|out) to x\b", r"zoom \1 2 x"),
    (r"\b(\d+(?:\.\d+)?) eggs\b", r"\1 x"),
    # emergency
    (r"\bbig off\b", "back off"),
    (r"\b(hot|haunt) robot\b", "halt robot"),
    (r"\browboat\b", "robot"),
    # navigate_slice
    (r"\b(begg|big) (\d+) slices?\b", r"back \2 slices"),
    (r"\b(next|previous) slide\b", r"\1 slice"),
    # tools
    (r"\b(scampel|scamper)\b", "scalpel"),
    (r"\b(hemmostat|him most it)\b", "hemostat"),
    (r"\bmitzen bow\b", "metzenbaum"),
    # scans
    (r"\b(ct|mri|xray) skin\b", r"\1 scan"),
    (r"\bcity (scan|skin)\b", "ct scan"),
    (r"\bem or i\b", "mri"),
    (r"^this means (ct|mri|xray|scan)$", r"dismiss \1"),
    # camera
    (r"\bcenter (review|of you)\b", "center view"),
    (r"\bcameron (left|right|up|down|forward|backward|home)\b", r"camera \1"),
    # log_event
    (r"\b(lock|knock|nor) event\b", "log event"),
]


def strip_punctuation(text: str) -> str:
    text = text.lower().replace("[unk]", " ").replace("-", " ")
    text = text.replace("×", " x ").replace("°", " degrees ")
    text = re.sub(r"[^\w\s.]", "", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", "", text)  # keep only dots inside numbers ("1.5")
    return NUMBER_UNIT.sub(r"\1 \2", text)


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
    """Adds hundreds: "one hundred (and) twenty", "a hundred", "2 hundred"."""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
    if nxt == "hundred" and (tokens[i] == "a" or tokens[i].isdigit()):
        start: Optional[tuple[int, int]] = (1 if tokens[i] == "a" else int(tokens[i]), i + 1)
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


def _tidy_decimal(token: str) -> str:
    """"2.0" -> "2", "2.50" -> "2.5" (Whisper sometimes writes trailing zeros)."""
    if re.fullmatch(r"\d+\.\d+", token):
        return token.rstrip("0").rstrip(".")
    return token


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
            out.append(_tidy_decimal(tokens[i]))
            i += 1
    return out


def canonicalize_spellings(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    for spelling, canonical in SPELLINGS.items():
        joined = re.sub(rf"\b{spelling}\b", canonical, joined)
    return joined.split()


def drop_fillers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FILLERS]


def apply_corrections(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    for pattern, replacement in CORRECTIONS:
        joined = re.sub(pattern, replacement, joined)
    return joined.split()


def normalize(text: str, *, corrections: bool = False) -> str:
    """Applied to reference and hypothesis alike. corrections=True gives the
    benchmark's "+corr" rows."""
    tokens = strip_punctuation(text).split()
    tokens = words_to_digits(tokens)
    tokens = canonicalize_spellings(tokens)
    tokens = drop_fillers(tokens)
    if corrections:
        tokens = apply_corrections(tokens)
    return " ".join(tokens)
