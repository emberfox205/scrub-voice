"""Text normalization, used identically on reference and hypothesis for
scoring. The raw `text` an engine produced is never changed; this only
builds the cleaned version.

Steps: lowercase, drop Vosk's "[unk]" marker, split hyphens, strip
punctuation, number words -> digits, canonical acronyms, drop fillers.
Politeness words ("please", "thanks") are kept for scoring - a dropped
"please" counts as an error - but can be dropped for the JSON `normalized`
field (src/asr/CLAUDE.md: "Axial, please." -> "axial").
"""

from __future__ import annotations

import re

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
# Spoken / spelled-out forms -> one canonical token. Dotted forms like
# "C.T." already collapse to "ct" when punctuation is stripped.
ACRONYMS = {
    "c t": "ct", "see tea": "ct", "see t": "ct",
    "m r i": "mri",
    "d i c o m": "dicom", "die com": "dicom", "dye com": "dicom",
    "p a c s": "pacs",
}
FILLERS = {"uh", "uhh", "um", "umm", "uhm", "er", "erm", "hmm", "hm", "mm"}
POLITENESS = {"please", "thanks"}

# Post-correction map: (regex on normalized text, replacement). Only errors
# actually observed in the stage-1 diagnosis, and only phrases that would
# almost never be meant literally in an OR (so the map can't create false
# commands). Which engine produced each error: docs/ASR_FINDINGS.md, stage 4.
CORRECTIONS = [
    (r"\bpenn? (left|right|up|down)\b", r"pan \1"),
    (r"\bsoon (in|out)\b", r"zoom \1"),
    (r"\bboom in\b", "zoom in"),
    (r"\blong window\b", "lung window"),
    (r"\bphone window\b", "bone window"),
    (r"\bcoron(a|o) view\b", "coronal view"),
    (r"\b(paronal|plaronal|pluronal)\b", "coronal"),
    (r"\b(exhale|axiom) view\b", "axial view"),
    (r"\b(saggitel|seggitel|segital|sekito|succito|satchito)\b", "sagittal"),
    (r"\bsaggy tell\b", "sagittal"),
    (r"\b(set your tail|such a little|such eternal|said she to) view\b", "sagittal view"),
    (r"\band (notate|tate)\b", "annotate"),
]


def strip_punctuation(text: str) -> str:
    text = text.lower().replace("[unk]", " ").replace("-", " ")
    return re.sub(r"[^\w\s]", "", text)


def words_to_digits(tokens: list[str]) -> list[str]:
    """zero..ninety-nine and "N hundred" -> digits; everything else unchanged."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in TENS:
            value = TENS[tok]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt in NUMBER_WORDS and NUMBER_WORDS[nxt] < 10:
                value += NUMBER_WORDS[nxt]
                i += 1
            out.append(str(value))
        elif tok in NUMBER_WORDS:
            out.append(str(NUMBER_WORDS[tok]))
        elif tok == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        else:
            out.append(tok)
        i += 1
    return out


def canonicalize_acronyms(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    for spoken, canonical in ACRONYMS.items():
        joined = re.sub(rf"\b{spoken}\b", canonical, joined)
    return joined.split()


def drop_fillers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FILLERS]


def drop_politeness_words(tokens: list[str]) -> list[str]:
    joined = re.sub(r"\bthank you\b", " ", " ".join(tokens))
    return [t for t in joined.split() if t not in POLITENESS]


def apply_corrections(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    for pattern, replacement in CORRECTIONS:
        joined = re.sub(pattern, replacement, joined)
    return joined.split()


def normalize(text: str, *, corrections: bool = False, drop_politeness: bool = False) -> str:
    """Scoring uses the defaults (applied to reference and hypothesis alike).
    The JSON `normalized` field uses corrections=True, drop_politeness=True."""
    tokens = strip_punctuation(text).split()
    tokens = words_to_digits(tokens)
    tokens = canonicalize_acronyms(tokens)
    tokens = drop_fillers(tokens)
    if corrections:
        tokens = apply_corrections(tokens)
    if drop_politeness:
        tokens = drop_politeness_words(tokens)
    return " ".join(tokens)
