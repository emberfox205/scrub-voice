"""Command vocabulary and the term metrics built on it.

!! OUTDATED VOCABULARY !! COMMAND_VOCAB, SPOKEN_FORMS and WHISPER_PROMPT
below still describe the OLD image-viewer commands (pan, window, axial, ...)
that the first 46-clip test set (not in this repo) was recorded with. The real
command set is now defined by schemas/command_schema.json - rewriting this
file from it is Phase A in src/asr/CLAUDE.md. The functions (find_terms,
term_counts, command_group, vosk_grammar_phrases, check_leakage) are
vocabulary-independent and can stay.

Terms are matched on normalized text, as whole words, longest phrase first
and without overlap, and counted per occurrence: "lung window" counts once
as "lung window", not also as "window".
"""

from __future__ import annotations

from collections import Counter

from asr.normalize import normalize

COMMAND_VOCAB = [
    "zoom in", "zoom out",
    "pan left", "pan right", "pan up", "pan down",
    "rotate", "flip", "reset",
    "contrast", "brightness", "window", "level",
    "lung window", "bone window", "soft tissue",
    "axial", "coronal", "sagittal",
    "next slice", "previous slice", "scroll", "series",
    "DICOM", "PACS", "CT", "MRI",
    "measure", "annotate", "undo",
]

# How each acronym is said aloud, for grammars that need real words. Each
# spoken form must normalize back to the canonical term (asr/normalize.py).
# PACS has no unambiguous spoken form ("packs" is an everyday word), so it is
# left out of the grammar rather than risk false commands.
SPOKEN_FORMS = {"CT": "c t", "MRI": "m r i", "DICOM": "die com", "PACS": None}

GRAMMAR_EXTRA_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "please", "by", "times",
]

# One short natural sentence containing the key terms, for Whisper's
# initial_prompt (it nudges spelling and word choice toward these terms).
WHISPER_PROMPT = (
    "In the DICOM viewer we zoom in, zoom out, pan left, right, up and down, rotate, flip, "
    "reset, adjust contrast, brightness, window and level, use the lung window, bone window "
    "or soft tissue, switch between axial, coronal and sagittal, go to the next slice or "
    "previous slice, scroll the series, measure, annotate and undo on CT, MRI and PACS images."
)

_TERMS = sorted({tuple(normalize(t).split()) for t in COMMAND_VOCAB}, key=len, reverse=True)


def vosk_grammar_phrases() -> list[str]:
    """Allowed phrases for Vosk's restricted grammar, built only from this
    file (never from test-set sentences). "[unk]" is added by the engine."""
    phrases = []
    for term in COMMAND_VOCAB:
        spoken = SPOKEN_FORMS.get(term, term.lower())
        if spoken:
            phrases.append(spoken)
    return phrases + GRAMMAR_EXTRA_WORDS


def check_leakage(texts: list[str], references: list[str], what: str) -> None:
    """Fail if a test sentence appears (as a whole phrase) in any biasing
    text: grammar phrases or the prompt sentence. References that are
    themselves a single vocab term ("Zoom in", "Undo") are allowed - they are
    in the vocabulary because they are commands, not because of the test set."""
    vocab = {normalize(t) for t in COMMAND_VOCAB}
    padded = [f" {normalize(t)} " for t in texts]
    leaked = sorted({
        r for r in map(normalize, references)
        if r and r not in vocab and any(f" {r} " in t for t in padded)
    })
    assert not leaked, f"Test-set sentences found verbatim in the {what}: {leaked}"


def check_all_biasing_leakage(references: list[str]) -> None:
    # The grammar is checked phrase by phrase: joining its entries would glue
    # neighbours like "window" + "level" into the test sentence "window level".
    check_leakage(vosk_grammar_phrases(), references, "Vosk grammar")
    check_leakage([WHISPER_PROMPT], references, "Whisper prompt")


def find_terms(normalized_text: str) -> list[str]:
    """Vocab terms in order of appearance, each occurrence listed once."""
    tokens = normalized_text.split()
    found: list[str] = []
    i = 0
    while i < len(tokens):
        for term in _TERMS:
            if tuple(tokens[i : i + len(term)]) == term:
                found.append(" ".join(term))
                i += len(term)
                break
        else:
            i += 1
    return found


def term_counts(ref_normalized: str, hyp_normalized: str) -> tuple[int, int, int]:
    """(terms said, terms output, terms output that were also said)."""
    ref = Counter(find_terms(ref_normalized))
    hyp = Counter(find_terms(hyp_normalized))
    matched = sum((ref & hyp).values())
    return sum(ref.values()), sum(hyp.values()), matched


def command_group(ref_normalized: str) -> str:
    """Cluster key for bootstrapping: the first vocab term in the reference,
    so "Zoom in" and "Um, zoom in a little, please" land in one group.
    References without any vocab term form a group of their own."""
    terms = find_terms(ref_normalized)
    return terms[0] if terms else ref_normalized
