"""Command vocabulary (from schemas/command_schema.json and
docs/COMMAND_SCHEMA_GUIDE.md) and the term metrics built on it.

Terms are the words that carry an intent or a slot value. They are matched
on normalized text, as whole words, longest phrase first and without
overlap, and counted per occurrence: "back off" counts once as "back off",
not also as "back". Every number is a term too (slot values: zoom factor,
slice, centimeters, degrees), so a wrong or invented number counts as a
wrong or invented term.

Nothing in this file may be copied from the test phrase lists in
data/asr/prompts/ (check_leakage, src/asr/test/test_vocab.py).
"""

from __future__ import annotations

import re
from collections import Counter

from asr.normalize import NUMBER_WORDS, TENS, normalize

# The words the ASR must get right, per intent (src/asr/CLAUDE.md "Commands
# to recognise"). Synonyms are listed because the speaker may use them; the
# language understanding stage maps them to the schema tokens.
INTENT_TERMS: dict[str, list[str]] = {
    "pick_tool": [
        "hand", "pass", "give me",
        "scalpel", "scissors", "hemostat", "forceps",
        "knife", "ten blade", "blade", "iris", "metz", "metzenbaum",
        "clamp", "snap", "tweezers", "pickups",
    ],
    "park_arm": ["park", "standby", "retract", "arm", "robot"],
    "clear_emergency": ["emergency", "clear", "area", "back off"],
    "halt_emergency": ["emergency", "stop", "halt", "freeze", "robot"],
    "open_scan": ["open", "show", "display", "scan", "CT", "MRI", "X-ray"],
    "close_scan": ["close", "dismiss", "scan", "viewer", "CT", "MRI", "X-ray"],
    # "in"/"out" only count together with "zoom": alone they are everyday words.
    "zoom_display": ["zoom", "zoom in", "zoom out", "magnify"],
    "navigate_slice": ["next", "previous", "back", "go to", "slice", "slices"],
    # Plus free text (event name, comment), which is not a term.
    "log_event": ["log", "record", "event", "milestone", "note", "comment"],
    "move_camera": [
        "camera", "move", "left", "right", "up", "down", "forward", "backward",
        "centimeter", "centimeters",
    ],
    "tilt_camera": ["tilt", "up", "down", "left", "right", "degree", "degrees"],
    "reset_camera": ["reset", "center", "view", "home", "camera"],
}

COMMAND_VOCAB = sorted({term for terms in INTENT_TERMS.values() for term in terms})

# Words that start a log_event command; free text follows them. The Vosk
# engine uses these to switch to free-form recognition for the rest.
LOG_TRIGGERS = {"log", "record"}
LOG_HEADER_WORDS = LOG_TRIGGERS | {"event", "milestone"}

# How a term is said aloud, for the Vosk grammar: a grammar can only contain
# words from the model's dictionary, and "hemostat" and "xray" are not in
# vosk-model-small-en-us-0.15. Each spoken form must normalize back to the
# term (asr/normalize.py SPELLINGS).
SPOKEN_FORMS = {"CT": "c t", "MRI": "m r i", "X-ray": "x ray", "hemostat": "hemo stat"}

# Small words the guide's commands are built with, plus the words needed to
# say a zoom factor ("two x", "two times") or a number ("one hundred and
# twenty", "one point five").
GRAMMAR_FUNCTION_WORDS = ["the", "a", "me", "and", "by", "x", "times", "please"]
GRAMMAR_NUMBER_WORDS = list(NUMBER_WORDS) + list(TENS) + ["hundred", "point"]
# Softeners people add to commands. Without them the grammar forced
# "tilt the camera up a little" into "... up eleven" - an invented angle
# (dev set, Phase B).
GRAMMAR_MODIFIER_WORDS = ["little", "bit", "okay", "could", "you"]

# One natural sentence with the key terms, for Whisper's initial_prompt (it
# nudges spelling and word choice toward these terms). Written from the
# schema guide, avoiding the exact test sentences. Numbers are written as
# digits so Whisper writes digits too.
WHISPER_PROMPT = (
    "During surgery the team asks the robot to pass the scalpel, knife, ten blade, "
    "Metzenbaum or iris scissors, hemostat, clamp, snap, forceps, tweezers or pickups, "
    "to park, retract or standby the arm, to halt, freeze, stop or back off in an emergency, "
    "to open, show, display, close or dismiss the CT, MRI or X-ray scan in the viewer, "
    "to zoom in or out by 2.5x or magnify, to step to the next, previous or a numbered slice, "
    "to log an event or record a milestone with a note or comment, "
    "and to move or tilt the camera left, right, up, down, forward or backward "
    "by 7 cm or 20 degrees, then reset or center it at home."
)

# A shorter variant to measure the latency cost of the prompt (Phase B):
# only the terms Whisper is most likely to misspell or mishear.
WHISPER_PROMPT_SHORT = (
    "The robot passes the scalpel, Metzenbaum scissors, hemostat, forceps or pickups, "
    "opens the CT, MRI or X-ray scan, zooms, magnifies, logs a milestone and tilts the camera."
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TERMS = sorted({tuple(normalize(t).split()) for t in COMMAND_VOCAB}, key=len, reverse=True)


def vosk_grammar_phrases() -> list[str]:
    """Allowed phrases for Vosk's restricted grammar, built only from this
    file (never from test-set sentences). "[unk]" is added by the engine.

    A Vosk grammar accepts any sequence of its entries, but the entries
    also teach it which words follow each other. So multi-word terms
    ("go to", "back off", "x ray") are entries of their own, and a word that
    only occurs inside them is not added alone: with "to" as a single word
    next to "two", the grammar heard "go two slice 45" (dev set, Phase B)."""
    phrases: set[str] = set()
    single: set[str] = set(GRAMMAR_FUNCTION_WORDS + GRAMMAR_NUMBER_WORDS + GRAMMAR_MODIFIER_WORDS)
    for term in COMMAND_VOCAB:
        spoken = SPOKEN_FORMS.get(term, term.lower())
        if " " in spoken:
            phrases.add(spoken)
        else:
            single.add(spoken)
    return sorted(phrases) + sorted(single)


def check_leakage(texts: list[str], references: list[str], what: str) -> None:
    """Fail if a test sentence appears (as a whole phrase) in any biasing
    text: grammar phrases or the prompt sentence. References that are
    themselves a single vocab term ("Freeze", "Back off") are allowed - they
    are in the vocabulary because they are commands, not because of the test
    set."""
    vocab = {normalize(t) for t in COMMAND_VOCAB}
    padded = [f" {normalize(t)} " for t in texts]
    leaked = sorted({
        r for r in map(normalize, references)
        if r and r not in vocab and any(f" {r} " in t for t in padded)
    })
    assert not leaked, f"Test-set sentences found verbatim in the {what}: {leaked}"


def check_all_biasing_leakage(references: list[str]) -> None:
    # The grammar is checked entry by entry: joining its entries would glue
    # neighbours into phrases nobody wrote.
    check_leakage(vosk_grammar_phrases(), references, "Vosk grammar")
    check_leakage([WHISPER_PROMPT], references, "Whisper prompt")
    check_leakage([WHISPER_PROMPT_SHORT], references, "short Whisper prompt")


def find_terms(normalized_text: str) -> list[str]:
    """Vocab terms and numbers in order of appearance, each occurrence
    listed once."""
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
            if _NUMBER.fullmatch(tokens[i]):
                found.append(tokens[i])
            i += 1
    return found


def term_counts(ref_normalized: str, hyp_normalized: str) -> tuple[int, int, int]:
    """(terms said, terms output, terms output that were also said)."""
    ref = Counter(find_terms(ref_normalized))
    hyp = Counter(find_terms(hyp_normalized))
    matched = sum((ref & hyp).values())
    return sum(ref.values()), sum(hyp.values()), matched


def command_group(ref_normalized: str) -> str:
    """Fallback cluster key for bootstrapping when a clip has no intent in
    its manifest `notes` column: the first vocab term in the reference.
    References without any vocab term form a group of their own."""
    terms = find_terms(ref_normalized)
    return terms[0] if terms else ref_normalized


def cluster_key(notes: str, ref_normalized: str) -> str:
    """Cluster for the bootstrap: the intent from the manifest `notes`
    column (the phrase-list section, e.g. "pick_tool"), so "Hand scalpel"
    and "Pass the hemostat" are resampled together."""
    return notes or command_group(ref_normalized)
