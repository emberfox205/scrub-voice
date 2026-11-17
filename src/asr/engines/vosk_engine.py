"""Vosk engine: vosk-model-small-en-us-0.15, free-form or with a restricted
grammar (see src/asr/CLAUDE.md "Engines to compare").

Output text: the words as heard, without the grammar's artifacts ("[unk]",
sound-alike spellings; `clean_words`) and without any normalization - NLU
does that.

Confidence: per-word `conf` is a real signal (it drops on words Vosk gets
wrong); the transcript confidence is the mean confidence of the output
words, as src/asr/CLAUDE.md specifies for Vosk - "[unk]" is not output, so
it is null if only "[unk]" was heard.

Setup choices, each found while benchmarking (details in docs/ASR_FINDINGS.md):
- One loaded model per process. Loading a second `vosk.Model` after freeing
  one segfaults the Kaldi library.
- Dither off. Kaldi's default (dither=1.0) adds random noise before feature
  extraction, so the same clip gave different words on repeated calls.
  `_disable_dither` adds `--dither=0` to the model's mfcc.conf once.
- One recognizer per engine, `Reset()` after every clip. Building a new
  recognizer per clip cost ~0.5 s, more than recognition itself.
- Word times keep counting across `Reset()` (the 2nd clip's first word
  starts at "length of clip 1 + ..."), so `_Recognizer` shifts them back to
  start at 0 for every utterance.
- Fed chunk by chunk (`accept_chunk`, for the spoken delimiter), Vosk is
  only given whole 0.2 s steps, the rest waits for the next chunk. Vosk
  works in 0.2 s steps internally and adapts to the speaker after each
  one; 0.5 s chunks moved those steps and changed the words of 3 of 99
  clips ("scalpel" -> "scan home"). With whole steps the result equals
  decoding the clip in one go on all 99 clips (2026-10-28), so the
  benchmark measures exactly what the stream does.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import requests
import vosk

from asr.audio import TARGET_SR
from asr.schema import ASRResult, WordInfo
from asr.vocab import LOG_HEADER_WORDS, LOG_TRIGGERS, SPOKEN_FORMS, ends_with_delimiter

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"  # src/asr/models (gitignored)
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"

vosk.SetLogLevel(-1)  # silence Vosk's own C++ logging


def _ensure_model_downloaded(model_dir: Path) -> None:
    if model_dir.exists():
        return
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(MODEL_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(model_dir.parent)


def _disable_dither(model_dir: Path) -> None:
    conf = model_dir / "conf" / "mfcc.conf"
    lines = conf.read_text(encoding="utf-8").splitlines()
    if "--dither=0" not in lines:
        conf.write_text("\n".join(lines + ["--dither=0"]) + "\n", encoding="utf-8")


_cached_model: Optional[vosk.Model] = None


def _get_model() -> vosk.Model:
    global _cached_model
    if _cached_model is None:
        model_dir = MODELS_DIR / MODEL_NAME
        _ensure_model_downloaded(model_dir)
        _disable_dither(model_dir)
        _cached_model = vosk.Model(model_path=str(model_dir))
    return _cached_model


STEP_BYTES = int(0.2 * TARGET_SR) * 2  # Vosk's internal step: 0.2 s of int16 samples


def _to_pcm(audio: np.ndarray) -> bytes:
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()


class _Recognizer:
    """A KaldiRecognizer fed one utterance at a time, in pieces of whole
    0.2 s steps (see module notes). Remembers how much audio it has been
    fed, so word times can be made relative to the start of each utterance."""

    def __init__(self, model: vosk.Model, grammar: Optional[list[str]] = None) -> None:
        if grammar is None:
            self.kaldi = vosk.KaldiRecognizer(model, TARGET_SR)
        else:
            self.kaldi = vosk.KaldiRecognizer(model, TARGET_SR, json.dumps(grammar))
        self.kaldi.SetWords(True)
        self.seconds_fed = 0.0       # all audio given to Vosk since the start
        self.utterance_start = 0.0   # seconds_fed when the current utterance began
        self.waiting = b""           # less than one step, kept for the next piece

    def _feed(self, pcm: bytes) -> None:
        # AcceptWaveform returns True when Vosk itself guesses the utterance
        # is over; ignored - the stream decides (DSP "end", delimiter, ...).
        self.kaldi.AcceptWaveform(pcm)
        self.seconds_fed += len(pcm) / 2 / TARGET_SR  # 2 bytes per int16 sample

    def accept(self, pcm: bytes) -> None:
        data = self.waiting + pcm
        whole = len(data) - len(data) % STEP_BYTES
        self.waiting = data[whole:]
        if whole:
            self._feed(data[:whole])

    def partial_words(self) -> list[str]:
        """The words heard so far in this utterance (best guess, may change).
        Words only, no times: SetPartialWords(True) would add times, but
        then "halt" showed up in the partial result 2 s late instead of
        0.2 s after it was said (2026-10-28) - useless for a fast path."""
        return json.loads(self.kaldi.PartialResult()).get("partial", "").split()

    def finish(self) -> list[dict]:
        """End the utterance; returns Vosk's word list ({"word", "start",
        "end", "conf"}, "[unk]" included), times from its start."""
        if self.waiting:
            self._feed(self.waiting)
            self.waiting = b""
        result = json.loads(self.kaldi.FinalResult())
        self.kaldi.Reset()
        offset, self.utterance_start = self.utterance_start, self.seconds_fed
        return [
            {**w, "start": round(w["start"] - offset, 3), "end": round(w["end"] - offset, 3)}
            for w in result.get("result", [])
        ]

    def decode(self, pcm: bytes) -> list[dict]:
        """A whole utterance at once."""
        self.accept(pcm)
        return self.finish()


def _starts_log_event(words: list[dict]) -> bool:
    """True if one of the first three real words is "log" or "record"."""
    real = [w["word"] for w in words if w["word"] != "[unk]"]
    return bool(LOG_TRIGGERS & set(real[:3]))


def splice_log_event(grammar_words: list[dict], free_words: list[dict]) -> list[dict]:
    """Command header from the grammar ("log event", "record milestone"),
    free text after it from the free-form recognizer. A free-form word is
    kept if its middle lies after the end of the header, so a word both
    recognizers heard (e.g. "event") is not doubled.

    Without a full header (trigger + "event"/"milestone") the grammar was
    probably forcing other speech into "log" ("Hello" -> "log" on the dev
    set), so the free-form result is used as it is."""
    header: list[dict] = []
    for w in grammar_words:
        if w["word"] in LOG_HEADER_WORDS:
            header.append(w)
        elif header or w["word"] != "[unk]":
            break  # the header is over (leading "[unk]", e.g. "um", is skipped)
    if len(header) < 2 or not LOG_TRIGGERS & {w["word"] for w in header}:
        return free_words
    header_end = header[-1]["end"]
    return header + [w for w in free_words if (w["start"] + w["end"]) / 2 > header_end]


# Sound-alike spellings the grammar uses for words missing from the model's
# dictionary, back to the word: ("hemo", "stat") -> "hemostat",
# ("x", "ray") -> "x-ray", ("c", "t") -> "ct".
JOINED_FORMS = {tuple(spoken.split()): term.lower() for term, spoken in SPOKEN_FORMS.items()}


def clean_words(vosk_words: list[dict]) -> list[dict]:
    """The words as heard, without the grammar's artifacts: sound-alike
    spellings are joined back into their word (NLU would not recognize
    "hemo stat" or "c t"), and "[unk]" (speech outside the grammar) is
    dropped - Vosk gives it conf 1.0, which would also inflate the
    transcript confidence. A joined word spans its parts and keeps their
    lowest confidence. Nothing else is changed: numbers stay words,
    synonyms stay as said."""
    out: list[dict] = []
    i = 0
    while i < len(vosk_words):
        for spoken, word in JOINED_FORMS.items():
            parts = vosk_words[i : i + len(spoken)]
            if tuple(w["word"] for w in parts) == spoken:
                out.append({"word": word, "start": parts[0]["start"], "end": parts[-1]["end"],
                            "conf": min(w["conf"] for w in parts)})
                i += len(spoken)
                break
        else:
            if vosk_words[i]["word"] != "[unk]":
                out.append(vosk_words[i])
            i += 1
    return out


def drop_delimiter(words: list[dict]) -> list[dict]:
    """Without a spoken delimiter at the end ("next slice now" -> "next
    slice"): it only tells ASR to send, it is not part of the command. Kept
    when no command word comes before it (vocab.ends_with_delimiter)."""
    if ends_with_delimiter([w["word"] for w in words]):
        return words[:-1]
    return words


class VoskEngine:
    name = "vosk"

    def __init__(self, grammar: Optional[Sequence[str]] = None, freeform_log: bool = False) -> None:
        """grammar: allowed phrases to restrict decoding to; "[unk]" is added
        so speech outside the grammar can be rejected instead of being forced
        into a command. Phrases containing a word the model doesn't know are
        dropped and listed in `self.dropped_phrases`. None = free-form.

        freeform_log (only with a grammar): log_event carries free text the
        grammar cannot write. When the grammar hears "log"/"record" at the
        start, a second, free-form recognizer (same loaded model) decodes the
        same audio and supplies the words after the command header."""
        self.model = _get_model()
        self.dropped_phrases: list[str] = []
        self.freeform: Optional[_Recognizer] = None
        self._partial: list[str] = []  # words heard so far in the current utterance
        if grammar:
            phrases = []
            for phrase in grammar:
                if all(self.model.vosk_model_find_word(w) >= 0 for w in phrase.split()):
                    phrases.append(phrase)
                else:
                    self.dropped_phrases.append(phrase)
            self.recognizer = _Recognizer(self.model, phrases + ["[unk]"])
            if freeform_log:
                # Note: the first "log event" of a session takes ~0.6 s longer
                # (one-time setup inside Vosk, laptop). Warming up with silence
                # or noise did not help; only real speech triggers it.
                self.freeform = _Recognizer(self.model)
        else:
            self.recognizer = _Recognizer(self.model)

    def accept_chunk(self, chunk: np.ndarray) -> list[str]:
        """Feed the next piece of the current utterance (16 kHz float32);
        returns the words heard so far, "[unk]" included - the stream looks
        for a spoken delimiter or a halt command in them."""
        self.recognizer.accept(_to_pcm(chunk))
        self._partial = self.recognizer.partial_words()
        return self._partial

    def heard_so_far(self) -> str:
        """The words of the last accept_chunk as text, cleaned like the final
        result (`clean_words`) - for the early copy of a halt command."""
        words = [{"word": w, "start": 0.0, "end": 0.0, "conf": 0.0} for w in self._partial]
        return " ".join(w["word"] for w in clean_words(words))

    def finish(self, audio: np.ndarray) -> ASRResult:
        """End the current utterance. `audio` = all of it (the chunks given
        to accept_chunk), for the free-form pass of log_event."""
        vosk_words = self.recognizer.finish()
        if self.freeform is not None and _starts_log_event(vosk_words):
            vosk_words = splice_log_event(vosk_words, self.freeform.decode(_to_pcm(audio)))

        words = [
            WordInfo(word=w["word"], start=w["start"], end=w["end"], confidence=w["conf"])
            for w in drop_delimiter(clean_words(vosk_words))
        ]
        confidence = float(np.mean([w.confidence for w in words])) if words else None

        return ASRResult(
            text=" ".join(w.word for w in words),  # empty if only "[unk]" was heard
            confidence=confidence,
            language="en",
            words=words,
            segment_start=0.0,  # filled in by the caller from the DSP timestamp
        )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRResult:
        """A whole utterance at once (files, benchmark) - same result as
        feeding it chunk by chunk."""
        if sr != TARGET_SR:
            raise ValueError(f"VoskEngine expects {TARGET_SR} Hz audio, got {sr} Hz")
        self.recognizer.accept(_to_pcm(audio))
        return self.finish(audio)
