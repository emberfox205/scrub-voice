"""Vosk engine: vosk-model-small-en-us-0.15, free-form or with a restricted
grammar (see src/asr/CLAUDE.md "Engines to compare").

Confidence: per-word `conf` is a real signal (it drops on words Vosk gets
wrong); the transcript confidence is the mean word confidence, as src/asr/CLAUDE.md
specifies for Vosk.

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
from asr.normalize import normalize
from asr.schema import ASRResult, WordInfo
from asr.vocab import LOG_HEADER_WORDS, LOG_TRIGGERS

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


class _Recognizer:
    """A KaldiRecognizer that remembers how much audio it has been fed, so
    word times can be made relative to the start of each utterance."""

    def __init__(self, model: vosk.Model, grammar: Optional[list[str]] = None) -> None:
        if grammar is None:
            self.kaldi = vosk.KaldiRecognizer(model, TARGET_SR)
        else:
            self.kaldi = vosk.KaldiRecognizer(model, TARGET_SR, json.dumps(grammar))
        self.kaldi.SetWords(True)
        self.seconds_fed = 0.0

    def decode(self, pcm: bytes) -> list[dict]:
        """Run over a whole utterance; returns Vosk's word list
        ({"word", "start", "end", "conf"}, "[unk]" included)."""
        self.kaldi.AcceptWaveform(pcm)
        result = json.loads(self.kaldi.FinalResult())
        self.kaldi.Reset()
        offset = self.seconds_fed
        self.seconds_fed += len(pcm) / 2 / TARGET_SR  # 2 bytes per int16 sample
        return [
            {**w, "start": round(w["start"] - offset, 3), "end": round(w["end"] - offset, 3)}
            for w in result.get("result", [])
        ]


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
        if grammar:
            phrases = []
            for phrase in grammar:
                if all(self.model.vosk_model_find_word(w) >= 0 for w in phrase.split()):
                    phrases.append(phrase)
                else:
                    self.dropped_phrases.append(phrase)
            self.recognizer = _Recognizer(self.model, phrases + ["[unk]"])
            if freeform_log:
                self.freeform = _Recognizer(self.model)
        else:
            self.recognizer = _Recognizer(self.model)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRResult:
        if sr != TARGET_SR:
            raise ValueError(f"VoskEngine expects {TARGET_SR} Hz audio, got {sr} Hz")
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        vosk_words = self.recognizer.decode(pcm)
        if self.freeform is not None and _starts_log_event(vosk_words):
            vosk_words = splice_log_event(vosk_words, self.freeform.decode(pcm))

        # Vosk's own "text" is these words joined by spaces.
        text = " ".join(w["word"] for w in vosk_words)
        words = [
            WordInfo(word=w["word"], start=w["start"], end=w["end"], confidence=w["conf"])
            for w in vosk_words
        ]
        confidence = float(np.mean([w.confidence for w in words])) if words else None

        return ASRResult(
            text=text,
            normalized=normalize(text, corrections=True, drop_politeness=True),
            confidence=confidence,
            language="en",
            words=words,
            segment_start=0.0,  # filled in by the caller from the DSP timestamp
        )
