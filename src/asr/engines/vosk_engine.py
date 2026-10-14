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


class VoskEngine:
    name = "vosk"

    def __init__(self, grammar: Optional[Sequence[str]] = None) -> None:
        """grammar: allowed phrases to restrict decoding to; "[unk]" is added
        so speech outside the grammar can be rejected instead of being forced
        into a command. Phrases containing a word the model doesn't know are
        dropped and listed in `self.dropped_phrases`. None = free-form."""
        self.model = _get_model()
        self.dropped_phrases: list[str] = []
        if grammar:
            phrases = []
            for phrase in grammar:
                if all(self.model.vosk_model_find_word(w) >= 0 for w in phrase.split()):
                    phrases.append(phrase)
                else:
                    self.dropped_phrases.append(phrase)
            self.recognizer = vosk.KaldiRecognizer(self.model, TARGET_SR, json.dumps(phrases + ["[unk]"]))
        else:
            self.recognizer = vosk.KaldiRecognizer(self.model, TARGET_SR)
        self.recognizer.SetWords(True)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRResult:
        if sr != TARGET_SR:
            raise ValueError(f"VoskEngine expects {TARGET_SR} Hz audio, got {sr} Hz")
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        self.recognizer.AcceptWaveform(pcm)
        result = json.loads(self.recognizer.FinalResult())
        self.recognizer.Reset()

        text = result.get("text", "")
        words = [
            WordInfo(word=w["word"], start=w["start"], end=w["end"], confidence=w["conf"])
            for w in result.get("result", [])
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
