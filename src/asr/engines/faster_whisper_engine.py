"""faster-whisper engine: tiny.en, int8, CPU, beam 1, optional prompt
sentence (see src/asr/CLAUDE.md "Engines").

Decoding: language fixed to "en", temperature=0 (no random-sampling
fallback - the default retries with temperature > 0 when a decode looks
bad, which made beam1 output differ between runs), and no conditioning on
previous text (each clip is an independent command).

Hot words are deliberately not supported: with temperature=0 they caused
repetition loops and "zoom in" on pure noise (docs/ASR_FINDINGS.md, stage 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from asr.normalize import normalize
from asr.schema import ASRResult, WordInfo

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"  # src/asr/models (gitignored)


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "tiny.en",
        compute_type: str = "int8",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRResult:
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=self.initial_prompt,
        )
        segments = list(segments)

        text = "".join(seg.text for seg in segments).strip()
        words = [
            WordInfo(word=w.word.strip(), start=w.start, end=w.end, confidence=w.probability)
            for seg in segments
            for w in seg.words
        ]

        # faster-whisper gives avg_logprob (log of the average probability)
        # per segment. exp() turns it back into a 0-1 confidence score.
        # Empty output has no logprob to score; exp(0) would claim 1.0.
        if text:
            avg_logprob = sum(seg.avg_logprob for seg in segments) / len(segments)
            confidence = float(np.exp(avg_logprob))
        else:
            confidence = None

        return ASRResult(
            text=text,
            normalized=normalize(text, corrections=True, drop_politeness=True),
            confidence=confidence,
            language=info.language,
            words=words,
            segment_start=0.0,  # filled in by the caller from the DSP timestamp
        )
