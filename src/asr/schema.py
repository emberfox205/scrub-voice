"""The shared output shape every ASR engine returns.

This is the contract with the NLP stage - see src/asr/CLAUDE.md "Output schema".
Do not change field names or shape without telling Khanh.

`text` is the transcript as heard, without any normalization (numbers stay
words, "please" stays): NLU is the only stage that normalizes (decision
2026-10-28, evaluations/asr/README.md "Priority 3").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    confidence: Optional[float]


@dataclass
class ASRResult:
    text: str
    confidence: Optional[float]
    language: str
    words: list[WordInfo]
    segment_start: float
    # Why the utterance ended (set by asr.stream; null for a whole file):
    # "keyword" (spoken delimiter, vocab.DELIMITERS), "dsp_end", "gap",
    # "timeout" or "max_length". Telemetry only.
    end_reason: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "confidence": self.confidence,
                "language": self.language,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in self.words
                ],
                "segment_start": self.segment_start,
                "end_reason": self.end_reason,
            }
        )
