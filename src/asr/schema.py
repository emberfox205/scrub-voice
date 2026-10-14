"""The shared output shape every ASR engine returns.

This is the contract with the NLP stage - see src/asr/CLAUDE.md "Output schema".
Do not change field names or shape without telling Khanh.
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
    normalized: str
    confidence: Optional[float]
    language: str
    words: list[WordInfo]
    segment_start: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "normalized": self.normalized,
                "confidence": self.confidence,
                "language": self.language,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in self.words
                ],
                "segment_start": self.segment_start,
            }
        )
