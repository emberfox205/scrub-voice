"""NLU module for medical intent classification and slot extraction.

Uses dynamic getattr to avoid eagerly loading heavy dependencies (torch, transformers,
SetFit) when only lightweight utilities (e.g., LexicalNormalizer, SlotExtractor) are needed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .intent_classifier import IntentClassifier
    from .normalizer import LexicalNormalizer
    from .pipeline import VoiceToIntentPipeline
    from .slot_extractor import SlotExtractor


def __getattr__(name: str):
    if name == "LexicalNormalizer":
        from .normalizer import LexicalNormalizer
        return LexicalNormalizer
    if name == "SlotExtractor":
        from .slot_extractor import SlotExtractor
        return SlotExtractor
    if name == "IntentClassifier":
        from .intent_classifier import IntentClassifier
        return IntentClassifier
    if name == "VoiceToIntentPipeline":
        from .pipeline import VoiceToIntentPipeline
        return VoiceToIntentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IntentClassifier",
    "LexicalNormalizer",
    "SlotExtractor",
    "VoiceToIntentPipeline",
]
