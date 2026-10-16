"""Named engine configurations for benchmark.py, bench_worker.py and
transcribe.py. Each factory imports its engine lazily, so a worker process
only loads the library it actually benchmarks.

Only the two shortlisted engines are kept (decision 2026-10-14, see
src/asr/CLAUDE.md): Vosk with a command grammar and faster-whisper tiny.en with a
prompt sentence, each next to its unbiased baseline so the effect of the
vocabulary help stays measurable, plus one variant of each (shorter prompt;
free-form recognizer for log_event free text). Moonshine, Whisper base.en, beam 5 and
hot words were dropped; their results are in docs/ASR_FINDINGS.md.
"""

from __future__ import annotations

from typing import Callable


def _faster_whisper(model_size: str, beam_size: int, prompt: str = "") -> Callable[[], object]:
    """prompt: "" (none), "full" (WHISPER_PROMPT) or "short" (WHISPER_PROMPT_SHORT)."""
    def make():
        from asr.engines.faster_whisper_engine import FasterWhisperEngine
        from asr.vocab import WHISPER_PROMPT, WHISPER_PROMPT_SHORT

        prompts = {"": None, "full": WHISPER_PROMPT, "short": WHISPER_PROMPT_SHORT}
        return FasterWhisperEngine(
            model_size=model_size,
            beam_size=beam_size,
            initial_prompt=prompts[prompt],
        )

    return make


def _vosk(use_grammar: bool, freeform_log: bool = False) -> Callable[[], object]:
    def make():
        from asr.engines.vosk_engine import VoskEngine
        from asr.vocab import vosk_grammar_phrases

        return VoskEngine(
            grammar=vosk_grammar_phrases() if use_grammar else None,
            freeform_log=freeform_log,
        )

    return make


BASELINES: dict[str, Callable[[], object]] = {
    "faster-whisper-tiny.en-beam1": _faster_whisper("tiny.en", 1),
    "vosk-freeform": _vosk(False),
}

BIASED: dict[str, Callable[[], object]] = {
    "faster-whisper-tiny.en-beam1+prompt": _faster_whisper("tiny.en", 1, prompt="full"),
    # Phase A variants, to be measured in Phase B: a shorter prompt (the
    # full prompt cost 0.35 -> 0.87 s per clip) and a free-form recognizer
    # for the free text of log_event.
    "faster-whisper-tiny.en-beam1+shortprompt": _faster_whisper("tiny.en", 1, prompt="short"),
    "vosk-grammar": _vosk(True),
    "vosk-grammar+freeform-log": _vosk(True, freeform_log=True),
}

CONFIGS: dict[str, Callable[[], object]] = {**BASELINES, **BIASED}

# The setup the pipeline runs (decision 2026-10-16, docs/ASR_FINDINGS.md):
# best command accuracy, fastest, least memory, no audio tricks needed.
PRODUCTION_CONFIG = "vosk-grammar+freeform-log"
