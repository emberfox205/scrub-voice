"""Named engine configurations for benchmark.py, bench_worker.py and
transcribe.py. Each factory imports its engine lazily, so a worker process
only loads the library it actually benchmarks.

Only the two shortlisted engines are kept (decision 2026-10-14, see
src/asr/CLAUDE.md): Vosk with a command grammar and faster-whisper tiny.en with a
prompt sentence, each next to its unbiased baseline so the effect of the
vocabulary help stays measurable. Moonshine, Whisper base.en, beam 5 and
hot words were dropped; their results are in docs/ASR_FINDINGS.md.
"""

from __future__ import annotations

from typing import Callable


def _faster_whisper(model_size: str, beam_size: int, prompt: bool = False) -> Callable[[], object]:
    def make():
        from asr.engines.faster_whisper_engine import FasterWhisperEngine
        from asr.vocab import WHISPER_PROMPT

        return FasterWhisperEngine(
            model_size=model_size,
            beam_size=beam_size,
            initial_prompt=WHISPER_PROMPT if prompt else None,
        )

    return make


def _vosk(use_grammar: bool) -> Callable[[], object]:
    def make():
        from asr.engines.vosk_engine import VoskEngine
        from asr.vocab import vosk_grammar_phrases

        return VoskEngine(grammar=vosk_grammar_phrases() if use_grammar else None)

    return make


BASELINES: dict[str, Callable[[], object]] = {
    "faster-whisper-tiny.en-beam1": _faster_whisper("tiny.en", 1),
    "vosk-freeform": _vosk(False),
}

BIASED: dict[str, Callable[[], object]] = {
    "faster-whisper-tiny.en-beam1+prompt": _faster_whisper("tiny.en", 1, prompt=True),
    "vosk-grammar": _vosk(True),
}

CONFIGS: dict[str, Callable[[], object]] = {**BASELINES, **BIASED}
