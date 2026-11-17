"""Named Vosk setups for benchmark.py, bench_worker.py, transcribe.py and
run_asr.py.

Vosk is the only engine (decision 2026-10-16; Whisper and Moonshine code was
removed on 2026-10-28 - their results stay in docs/ASR_FINDINGS.md). Next to
the production setup, the plain grammar and the unbiased free-form
recognizer stay, so the effect of the grammar and of the free-form pass for
log_event stays measurable.
"""

from __future__ import annotations

from typing import Callable

from asr.engines.vosk_engine import VoskEngine
from asr.vocab import vosk_grammar_phrases


def _vosk(use_grammar: bool, freeform_log: bool = False) -> Callable[[], VoskEngine]:
    def make() -> VoskEngine:
        return VoskEngine(
            grammar=vosk_grammar_phrases() if use_grammar else None,
            freeform_log=freeform_log,
        )

    return make


CONFIGS: dict[str, Callable[[], VoskEngine]] = {
    "vosk-freeform": _vosk(False),
    "vosk-grammar": _vosk(True),
    "vosk-grammar+freeform-log": _vosk(True, freeform_log=True),
}

# The setup the pipeline runs (decision 2026-10-16, docs/ASR_FINDINGS.md):
# best command accuracy, fastest, least memory, no audio tricks needed.
PRODUCTION_CONFIG = "vosk-grammar+freeform-log"
