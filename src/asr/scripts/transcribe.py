"""Audio file(s) -> one ASR JSON line per file on stdout. The model is
loaded once for all files.

Example:
    python src/asr/scripts/transcribe.py data/asr/sample_01.wav --config vosk-grammar
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on sys.path, like the other modules

from asr.audio import load_audio
from asr.bench_configs import CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio files to ASR JSON")
    parser.add_argument("wav_paths", type=Path, nargs="+")
    parser.add_argument("--config", default="vosk-grammar", choices=list(CONFIGS))
    args = parser.parse_args()

    engine = CONFIGS[args.config]()
    for wav_path in args.wav_paths:
        print(engine.transcribe(load_audio(wav_path)).to_json())


if __name__ == "__main__":
    main()
