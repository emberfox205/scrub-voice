"""One audio file -> ASR JSON on stdout.

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
    parser = argparse.ArgumentParser(description="Transcribe one audio file to ASR JSON")
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--config", default="faster-whisper-tiny.en-beam5", choices=list(CONFIGS))
    args = parser.parse_args()

    audio = load_audio(args.wav_path)
    engine = CONFIGS[args.config]()
    print(engine.transcribe(audio).to_json())


if __name__ == "__main__":
    main()
