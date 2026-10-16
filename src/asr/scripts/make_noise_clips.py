"""Generate noise-only clips for the false-command / hallucination check.

Writes 16 kHz mono wavs to data/asr/noise_only/ (gitignored) and their rows to
data/asr/noise_only/manifest.csv (empty reference_text, speaker="generated"):
digital silence, white noise and pink noise at a low (-50 dBFS RMS, a quiet
room) and a medium (-30 dBFS RMS, roughly the level of our speech
recordings) level, each 2, 3 and 4 s long. Fixed random seed, so reruns give
identical files.

Rows of any other speaker in the manifest (e.g. real room-noise recordings
added by hand) are kept.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

ASR_DIR = Path(__file__).resolve().parents[1]  # src/asr
SRC_DIR = ASR_DIR.parent                         # src/ (on sys.path, like the other modules)
REPO_ROOT = SRC_DIR.parent
SAMPLE_RATE = 16000
LEVELS_DBFS = {"low": -50.0, "medium": -30.0}
DURATIONS_S = [2, 3, 4]
FIELDS = ["file", "reference_text", "speaker", "notes"]


def white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """White noise shaped to 1/f power (each octave carries equal energy)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / np.sqrt(freqs), n)


def at_level(noise: np.ndarray, dbfs: float) -> np.ndarray:
    target_rms = 10 ** (dbfs / 20)
    return noise / np.sqrt(np.mean(noise**2)) * target_rms


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noise-only clips")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "data" / "asr" / "noise_only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = args.outdir / "manifest.csv"
    kept = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f) if row["speaker"] != "generated"]

    rng = np.random.default_rng(args.seed)
    generated = []
    for duration in DURATIONS_S:
        n = duration * SAMPLE_RATE
        clips = {f"silence_{duration}s": np.zeros(n)}
        for level, dbfs in LEVELS_DBFS.items():
            clips[f"white_{level}_{duration}s"] = at_level(white(n, rng), dbfs)
            clips[f"pink_{level}_{duration}s"] = at_level(pink(n, rng), dbfs)
        for name, audio in clips.items():
            path = args.outdir / f"{name}.wav"
            sf.write(path, audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
            generated.append({
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "reference_text": "",
                "speaker": "generated",
                "notes": name.rsplit("_", 1)[0],
            })

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept + generated)
    print(f"Wrote {len(generated)} generated clips to {args.outdir} (kept {len(kept)} other rows)")


if __name__ == "__main__":
    main()
