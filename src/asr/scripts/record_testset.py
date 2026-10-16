"""Record a test set: shows one phrase at a time, records you saying it,
saves a 16 kHz mono wav and appends a row to a manifest csv.

Phrase lists live in data/asr/prompts/ (one phrase per line; "## ..." is a
comment; "# name" starts a new section, and the name - e.g. the intent - is
stored in the manifest's `notes` column). Run from the repository root, once
per speaker:

    python src/asr/scripts/record_testset.py --speaker josias --prompts-file data/asr/prompts/commands_dev.txt
    python src/asr/scripts/record_testset.py --speaker josias --prompts-file data/asr/prompts/non_command.txt

Recordings go to data/asr/<list name>/ (e.g. data/asr/commands_dev/) with a
manifest.csv next to them; both are gitignored (recordings of people are
never committed). Re-running with the same --speaker skips phrases you
already recorded, so you can stop anytime (Ctrl+C) and continue later.

To record a phrase again (e.g. it was cut off), overwriting its wav:

    python src/asr/scripts/record_testset.py --speaker josias --prompts-file data/asr/prompts/commands_dev.txt --duration 7 --redo "Log event, first incision, note patient vitals stable"
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "asr"
SAMPLE_RATE = 16000
MANIFEST_FIELDS = ["file", "reference_text", "speaker", "notes"]


def load_prompts(prompts_file: Path) -> list[tuple[str, str]]:
    """(phrase, section) pairs; the section name ends up in the notes column."""
    prompts = []
    section = prompts_file.stem
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip() or prompts_file.stem
        elif line:
            prompts.append((line, section))
    return prompts


def load_already_recorded(manifest_path: Path, speaker: str) -> dict[str, str]:
    """phrase -> wav path (as in the manifest) for this speaker."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["reference_text"]: row["file"] for row in reader if row["speaker"] == speaker}


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def record_clip(duration: float) -> np.ndarray:
    for n in (3, 2, 1):
        print(f"  {n}...", end=" ", flush=True)
        time.sleep(0.6)
    print("speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def speech_at_the_end(audio: np.ndarray) -> bool:
    """True if the last 0.2 s is still loud (at least 10% of the peak level
    in some 10 ms frame): the speaker was probably cut off."""
    tail = audio[-int(0.2 * SAMPLE_RATE):]
    frames = tail[: len(tail) // 160 * 160].reshape(-1, 160)
    return float(np.sqrt((frames ** 2).mean(axis=1)).max()) > 0.1 * float(np.abs(audio).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an ASR test set, one phrase at a time")
    parser.add_argument("--speaker", required=True, help="Your name or an id, e.g. josias")
    parser.add_argument("--prompts-file", type=Path, required=True, help="Phrase list from data/asr/prompts/")
    parser.add_argument("--outdir", type=Path, default=None, help="Default: data/asr/<prompts file name>/")
    parser.add_argument("--manifest", type=Path, default=None, help="Default: <outdir>/manifest.csv")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds to record per clip")
    parser.add_argument("--redo", action="append", default=[], metavar="PHRASE",
                        help="Record an already recorded phrase again, overwriting its wav (repeat for several)")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    outdir = (args.outdir or DATA_DIR / args.prompts_file.stem).resolve()
    manifest = args.manifest or outdir / "manifest.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    done = load_already_recorded(manifest, args.speaker)
    if args.redo:
        unknown = [t for t in args.redo if t not in done]
        if unknown:
            sys.exit(f"Not recorded yet for speaker '{args.speaker}', can't redo: {unknown}")
        remaining = [(text, notes) for text, notes in prompts if text in args.redo]
    else:
        remaining = [(text, notes) for text, notes in prompts if text not in done]
    if not remaining:
        print(f"Nothing left to record for speaker '{args.speaker}' - all {len(prompts)} phrases are done.")
        return

    print(f"{len(remaining)} of {len(prompts)} phrases left for speaker '{args.speaker}' -> {outdir}")
    print("For each one: press Enter to start, speak after the countdown, then review.")
    print("Type 's' + Enter to skip a phrase, 'q' + Enter to stop.\n")

    recorded_this_session = 0
    for i, (text, notes) in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] ({notes}) Say: \"{text}\"")
        choice = input("  Press Enter to record ('s'=skip, 'q'=quit): ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            print("  Skipped.\n")
            continue

        while True:
            audio = record_clip(args.duration)
            peak = float(np.abs(audio).max())
            if peak < 0.01:
                print(f"  Very quiet (peak={peak:.3f}) - mic may not be picking you up. Redo? [Y/n]: ", end="")
                if input().strip().lower() != "n":
                    continue
            elif peak > 0.99:
                print(f"  Clipping (peak={peak:.3f}) - too loud. Redo? [Y/n]: ", end="")
                if input().strip().lower() != "n":
                    continue
            elif speech_at_the_end(audio):
                print(f"  Still speaking at the end of the {args.duration:g} s window - cut off? "
                      "(use a longer --duration) Redo? [Y/n]: ", end="")
                if input().strip().lower() != "n":
                    continue
            break

        if text in args.redo:
            # Overwrite the existing clip; its manifest row stays as it is.
            sf.write(REPO_ROOT / done[text], audio, SAMPLE_RATE, subtype="PCM_16")
            recorded_this_session += 1
            print(f"  Re-recorded {Path(done[text]).name} (peak={peak:.2f})\n")
            continue

        existing = list(outdir.glob(f"{args.speaker}_*.wav"))
        clip_path = outdir / f"{args.speaker}_{len(existing) + 1:03d}.wav"
        sf.write(clip_path, audio, SAMPLE_RATE, subtype="PCM_16")

        # Manifest paths are relative to the repository root (what the benchmark expects).
        try:
            file_entry = clip_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            file_entry = clip_path.as_posix()
        append_manifest_row(
            manifest,
            {"file": file_entry, "reference_text": text, "speaker": args.speaker, "notes": notes},
        )
        recorded_this_session += 1
        print(f"  Saved {clip_path.name} (peak={peak:.2f})\n")

    print(f"Done. Recorded {recorded_this_session} clip(s) this session.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
