#!/usr/bin/env python3
"""Interactive ASR Audio File Tester.

Allows testing any audio file (DSP stage outputs, internet recordings, or AI-generated
TTS speech) against the Vosk speech recognition engine.

Reports:
1. Audio duration, inference time, and Real-Time Factor (RTF).
2. Transcribed text and overall confidence.
3. Word-level timestamps and individual word confidences.
4. Surgical grammar vocabulary hit rate.
5. (Optional) End-to-end NLU intent & slot extraction preview.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.asr.audio import load_audio
from src.asr.bench_configs import CONFIGS, PRODUCTION_CONFIG
from src.asr.vocab import vosk_grammar_phrases


def inspect_audio_asr(
    audio_paths: List[Path],
    config_name: str = PRODUCTION_CONFIG,
    run_nlu_check: bool = True,
) -> None:
    """Runs ASR and optional NLU check on provided audio files."""
    print("=" * 70)
    print("                  SCRUB-VOICE ASR AUDIO TESTER")
    print("=" * 70)
    print(f"  Configuration : {config_name}")
    print(f"  Engine Model  : Vosk Small EN-US (0.15)")
    print(f"  Files Count   : {len(audio_paths)}")
    print("=" * 70)

    # 1. Initialize ASR engine
    print(f"\n[*] Loading ASR engine ({config_name})...")
    t0_load = time.perf_counter()
    engine = CONFIGS[config_name]()
    t_load = time.perf_counter() - t0_load
    print(f"    -> Engine loaded in {t_load:.2f}s.\n")

    # 2. Optionally load NLU pipeline for preview
    nlu_pipeline = None
    if run_nlu_check:
        try:
            from src.nlu.pipeline import VoiceToIntentPipeline
            print("[*] Loading NLU Pipeline for intent preview...")
            nlu_pipeline = VoiceToIntentPipeline(num_threads=1)
            print("    -> NLU Pipeline loaded.\n")
        except Exception as e:
            print(f"[!] Warning: Could not load NLU pipeline: {e}\n")

    grammar_words = set()
    for phrase in vosk_grammar_phrases():
        for w in phrase.split():
            grammar_words.add(w.lower())

    results = []

    for idx, wav_path in enumerate(audio_paths, 1):
        if not wav_path.exists():
            print(f"[-] File not found: {wav_path}")
            continue

        print(f"\n[{idx}/{len(audio_paths)}] Processing: {wav_path.name}")
        print("-" * 70)

        # Load audio (automatically resamples to 16 kHz mono float32)
        audio = load_audio(wav_path)
        audio_dur = len(audio) / 16000.0

        t0_infer = time.perf_counter()
        asr_result = engine.transcribe(audio)
        infer_time = time.perf_counter() - t0_infer
        rtf = infer_time / max(0.01, audio_dur)

        transcript = asr_result.text.strip()
        confidence = asr_result.confidence
        words = asr_result.words

        print(f"  Audio Duration   : {audio_dur:.2f}s")
        print(f"  Inference Latency: {infer_time * 1000.0:.1f}ms  (RTF: {rtf:.3f})")
        print(f"  Transcription    : \"{transcript if transcript else '[NO SPEECH DETECTED]'}\"")
        print(f"  Mean Confidence  : {confidence:.2%}")

        # Vocabulary check
        if words:
            in_vocab_count = sum(1 for w in words if w.word.lower() in grammar_words)
            vocab_ratio = in_vocab_count / len(words)
            print(f"  Grammar Coverage : {in_vocab_count}/{len(words)} words ({vocab_ratio:.1%})")

            # Word breakdown
            print("\n  Word-Level Timestamps:")
            for w in words:
                marker = "✓" if w.word.lower() in grammar_words else "○"
                print(f"    [{w.start:5.2f}s - {w.end:5.2f}s] {marker} {w.word:<15} (conf: {w.confidence:.2%})")

        # NLU preview
        nlu_info = None
        if nlu_pipeline and transcript:
            try:
                envelope = nlu_pipeline.process(transcript)
                nlu_info = {
                    "intent": envelope.get("intent"),
                    "category": envelope.get("category"),
                    "slots": envelope.get("slots"),
                    "status": envelope.get("status"),
                    "priority": envelope.get("priority"),
                }
                print("\n  NLU Command Interpretation:")
                print(f"    -> Intent   : {envelope.get('intent')} ({envelope.get('category')})")
                print(f"    -> Status   : {envelope.get('status')} [Priority: {envelope.get('priority')}]")
                print(f"    -> Slots    : {json.dumps(envelope.get('slots'))}")
            except Exception as e:
                print(f"    -> NLU Error: {e}")

        results.append({
            "file": wav_path.name,
            "duration": audio_dur,
            "latency_ms": infer_time * 1000.0,
            "rtf": rtf,
            "transcript": transcript,
            "confidence": confidence,
            "nlu": nlu_info,
        })
        print("-" * 70)

    # Multi-file Summary Table
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("                        BATCH SUMMARY")
        print("=" * 70)
        print(f"{'File':<25} | {'Dur(s)':<6} | {'RTF':<6} | {'Intent':<16} | {'Transcript':<25}")
        print("-" * 70)
        for r in results:
            intent = r["nlu"]["intent"] if r["nlu"] else "N/A"
            text = (r["transcript"][:22] + "...") if len(r["transcript"]) > 22 else r["transcript"]
            print(f"{r['file'][:25]:<25} | {r['duration']:<6.2f} | {r['rtf']:<6.3f} | {intent:<16} | {text:<25}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test speech recognition on arbitrary WAV files")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help="One or more WAV files or directories containing audio",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default=PRODUCTION_CONFIG,
        help="ASR configuration to use",
    )
    parser.add_argument(
        "--no-nlu",
        dest="run_nlu",
        action="store_false",
        help="Disable downstream NLU intent classification preview",
    )
    parser.set_defaults(run_nlu=True)

    args = parser.parse_args()

    files: List[Path] = []
    if not args.inputs:
        # Default sample files in repo
        sample = REPO_ROOT / "data" / "asr" / "sample_01.wav"
        if sample.exists():
            files.append(sample)
    else:
        for p in args.inputs:
            if p.is_dir():
                files.extend(sorted(p.glob("*.wav")))
            elif p.is_file():
                files.append(p)

    if not files:
        print("No audio files found to process.")
        sys.exit(1)

    inspect_audio_asr(files, config_name=args.config, run_nlu_check=args.run_nlu)


if __name__ == "__main__":
    main()

