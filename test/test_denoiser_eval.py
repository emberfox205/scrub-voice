#!/usr/bin/env python3
# ==============================================================================
# HELPER / RUN COMMANDS:
#   # 1. Comparative denoiser matrix evaluation on single audio file:
#   python test/test_denoiser_eval.py --audio data/asr/test_clip_event.wav
#
#   # 2. Batch evaluation across all files matching glob:
#   python test/test_denoiser_eval.py --audio "data/asr/sample_*.wav"
#
#   # 3. Custom output report:
#   python test/test_denoiser_eval.py --audio data/asr/test_mix_voice.wav --out results/denoiser_eval.json
# ==============================================================================
"""Unified DSP Denoiser & ASR/NLU Comparative Benchmark Suite.

Evaluates acoustic denoiser configurations (Beamformed baseline, Old full-band
denoiser, New Split-Band Consonant Shield, and Dialed-back attenuation) against
the surgical Vosk ASR grammar model and downstream NLU intent classification.

Key Invariants:
1. Single-Load In-Memory Pipeline: Loads Vosk and SetFit NLU exactly once,
   eliminating the 15-20s model initialization overhead across files.
2. Multi-Condition Matrix: Evaluates Beamformed, Denoised Old (Full-band 20dB),
   Denoised New (Split-band 2.5kHz 20dB), and Denoised Dialed-back (10dB).
3. Ground Truth Evaluation: Calculates Word Error Rate (WER), keyword hit rate,
   phonetic consonant preservation (e.g., /θ/ in 'three' vs 'ray'), and NLU intent match.
4. Batch Audio Support: Evaluates built-in clinical audio datasets or arbitrary files/globs.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.asr.audio import load_audio
from src.asr.bench_configs import CONFIGS, PRODUCTION_CONFIG
from src.asr.vocab import vosk_grammar_phrases
from src.dsp.source.beamformer import DASBeamformer
from src.dsp.source.localizer import SRPLocalizer
from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import DuoFilterStreamingDenoiser

# Default Clinical Benchmark Dataset with Ground Truth Captions
BENCHMARK_CASES = [
    {
        "id": "live_capture",
        "path": REPO_ROOT / "data" / "asr" / "live_capture.wav",
        "ground_truth": "zoom in 3 times, zoom out 3 times, next, next slice, zoom in three times",
        "key_words": ["three", "slice", "zoom"],
    },
    {
        "id": "recording_4",
        "path": REPO_ROOT / "data" / "asr" / "recording_4.wav",
        "ground_truth": "zoom in 2 times end zoom out 2 times end next slice, next slice zoom..",
        "key_words": ["two", "slice", "zoom"],
    },
    {
        "id": "test_mix_voice",
        "path": REPO_ROOT / "data" / "wav_files" / "test_mix_voice.wav",
        "ground_truth": "zoom in 2 time zoom out 3 times next slice",
        "key_words": ["two", "three", "slice"],
    },
    {
        "id": "test_Minh_voice",
        "path": REPO_ROOT / "data" / "asr" / "test_Minh_voice.wav",
        "ground_truth": "Zoom in 2 time zoom in 3 time zoom in 4 time",
        "key_words": ["two", "three", "four"],
    },
]

NUMBER_MAP = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "0": "zero", "x": "times", "3x": "three times", "2x": "two times",
}


def normalize_words(text: str) -> List[str]:
    """Canonicalizes transcript words for robust WER and accuracy comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    expanded = []
    for t in tokens:
        if t in NUMBER_MAP:
            expanded.extend(NUMBER_MAP[t].split())
        else:
            expanded.append(t)
    return expanded


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) using Levenshtein distance."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[len(ref), len(hyp)]) / len(ref)


def beamform_audio(data: np.ndarray, sr: int) -> np.ndarray:
    """Performs spatial localization and Delay-and-Sum beamforming on multichannel audio."""
    if data.ndim == 1:
        return data

    if data.shape[0] > data.shape[1]:
        data = data.T  # -> [channels, samples]
    if data.shape[0] > 6:
        data = data[:6, :]  # Keep first 6 channels of 8-channel array

    localizer = SRPLocalizer(sample_rate=sr, radius=0.04625)
    loc_chunk = data[:, : min(data.shape[1], sr)]
    angles, _ = localizer.localize(loc_chunk)
    angle = float(angles[0]) if len(angles) > 0 else 0.0

    beamformer = DASBeamformer(sample_rate=sr, radius=0.04625)
    beam = beamformer.beamform(data, target_angles=angle)
    if isinstance(beam, list):
        beam = beam[0]
    return beam.astype(np.float32)


@dataclass
class EvaluationResult:
    condition_name: str
    transcript: str
    confidence: float
    wer: float
    intent: str
    status: str
    slots: Dict[str, Any]
    rtf: float
    preserved_keywords: List[str]
    missing_keywords: List[str]


class DenoiserBenchmarkRunner:
    """Manages preloaded ASR & NLU engines and runs denoiser conditions."""

    def __init__(
        self,
        config_name: str = PRODUCTION_CONFIG,
        run_nlu: bool = True,
    ):
        print("=" * 70)
        print("         SCRUB-VOICE DENOISER & ASR/NLU BENCHMARK SUITE")
        print("=" * 70)
        sys.stdout.flush()

        t0 = time.perf_counter()
        print(f"[*] Initializing Vosk ASR Engine ({config_name})...", end="", flush=True)
        self.asr_engine = CONFIGS[config_name]()
        print(f" done ({time.perf_counter() - t0:.2f}s).", flush=True)

        self.nlu_pipeline = None
        if run_nlu:
            t1 = time.perf_counter()
            print("[*] Initializing NLU Intent Pipeline...", end="", flush=True)
            try:
                from src.nlu.pipeline import VoiceToIntentPipeline
                self.nlu_pipeline = VoiceToIntentPipeline(num_threads=1)
                print(f" done ({time.perf_counter() - t1:.2f}s).", flush=True)
            except Exception as e:
                print(f" failed ({e}). Proceeding without NLU.", flush=True)

        # Pre-initialize denoiser models
        print("[*] Pre-warming Neural Denoiser ONNX models...", end="", flush=True)
        # Condition B: Old Full-Band @ 20dB
        self.denoiser_old_20db = DuoFilterStreamingDenoiser(atten_lim_db=20.0, split_band_hz=None)
        # Condition C: New Split-Band Consonant Shield @ 2.5kHz, 20dB
        self.denoiser_new_shield = DuoFilterStreamingDenoiser(atten_lim_db=20.0, split_band_hz=2500.0)
        # Condition D: Dialed-Back Full-Band @ 10dB
        self.denoiser_dialed_10db = DuoFilterStreamingDenoiser(atten_lim_db=10.0, split_band_hz=None)
        # Condition E: Split-Band Shield @ 10dB
        self.denoiser_shield_10db = DuoFilterStreamingDenoiser(atten_lim_db=10.0, split_band_hz=2500.0)
        print(" done.\n", flush=True)

    def evaluate_audio(
        self,
        audio_mono: np.ndarray,
        reference_text: str,
        condition_name: str,
        expected_keywords: List[str],
    ) -> EvaluationResult:
        """Transcribes single-channel audio and evaluates against reference text."""
        duration = len(audio_mono) / 16000.0
        t0 = time.perf_counter()
        asr_res = self.asr_engine.transcribe(audio_mono)
        infer_time = time.perf_counter() - t0
        rtf = infer_time / max(1e-4, duration)

        transcript = asr_res.text.strip()
        confidence = asr_res.confidence or 0.0
        wer = calculate_wer(reference_text, transcript) if reference_text else 0.0

        # Check key phonetic keywords
        norm_tokens = set(normalize_words(transcript))
        preserved = []
        missing = []
        for kw in expected_keywords:
            if kw.lower() in norm_tokens:
                preserved.append(kw)
            else:
                missing.append(kw)

        # Run NLU
        intent = "N/A"
        status = "N/A"
        slots = {}
        if self.nlu_pipeline and transcript:
            try:
                env = self.nlu_pipeline.process(transcript)
                intent = env.get("intent", "unknown")
                status = env.get("status", "unknown")
                slots = env.get("slots", {})
            except Exception as e:
                intent = f"ERR: {e}"

        return EvaluationResult(
            condition_name=condition_name,
            transcript=transcript,
            confidence=confidence,
            wer=wer,
            intent=intent,
            status=status,
            slots=slots,
            rtf=rtf,
            preserved_keywords=preserved,
            missing_keywords=missing,
        )

    def process_file(
        self,
        file_path: Path,
        ground_truth: str = "",
        expected_keywords: Optional[List[str]] = None,
        export_dir: Optional[Path] = None,
    ) -> Dict[str, EvaluationResult]:
        """Runs all denoiser conditions on one multi-channel or single-channel audio file."""
        expected_keywords = expected_keywords or []
        print(f"\n{'='*70}")
        print(f"  TEST FILE: {file_path.name}")
        if ground_truth:
            print(f"  GROUND TRUTH: \"{ground_truth}\"")
        print(f"{'='*70}")
        sys.stdout.flush()

        data, sr = sf.read(str(file_path), dtype="float32")
        duration = (data.shape[0] if data.ndim == 1 else data.shape[1]) / sr

        # 1. Spatial Beamforming
        t0 = time.perf_counter()
        beamformed = beamform_audio(data, sr)
        t_beam = time.perf_counter() - t0

        # 2. Generate Denoiser variants
        clean_old_20db = self.denoiser_old_20db.enhance(beamformed)
        clean_new_shield = self.denoiser_new_shield.enhance(beamformed)
        clean_dialed_10db = self.denoiser_dialed_10db.enhance(beamformed)
        clean_shield_10db = self.denoiser_shield_10db.enhance(beamformed)

        # Optionally export WAVs for acoustic listening
        if export_dir:
            export_dir.mkdir(parents=True, exist_ok=True)
            stem = file_path.stem
            sf.write(str(export_dir / f"{stem}_1_beamformed.wav"), beamformed, 16000)
            sf.write(str(export_dir / f"{stem}_2_old_20db.wav"), clean_old_20db, 16000)
            sf.write(str(export_dir / f"{stem}_3_new_shield_20db.wav"), clean_new_shield, 16000)
            sf.write(str(export_dir / f"{stem}_4_dialed_10db.wav"), clean_dialed_10db, 16000)
            sf.write(str(export_dir / f"{stem}_5_shield_10db.wav"), clean_shield_10db, 16000)

        # 3. Transcribe and evaluate each condition
        conditions = [
            ("Non-Denoised (Beamformed)", beamformed),
            ("Denoised Old (Full-Band @ 20dB)", clean_old_20db),
            ("Denoised New (Consonant Shield @ 2.5kHz, 20dB)", clean_new_shield),
            ("Denoised Dialed-Back (Full-Band @ 10dB)", clean_dialed_10db),
            ("Denoised Shield + 10dB (Shield @ 2.5kHz, 10dB)", clean_shield_10db),
        ]

        results = {}
        for cond_name, audio in conditions:
            res = self.evaluate_audio(audio, ground_truth, cond_name, expected_keywords)
            results[cond_name] = res

            status_mark = "✅" if res.wer < 0.20 else ("⚠️" if res.wer < 0.45 else "❌")
            print(f"  [{cond_name}]")
            print(f"    Transcript : \"{res.transcript}\"")
            print(f"    Metrics    : WER: {res.wer:.1%} | Conf: {res.confidence:.1%} | RTF: {res.rtf:.3f} | {status_mark}")
            if res.missing_keywords:
                print(f"    Missing KW : {res.missing_keywords} (Consonant cut!)")
            if res.intent != "N/A":
                print(f"    NLU Intent : {res.intent} [{res.status}]")
            print("-" * 70)
            sys.stdout.flush()

        return results


def print_comparison_markdown_table(all_results: Dict[str, Dict[str, EvaluationResult]]) -> None:
    """Prints a beautiful markdown table summarizing side-by-side results."""
    print("\n" + "=" * 80)
    print("                    SIDE-BY-SIDE BENCHMARK COMPARISON TABLE")
    print("=" * 80)

    header = (
        "| Test Audio File | Condition | Transcript | WER | Conf | Consonant Status | Intent |\n"
        "| :--- | :--- | :--- | :---: | :---: | :--- | :--- |"
    )
    print(header)

    for file_name, cond_results in all_results.items():
        for cond_name, res in cond_results.items():
            short_cond = cond_name.replace("Denoised ", "").replace("Non-Denoised ", "")
            tr = res.transcript or "[NO SPEECH]"
            wer_str = f"{res.wer:.1%}" if res.wer is not None else "-"
            conf_str = f"{res.confidence:.1%}"
            if not res.missing_keywords:
                consonant_status = "✅ Preserved"
            else:
                consonant_status = f"❌ Lost {res.missing_keywords}"

            intent_str = f"{res.intent}" if res.intent != "N/A" else "-"
            print(f"| `{file_name}` | {short_cond} | \"{tr}\" | {wer_str} | {conf_str} | {consonant_status} | {intent_str} |")

    print("=" * 80 + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Unified Denoiser & ASR/NLU Benchmark Suite")
    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        help="Optional audio files to test (if omitted, runs standard clinical benchmark dataset)",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default=PRODUCTION_CONFIG,
        help="ASR configuration (default: vosk-grammar+freeform-log)",
    )
    parser.add_argument(
        "--no-nlu",
        action="store_true",
        help="Disable NLU intent classification preview",
    )
    parser.add_argument(
        "--export-wavs",
        type=Path,
        default=None,
        help="Directory to save intermediate beamformed and denoised WAV files",
    )
    args = parser.parse_args()

    runner = DenoiserBenchmarkRunner(config_name=args.config, run_nlu=not args.no_nlu)
    all_results: Dict[str, Dict[str, EvaluationResult]] = {}

    if not args.files:
        # Run standard 4 clinical benchmark cases
        print(f"[*] Running standard clinical benchmark suite ({len(BENCHMARK_CASES)} cases)...")
        for case in BENCHMARK_CASES:
            p = case["path"]
            if not p.exists():
                print(f"[!] Warning: File {p} not found, skipping.")
                continue
            res = runner.process_file(
                file_path=p,
                ground_truth=case["ground_truth"],
                expected_keywords=case.get("key_words", []),
                export_dir=args.export_wavs,
            )
            all_results[case["id"]] = res
    else:
        # Run user-supplied files
        target_files = []
        for p in args.files:
            if p.is_dir():
                target_files.extend(sorted(p.glob("*.wav")))
            elif p.is_file():
                target_files.append(p)

        print(f"[*] Running custom batch on {len(target_files)} file(s)...")
        for f in target_files:
            res = runner.process_file(
                file_path=f,
                ground_truth="",
                expected_keywords=[],
                export_dir=args.export_wavs,
            )
            all_results[f.name] = res

    if all_results:
        print_comparison_markdown_table(all_results)


if __name__ == "__main__":
    main()

