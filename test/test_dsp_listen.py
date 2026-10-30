#!/usr/bin/env python3
"""Interactive DSP Acoustic Stage Listener.

Takes any input WAV file (multi-channel or single-channel), runs each DSP
transformation stage offline, and saves intermediate audio files so the tester
can listen to each step back-to-back:
  Stage 1: Raw Microphone Input (unfiltered ambient acoustic audio)
  Stage 2: DAS Beamformed Audio (spatially filtered towards detected DOA angle)
  Stage 3: Neural Denoised Audio (DuoFilter / DeepFilterNet ONNX suppression)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.dsp.source.beamformer import DASBeamformer
from src.dsp.source.localizer import SRPLocalizer
from src.dsp.source.vad import SileroVAD
from src.dsp.denoise.pi_reference_duofilter import DuoFilterDenoiser
from test.benchmarks.common.acoustics import simulate_array, NoiseSource


def compute_rms(signal: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) energy."""
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) + 1e-12


def process_and_dump_stages(
    input_wav: Path,
    out_dir: Path,
    denoiser_type: str = "duofilter",
    vad_threshold: float = 0.40,
    force_simulate_array: bool = False,
) -> dict:
    """Processes input audio stage-by-stage and saves WAVs for acoustic listening."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Audio
    data, sr = sf.read(str(input_wav), dtype="float32")
    
    # Ensure (channels, samples)
    if data.ndim == 1:
        # Single channel input
        raw_mono = data
        if force_simulate_array:
            print(f"[*] Single-channel input detected. Simulating 6-mic circular array (target 45°, SNR 5dB)...")
            sim_array = simulate_array(
                raw_mono,
                angle_deg=45.0,
                radius=0.04625,
                noise=NoiseSource(None, sr),
                snr_db=5.0,
                sr=sr,
                num_mics=6,
            )
            multichannel = sim_array
            raw_mic1 = sim_array[0]
        else:
            multichannel = None
            raw_mic1 = raw_mono
    else:
        # Multi-channel input
        if data.shape[0] > data.shape[1]:
            data = data.T  # Convert [samples, channels] to [channels, samples]
        # Keep first 6 microphone channels if more are present (e.g. 8-channel ReSpeaker with 2 loopbacks)
        if data.shape[0] > 6:
            data = data[:6, :]
        multichannel = data
        raw_mic1 = multichannel[0]
        print(f"[*] Multi-channel input detected: {multichannel.shape[0]} channels, {multichannel.shape[1]} samples.")

    # 2. Gate 1: Silero VAD (Voice Activity Detection)
    vad = SileroVAD(sample_rate=sr, threshold=vad_threshold)
    is_speech = vad.is_speech_active(multichannel if multichannel is not None else np.tile(raw_mono, (6, 1)))

    # 3. Stage 2: SRP-PHAT Localization & DAS Beamforming
    detected_angle = 0.0
    if multichannel is not None and multichannel.shape[0] >= 4:
        print("[*] Running SRP-PHAT Acoustic Localizer...")
        localizer = SRPLocalizer(sample_rate=sr, radius=0.04625)
        # Take a 1-second segment for localization
        loc_chunk = multichannel[:, : min(multichannel.shape[1], sr)]
        angles, conf = localizer.localize(loc_chunk)
        detected_angle = float(angles[0]) if len(angles) > 0 else 0.0
        print(f"    -> Estimated Speaker Angle of Arrival (DOA): {detected_angle:.1f}°")

        print("[*] Running Delay-and-Sum (DAS) Spatial Beamformer...")
        beamformer = DASBeamformer(sample_rate=sr, radius=0.04625)
        beamformed = beamformer.beamform(multichannel, target_angles=detected_angle)
        if isinstance(beamformed, list):
            beamformed = beamformed[0]
    else:
        print("[!] Single-channel input (no array). Beamforming bypassed.")
        beamformed = raw_mic1
        detected_angle = 0.0

    # 4. Stage 3: Neural Denoiser (DuoFilter / DeepFilterNet)
    print(f"[*] Running Neural Noise Suppressor ({denoiser_type})...")
    denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "onnx" / f"{denoiser_type}.onnx"
    if not denoiser_model.exists():
        denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / f"{denoiser_type}.onnx"

    denoiser = DuoFilterDenoiser(model_path=denoiser_model, threads=1)
    denoised = denoiser.enhance(beamformed, sample_rate=sr)

    # 5. Energy Metrics
    rms_raw = compute_rms(raw_mic1)
    rms_beam = compute_rms(beamformed)
    rms_denoised = compute_rms(denoised)

    delta_beam_db = 20 * math.log10(rms_beam / rms_raw)
    delta_denoise_db = 20 * math.log10(rms_denoised / rms_beam)
    total_delta_db = 20 * math.log10(rms_denoised / rms_raw)

    # 6. Save Audio Stages
    stem = input_wav.stem
    p_raw = out_dir / f"{stem}_stage1_raw_mic1.wav"
    p_beam = out_dir / f"{stem}_stage2_beamformed.wav"
    p_clean = out_dir / f"{stem}_stage3_denoised.wav"

    sf.write(str(p_raw), raw_mic1, sr)
    sf.write(str(p_beam), beamformed, sr)
    sf.write(str(p_clean), denoised, sr)

    duration_sec = len(raw_mic1) / sr

    print("\n" + "=" * 65)
    print("           DSP STAGE-BY-STAGE AUDIO EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Input File        : {input_wav.name}")
    print(f"  Duration          : {duration_sec:.2f} seconds")
    print(f"  Speech Detected   : {'YES' if is_speech else 'NO (Silence/Noise)'}")
    print(f"  Speaker Angle     : {detected_angle:.1f}°")
    print(f"  Beamforming Delta : {delta_beam_db:+.2f} dB")
    print(f"  Denoising Delta   : {delta_denoise_db:+.2f} dB (Noise attenuated)")
    print(f"  Total Suppression : {total_delta_db:+.2f} dB")
    print("-" * 65)
    print("  LISTEN TO EACH STAGE IN ORDER:")
    print(f"  [1] RAW INPUT    : {p_raw.resolve()}")
    print(f"  [2] BEAMFORMED   : {p_beam.resolve()}")
    print(f"  [3] DENOISED     : {p_clean.resolve()}")
    print("=" * 65 + "\n")

    return {
        "raw_path": str(p_raw),
        "beamformed_path": str(p_beam),
        "denoised_path": str(p_clean),
        "angle": detected_angle,
        "delta_db": total_delta_db,
    }


def main():
    parser = argparse.ArgumentParser(description="Listen to DSP output stage-by-stage")
    default_input = REPO_ROOT / "data" / "dsp" / "diagnostic" / "inturn_with_binarymask" / "live_raw.wav"
    if not default_input.exists():
        default_input = REPO_ROOT / "data" / "asr" / "test_mix_voice.wav"

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help="Path to input audio file (WAV)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=REPO_ROOT / "artifacts" / "dsp_listen",
        help="Directory to save stage-by-stage audio files",
    )
    parser.add_argument(
        "--denoiser-type",
        choices=["duofilter", "deepfilter"],
        default="duofilter",
        help="Neural denoiser architecture",
    )
    parser.add_argument(
        "--simulate-array",
        action="store_true",
        help="Force simulation of 6-microphone array if input is mono",
    )

    args = parser.parse_args()
    if not args.input.exists():
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    process_and_dump_stages(
        input_wav=args.input,
        out_dir=args.out_dir,
        denoiser_type=args.denoiser_type,
        force_simulate_array=args.simulate_array,
    )


if __name__ == "__main__":
    main()

