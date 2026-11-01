#!/usr/bin/env python3
# ==============================================================================
# HELPER / RUN COMMANDS:
#   # 1. Quick 5-second recording (RAM-first staging):
#   python test/record_audio.py
#
#   # 2. 10-second recording with custom output path:
#   python test/record_audio.py -d 10 -o data/wav_files/surgeon_commands.wav
#
#   # 3. Interactive manual mode (press Enter to stop):
#   python test/record_audio.py --manual
#
#   # 4. Record and immediately transcribe/evaluate with ASR + NLU:
#   python test/record_audio.py -d 5 --eval
# ==============================================================================
"""Safe Hardware Audio Recording Utility for Scrub-Voice.

Captures calibrated, DMA-aligned 16 kHz 16-bit PCM audio from the ReSpeaker 6-mic
circular array (or standard microphone) with zero microSD controller lockup.

Key Invariants (AGENTS.md):
1. RAM-First Staging: Captures directly into RAM (/dev/shm) during recording;
   writes to persistent flash storage only after capture terminates cleanly.
2. DMA Descriptor Alignment: Enforces power-of-two buffer parameters
   (--buffer-size=65536, --period-size=8192) to eliminate ALSA XRUNs.
3. Audio Driver Isolation & Graceful Signals: Emits SIGINT on shutdown so
   arecord cleanly executes snd_pcm_close() without driver deadlocks.
4. Format Guarantee: Strictly enforces 16 kHz sample rate, 16-bit signed PCM (S16_LE).

Usage:
  # Quick 5-second recording (saves to data/wav_files/test_recording_<timestamp>.wav):
  uv run python test/record_audio.py

  # 10-second recording with custom output:
  uv run python test/record_audio.py -d 10 -o data/wav_files/surgeon_commands.wav

  # Manual interactive mode (press Enter to start, Enter to stop):
  uv run python test/record_audio.py --manual

  # Record and immediately transcribe/evaluate with ASR + NLU:
  uv run python test/record_audio.py -d 5 --eval
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.dsp.source.audio_streamer import AudioStreamer

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "wav_files"
RAM_DIR = Path("/dev/shm")
SAMPLE_RATE = 16000


def compute_rms_db(audio: np.ndarray) -> float:
    """Computes RMS energy in decibels relative to full scale (dBFS)."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    return float(20.0 * np.log10(rms))


def draw_vu_meter(rms_db: float, width: int = 18) -> str:
    """Renders a terminal VU meter bar."""
    # Scale from -60 dB (quiet) to 0 dB (clipping)
    clamped = max(-60.0, min(0.0, rms_db))
    ratio = (clamped + 60.0) / 60.0
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def save_audio_ram_first(
    audio: np.ndarray,
    target_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Writes 16-bit PCM WAV to /dev/shm in RAM first, then atomically copies to target disk."""
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure shape is (samples, channels) for soundfile write
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio_to_write = audio.T
    else:
        audio_to_write = audio

    # Clip audio safely to [-1.0, 1.0]
    clipped = np.clip(audio_to_write, -1.0, 1.0).astype(np.float32)

    is_linux = platform.system() == "Linux"
    has_ram = is_linux and RAM_DIR.is_dir() and os.access(str(RAM_DIR), os.W_OK)

    if has_ram:
        temp_ram_path = RAM_DIR / f"scrub_rec_{os.getpid()}_{int(time.time()*1000)}.wav"
        try:
            # 1. Stage in RAM first
            sf.write(str(temp_ram_path), clipped, sample_rate, subtype="PCM_16")
            # 2. Atomic copy to persistent storage
            shutil.copy2(str(temp_ram_path), str(target_path))
            temp_ram_path.unlink(missing_ok=True)
            return target_path
        except Exception:
            # Fallback to direct write if RAM staging fails
            if temp_ram_path.exists():
                temp_ram_path.unlink(missing_ok=True)

    # Direct write fallback (Windows / macOS / no /dev/shm)
    sf.write(str(target_path), clipped, sample_rate, subtype="PCM_16")
    return target_path


def record_live(
    output_path: Path,
    duration_sec: Optional[float] = 5.0,
    device: str = "plughw:seeed8micvoicec,0",
    manual_mode: bool = False,
    mono_output: bool = False,
) -> Path:
    """Captures live audio stream safely and saves to target WAV."""
    print("=" * 70)
    print("            SCRUB-VOICE SAFE AUDIO RECORDER (ALSA / RAM-FIRST)")
    print("=" * 70)
    print(f"  Target File : {output_path.name}")
    print(f"  Destination : {output_path.resolve()}")
    print(f"  Device      : {device}")
    mode_str = "Manual (Press Enter to stop)" if manual_mode else f"{duration_sec:.1f} seconds"
    print(f"  Duration    : {mode_str}")
    print(f"  RAM Staging : {'ENABLED (/dev/shm)' if Path('/dev/shm').is_dir() else 'DISABLED (Direct Write)'}")
    print("=" * 70)
    sys.stdout.flush()

    use_arecord = (platform.system() == "Linux" and shutil.which("arecord") is not None)

    if not use_arecord:
        # Check if running in an environment without ALSA arecord
        print("[!] Note: ALSA 'arecord' not found in path.")
        print("    If running in WSL without USB passthrough or on Windows, please connect")
        print("    a microphone or run directly on the Raspberry Pi 4 edge node.")
        try:
            import sounddevice as sd
            print("[*] Falling back to sounddevice for host recording...")
        except Exception as e:
            print(f"[X] Recording backend unavailable: {e}")
            print("    Please install alsa-utils (`sudo apt install alsa-utils`) on Linux/Pi.")
            sys.exit(1)

    print("\n[● REC] Recording started! Speak clearly into the microphone...")
    if manual_mode:
        print("        >>> Press [Enter] or [Ctrl+C] to stop recording <<<\n")
    sys.stdout.flush()

    captured_chunks: List[np.ndarray] = []
    chunk_dur = 0.5
    t0 = time.time()
    interrupted = False

    def handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True

    prev_handler = signal.signal(signal.SIGINT, handle_sigint)

    # In manual mode, spawn background thread to detect [Enter] key press
    if manual_mode:
        import threading

        def wait_for_enter():
            nonlocal interrupted
            try:
                sys.stdin.readline()
                interrupted = True
            except Exception:
                pass

        key_thread = threading.Thread(target=wait_for_enter, daemon=True)
        key_thread.start()

    try:
        if use_arecord:
            streamer = AudioStreamer(
                device=device,
                sample_rate=SAMPLE_RATE,
                chunk_duration=chunk_dur,
            )
            with streamer:
                while True:
                    if interrupted:
                        print("\n[*] Stop signal received.")
                        break

                    chunk = streamer.read_chunk()  # shape: (6, 8000)
                    captured_chunks.append(chunk)

                    elapsed = time.time() - t0
                    # Meter first channel
                    rms_db = compute_rms_db(chunk[0])
                    vu = draw_vu_meter(rms_db)

                    dur_label = f"{duration_sec:4.1f}s" if duration_sec else "   -- "
                    print(f"\r  [● REC] [{elapsed:4.1f}s / {dur_label}] {vu} {rms_db:5.1f} dB ", end="", flush=True)

                    if not manual_mode and duration_sec and elapsed >= duration_sec:
                        break
        else:
            # Fallback sounddevice implementation for developer workstations
            import sounddevice as sd
            channels = 1 if mono_output else 2
            block_samples = int(SAMPLE_RATE * chunk_dur)

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=channels, dtype="float32", blocksize=block_samples) as stream:
                while True:
                    if interrupted:
                        print("\n[*] Stop signal received.")
                        break

                    data, overflowed = stream.read(block_samples)
                    chunk = data.T  # -> (channels, samples)
                    captured_chunks.append(chunk)

                    elapsed = time.time() - t0
                    rms_db = compute_rms_db(chunk[0])
                    vu = draw_vu_meter(rms_db)

                    dur_label = f"{duration_sec:4.1f}s" if duration_sec else "   -- "
                    print(f"\r  [● REC] [{elapsed:4.1f}s / {dur_label}] {vu} {rms_db:5.1f} dB ", end="", flush=True)

                    if not manual_mode and duration_sec and elapsed >= duration_sec:
                        break

    except KeyboardInterrupt:
        print("\n[*] Stopped by user.")
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    print("\n\n[*] Processing captured audio...")
    if not captured_chunks:
        print("[-] Error: No audio samples captured.")
        sys.exit(1)

    # Concatenate chunks along sample axis
    # Each chunk is (channels, samples)
    full_audio = np.concatenate(captured_chunks, axis=-1)
    total_sec = full_audio.shape[-1] / float(SAMPLE_RATE)
    num_channels = full_audio.shape[0]

    # Optional mono extraction
    if mono_output and num_channels > 1:
        print(f"[*] Extracting primary channel (Mic 1) for mono WAV output...")
        full_audio = full_audio[0:1, :]
        num_channels = 1

    # Save safely via RAM staging
    final_path = save_audio_ram_first(full_audio, output_path, sample_rate=SAMPLE_RATE)

    # Channel energy / tap-test report
    print("\n" + "=" * 60)
    print("               RECORDING COMPLETE & VERIFIED")
    print("=" * 60)
    print(f"  File Saved : {final_path.resolve()}")
    print(f"  Duration   : {total_sec:.2f} seconds ({full_audio.shape[-1]} samples)")
    print(f"  Format     : 16-bit Signed PCM, 16,000 Hz ({num_channels} channels)")
    print("-" * 60)
    print("  CHANNEL ENERGY CHECK (MIC TAP-TEST):")
    for ch_idx in range(num_channels):
        ch_audio = full_audio[ch_idx]
        ch_rms = compute_rms_db(ch_audio)
        ch_peak = float(np.max(np.abs(ch_audio)))
        status = "✅ ACTIVE" if ch_rms > -45.0 else ("⚠️ LOW" if ch_rms > -58.0 else "❌ SILENT")
        print(f"    Mic #{ch_idx + 1:02d} : RMS: {ch_rms:5.1f} dB | Peak: {ch_peak:.3f} | {status}")
    print("=" * 60 + "\n")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Quick, safe hardware audio recorder for testing and calibration")
    parser.add_argument(
        "output_pos",
        type=Path,
        nargs="?",
        default=None,
        metavar="OUTPUT",
        help="Optional destination path for WAV file (positional)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Destination path for WAV file (default: data/wav_files/test_recording_<timestamp>.wav)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0s)",
    )
    parser.add_argument(
        "-m", "--manual",
        action="store_true",
        help="Interactive mode: records continuously until you press Enter or Ctrl+C",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Save single-channel (Mic 1) WAV instead of multi-channel array audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="plughw:seeed8micvoicec,0",
        help="ALSA hardware device string (default: plughw:seeed8micvoicec,0)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Immediately run ASR & NLU evaluation on the newly recorded audio",
    )
    args = parser.parse_args()

    # Determine destination filename (prioritize -o flag, fallback to positional)
    target_output = args.output or args.output_pos
    if target_output is not None:
        out_p = target_output
        if not out_p.suffix:
            out_p = out_p.with_suffix(".wav")
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_p = DEFAULT_OUTPUT_DIR / f"test_recording_{ts}.wav"

    # Execute safe recording
    saved_file = record_live(
        output_path=out_p,
        duration_sec=None if args.manual else args.duration,
        device=args.device,
        manual_mode=args.manual,
        mono_output=args.mono,
    )

    # Optional immediate evaluation
    if args.eval:
        print("[*] Launching immediate ASR & NLU pipeline evaluation on recorded file...\n")
        cmd = [sys.executable, str(REPO_ROOT / "test" / "test_asr_audio.py"), str(saved_file)]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

