#!/usr/bin/env python3
"""
test_live_srp_clean.py — Live Hardware Diagnostic & Visualization for SRP-CLEAN

Captures live 6-mic audio from the ReSpeaker array in real-time, executes 
2-pass SRP-CLEAN (dominant detection + spatial nulling + secondary recovery),
and outputs:
1. Live real-time console telemetry
2. Terminal ASCII energy distribution diagram
3. Publication-grade diagnostic PNG plot: data/dsp/output/live_srp_clean_diagnostic.png
4. Snapshot WAV files for acoustic verification (raw & residual)
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import scipy.io.wavfile as wavfile

# Set matplotlib cache to a writable temporary directory for headless Linux / Pi
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib.pyplot as plt

# Dynamic path resolution
TEST_DIR = Path(__file__).resolve().parent
DSP_DIR = TEST_DIR.parent
SRC_DIR = DSP_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(DSP_DIR) not in sys.path:
    sys.path.insert(0, str(DSP_DIR))

from source.audio_streamer import AudioStreamer
from source.vad import SileroVAD
from source.localizer import SRPLocalizer

# Hardware configuration
DEVICE = "plughw:seeed8micvoicec,0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 500ms chunks

OUTPUT_DIR = REPO_ROOT / "data" / "dsp" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUTPUT_DIR / "live_srp_clean_diagnostic.png"


def null_dominant_source(localizer: SRPLocalizer, audio: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Subtracts the aligned beam of angle_deg from all 6 microphones,
    placing an exact mathematical spatial null at that direction in frequency domain.
    """
    n_samples = audio.shape[1]
    freqs = np.fft.rfftfreq(n_samples, 1.0 / localizer.sample_rate)
    ang_rad = np.deg2rad(angle_deg)
    tau = -(localizer.radius / localizer.speed_of_sound) * np.cos(ang_rad - localizer.mic_angles_rad)
    steer = np.exp(1j * 2 * np.pi * freqs[None, :] * tau[:, None])

    # Time-align to dominant source
    X = np.fft.rfft(audio, axis=1)
    X_aligned = X * steer

    # Extract dominant source beam (Delay-and-Sum average)
    beam = np.mean(X_aligned, axis=0)

    # Subtract dominant beam from all aligned microphones
    X_residual_aligned = X_aligned - beam[None, :]

    # Un-align back to original microphone phase coordinates
    X_residual = X_residual_aligned * np.conj(steer)
    return np.fft.irfft(X_residual, n=n_samples).astype(np.float32)


def render_ascii_diagram(p_pass1: np.ndarray, p_pass2: np.ndarray, dom_ang: float, sec_ang: Optional[float]):
    """
    Prints an ASCII spatial power bar chart directly to terminal comparing
    Only SRP-PHAT vs. SRP-PHAT + SRP-CLEAN.
    """
    sec_str = f"[{dom_ang:.0f}°, {sec_ang:.0f}°]" if sec_ang is not None else f"[{dom_ang:.0f}°]"
    print("\n" + "=" * 80)
    print("LIVE COMPARISON: ONLY SRP-PHAT vs. SRP-PHAT + SRP-CLEAN")
    print(f"  • Baseline (Only SRP-PHAT):          Detected [{dom_ang:.0f}°] (Secondary masked)")
    print(f"  • Combined (SRP-PHAT + SRP-CLEAN):   Detected {sec_str} (Both resolved!)")
    print("=" * 80)
    print("Angle | [Baseline] Only SRP-PHAT               | [Proposed] SRP-CLEAN Residual")
    print("-" * 80)

    sample_angles = list(range(0, 360, 15))
    for ang in sample_angles:
        v1 = p_pass1[ang]
        v2 = p_pass2[ang]
        bar1 = "█" * int(v1 * 24)
        bar2 = "▓" * int(v2 * 24)

        tag = ""
        if abs(ang - dom_ang) <= 7:
            tag = f" ◄── DOMINANT ({dom_ang:.0f}°)"
        elif sec_ang is not None and abs(ang - sec_ang) <= 7:
            tag = f" ◄── SECONDARY ({sec_ang:.0f}°)"

        print(f"{ang:3d}°  | {v1:4.2f} [{bar1:<24}] | {v2:4.2f} [{bar2:<24}]{tag}")

    print("=" * 80)


def generate_live_plot(
    p_pass1: np.ndarray,
    p_pass2: np.ndarray,
    dom_ang: float,
    sec_ang: Optional[float],
    save_path: Path,
):
    """
    Generates and saves a publication-grade diagnostic PNG comparing 
    live Pass 1 vs. live Pass 2 SRP-CLEAN residual.
    """
    angles = np.arange(360)
    angles_rad = np.deg2rad(angles)

    fig = plt.figure(figsize=(15, 6), dpi=150)

    # ---------------------------------------------------------
    # Panel 1: Polar Directivity Pattern (360°)
    # ---------------------------------------------------------
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_polar.plot(angles_rad, p_pass1, label=f"Pass 1: Dominant ({dom_ang:.0f}°)", color="#d62728", lw=2)
    ax_polar.plot(angles_rad, p_pass2, label="Pass 2: SRP-CLEAN Residual", color="#2ca02c", lw=2, linestyle="--")

    # Mark dominant and secondary peaks
    ax_polar.plot([np.deg2rad(dom_ang)], [p_pass1[int(dom_ang)]], "ro", markersize=9, label=f"Dominant ({dom_ang:.0f}°)")
    if sec_ang is not None:
        ax_polar.plot([np.deg2rad(sec_ang)], [p_pass2[int(sec_ang)]], "g*", markersize=12, label=f"Secondary ({sec_ang:.0f}°)")

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("Live ReSpeaker Array 360° Polar Directivity", pad=15, fontweight="bold", fontsize=12)
    ax_polar.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.25), frameon=True, fontsize=9)
    ax_polar.grid(True, linestyle=":", alpha=0.6)

    # ---------------------------------------------------------
    # Panel 2: Cartesian 360° Comparison
    # ---------------------------------------------------------
    ax_cart = fig.add_subplot(1, 2, 2)
    ax_cart.plot(angles, p_pass1, color="#d62728", lw=2, label=f"Baseline: Only SRP-PHAT (Finds only [{dom_ang:.0f}°])")
    ax_cart.plot(angles, p_pass2, color="#2ca02c", lw=2, linestyle="--", label="Proposed: SRP-CLEAN Residual (Nulls Dominant)")

    # Highlight dominant angle and secondary angle
    ax_cart.axvline(dom_ang, color="red", linestyle=":", lw=1.5)
    ax_cart.scatter([dom_ang], [p_pass1[int(dom_ang)]], color="red", s=100, zorder=5)
    ax_cart.annotate(f"Pass 1: {dom_ang:.0f}°", (dom_ang, p_pass1[int(dom_ang)] + 0.05), ha="center", fontweight="bold", color="darkred")

    if sec_ang is not None:
        ax_cart.axvline(sec_ang, color="green", linestyle=":", lw=1.5)
        ax_cart.scatter([sec_ang], [p_pass2[int(sec_ang)]], color="green", s=120, zorder=5, marker="^")
        ax_cart.annotate(
            f"Pass 2: {sec_ang:.0f}° (Recovered!)",
            (sec_ang, p_pass2[int(sec_ang)] + 0.05),
            ha="center",
            fontweight="bold",
            color="darkgreen",
        )

    ax_cart.set_xlim([0, 360])
    ax_cart.set_ylim([-0.05, 1.20])
    ax_cart.set_xlabel("Steering Angle (Degrees)", fontweight="bold")
    ax_cart.set_ylabel("Normalized Spatial Power", fontweight="bold")
    
    sec_str = f"[{dom_ang:.0f}°, {sec_ang:.0f}°]" if sec_ang is not None else f"[{dom_ang:.0f}°]"
    ax_cart.set_title(f"Baseline: [{dom_ang:.0f}°] vs. Combined SRP-CLEAN: {sec_str}", pad=10, fontweight="bold", fontsize=11)
    ax_cart.legend(loc="upper right", frameon=True, fontsize=9)
    ax_cart.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_live_pipeline():
    print("=" * 80)
    print("LIVE HARDWARE DIAGNOSTIC: SRP-PHAT vs. SRP-CLEAN (0.5s Chunks)")
    print("=" * 80)

    print("\n⏳ [1/2] Warming up Silero VAD & SRP Localizer...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=0.35)
    localizer = SRPLocalizer(sample_rate=SAMPLE_RATE, radius=0.04625, par_threshold=1.15)
    dummy = np.zeros((6, int(SAMPLE_RATE * CHUNK_DURATION)), dtype=np.float32)
    _ = vad.is_speech_active(dummy)
    _ = localizer.localize(dummy)
    print("   ✓ Hardware & DSP models ready in RAM.")

    print("\n🎙️ [2/2] Opening ReSpeaker 6-Mic Hardware Stream (ALSA plughw:seeed8micvoicec,0)...")
    print("=" * 80)
    print("INSTRUCTIONS FOR THIS LIVE TEST:")
    print("  1. Turn on your hospital noise / music source in the room.")
    print("  2. Speak into the microphone array (e.g. from 60° away from the noise).")
    print("  3. Observe Pass 1 locking the noise and Pass 2 revealing your voice.")
    print("  4. Press Ctrl+C when done ──► the script will automatically render & save the PNG plot!")
    print("=" * 80 + "\n")

    frame_count = 0

    # Best multi-source snapshot storage for plotting upon Ctrl+C
    best_snapshot = None
    best_contrast_score = -1.0

    try:
        with AudioStreamer(device=DEVICE, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION) as streamer:
            # Discard ADC startup power-on click
            _ = streamer.read_chunk()

            while True:
                chunk = streamer.read_chunk()
                frame_count += 1
                timestamp = time.strftime("%H:%M:%S")

                peak_amp = int(np.max(np.abs(chunk)) * 32768)
                bars = min(int(peak_amp / 600), 12)
                meter = "█" * bars + "░" * (12 - bars)

                # Gate 1: Voice Activity Detection
                t_vad = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                vad_ms = (time.perf_counter() - t_vad) * 1000

                if not is_speech:
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: SILENCE ({vad_ms:4.1f}ms) ──► Gate 1 ABORT"
                    )
                    continue

                # -------------------------------------------------------------
                # Gate 2: SRP-CLEAN 2-Pass Localization
                # -------------------------------------------------------------
                t_loc = time.perf_counter()

                # Pass 1: Standard SRP-PHAT
                pass1_angles, p_pass1 = localizer.localize(chunk)

                pass2_angles = []
                p_pass2 = np.zeros_like(p_pass1)
                dom_angle = None
                sec_angle = None

                if pass1_angles:
                    dom_angle = pass1_angles[0]
                    # Spatial Nulling: Erase dominant source
                    res_chunk = null_dominant_source(localizer, chunk, dom_angle)
                    # Pass 2: Localize on residual audio
                    raw_pass2, p_pass2 = localizer.localize(res_chunk)

                    for a in raw_pass2:
                        circ_diff = min(abs(a - dom_angle), 360.0 - abs(a - dom_angle))
                        if circ_diff >= 15.0:
                            pass2_angles.append(a)

                    if pass2_angles:
                        sec_angle = pass2_angles[0]

                loc_ms = (time.perf_counter() - t_loc) * 1000
                total_latency_ms = vad_ms + loc_ms

                # Format console telemetry: direct comparison
                only_srp_str = f"Only SRP: {pass1_angles}°" if pass1_angles else "Only SRP: [None]"
                combined_angles = list(pass1_angles) + list(pass2_angles)
                combined_str = f"SRP+CLEAN: {combined_angles}°" if combined_angles else "SRP+CLEAN: [None]"

                print(
                    f"[{timestamp}] #{frame_count:04d} | "
                    f"Peak: {peak_amp:5d} [{meter}] | "
                    f"📊 {only_srp_str:<22} ──► 🚀 {combined_str:<32} | "
                    f"Loc: {loc_ms:4.1f}ms | Latency: {total_latency_ms:4.1f}ms"
                )

                # Track best snapshot containing both dominant and secondary sources
                if dom_angle is not None and sec_angle is not None:
                    # Contrast score: strength of secondary peak in residual
                    contrast = p_pass2[int(sec_angle)]
                    if contrast > best_contrast_score:
                        best_contrast_score = contrast
                        best_snapshot = {
                            "frame": frame_count,
                            "timestamp": timestamp,
                            "p_pass1": p_pass1.copy(),
                            "p_pass2": p_pass2.copy(),
                            "dom_angle": dom_angle,
                            "sec_angle": sec_angle,
                            "raw_chunk": chunk.copy(),
                            "res_chunk": res_chunk.copy(),
                        }
                elif best_snapshot is None and dom_angle is not None:
                    best_snapshot = {
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "p_pass1": p_pass1.copy(),
                        "p_pass2": p_pass2.copy(),
                        "dom_angle": dom_angle,
                        "sec_angle": None,
                        "raw_chunk": chunk.copy(),
                        "res_chunk": chunk.copy(),
                    }

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping live diagnostic (Ctrl+C caught)...")
    except Exception as e:
        print(f"\n❌ Diagnostic runtime error: {e}")
    finally:
        print("\n" + "=" * 80)
        print("GENERATING LIVE DIAGNOSTIC VISUALIZATION & AUDIO DUMP")
        print("=" * 80)

        if best_snapshot is not None:
            dom_a = best_snapshot["dom_angle"]
            sec_a = best_snapshot["sec_angle"]
            p1 = best_snapshot["p_pass1"]
            p2 = best_snapshot["p_pass2"]
            snap_frame = best_snapshot["frame"]
            snap_time = best_snapshot["timestamp"]

            sec_str = f"{sec_a:.0f}°" if sec_a is not None else "None"
            print(f"✓ Selected Frame #{snap_frame:04d} at {snap_time} as best multi-source snapshot:")
            print(f"   • Dominant Source:  {dom_a:.1f}°")
            print(f"   • Secondary Source: {sec_str}")

            # 1. Terminal ASCII Diagram
            render_ascii_diagram(p1, p2, dom_a, sec_a)

            # 2. Publication-Grade Diagnostic Plot
            generate_live_plot(p1, p2, dom_a, sec_a, PLOT_PATH)
            print(f"✓ Diagnostic plot saved to:   {PLOT_PATH}")

            # 3. Save raw and residual audio snapshots for listening
            raw_path = OUTPUT_DIR / "live_snapshot_raw.wav"
            res_path = OUTPUT_DIR / "live_snapshot_residual.wav"
            wavfile.write(str(raw_path), SAMPLE_RATE, (best_snapshot["raw_chunk"][0] * 32767).astype(np.int16))
            wavfile.write(str(res_path), SAMPLE_RATE, (best_snapshot["res_chunk"][0] * 32767).astype(np.int16))
            print(f"✓ Raw audio snapshot saved:   {raw_path}")
            print(f"✓ Residual audio snapshot:    {res_path} (Noise nulled!)")

        else:
            print("⚠️ No speech frames were captured during this test session.")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_live_pipeline()

