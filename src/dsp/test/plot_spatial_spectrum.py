"""
Level 2: Visual Diagnostic Test for SRPLocalizer
Generates Cartesian and Polar spatial spectrum plots for single-speaker,
multi-speaker, and diffuse noise scenarios.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


test_dir = Path(__file__).resolve().parent
dsp_dir = test_dir.parent          # .../src/dsp (where 'source' folder is)
repo_root = dsp_dir.parent.parent  # .../scrub-voice (top-level repo where 'data' folder is)

if str(dsp_dir) not in sys.path:
    sys.path.insert(0, str(dsp_dir))

from source.localizer import SRPLocalizer

# ============================================================
# Acoustic Synthesizer Helper
# ============================================================
def make_source(loc, angle_deg, snr_db=20.0):
    """Generate 6-channel synthetic audio for a sound source at angle_deg."""
    n = int(0.5 * loc.sample_rate)
    t = np.arange(n) / loc.sample_rate
    angle_rad = np.deg2rad(angle_deg)

    # Multi-harmonic speech-like signal (200 - 3400 Hz)
    source = sum(
        np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
        for f in range(200, 3400, 100)
    )
    source /= np.max(np.abs(source))

    # Apply per-mic fractional delays
    S = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, 1.0 / loc.sample_rate)
    audio = np.zeros((loc.num_mics, n), dtype=np.float64)

    for m in range(loc.num_mics):
        tau = -(loc.radius / loc.speed_of_sound) * np.cos(angle_rad - loc.mic_angles_rad[m])
        audio[m] = np.fft.irfft(S * np.exp(-1j * 2 * np.pi * freqs * tau), n=n)

    # Add Gaussian background noise
    sig_power = np.mean(audio ** 2)
    audio += np.random.randn(*audio.shape) * np.sqrt(sig_power / (10 ** (snr_db / 10)))
    audio /= np.max(np.abs(audio)) + 1e-12
    return audio.astype(np.float32)

# ============================================================
# Main Diagnostic Runner
# ============================================================
def run_visual_diagnostic():
    print("🎨 Running Level 2 Visual Diagnostic for SRPLocalizer...")
    loc = SRPLocalizer(sample_rate=16000, radius=0.04625, par_threshold=1.25)

    # Define 3 test scenarios
    scenarios = [
        {
            "title": "Scenario A: Single Speaker (Surgeon @ 60°)",
            "ground_truth": [60.0],
            "audio": make_source(loc, 60.0, snr_db=25.0),
        },
        {
            "title": "Scenario B: Multi-Speaker (Surgeon @ 60° + Nurse @ 180°)",
            "ground_truth": [60.0, 180.0],
            "audio": (make_source(loc, 60.0, snr_db=25.0) + make_source(loc, 180.0, snr_db=25.0)) / 2.0,
        },
        {
            "title": "Scenario C: Diffuse Room Noise (Gate 2 Rejection)",
            "ground_truth": [],
            "audio": np.random.randn(6, 8000).astype(np.float32) * 0.05,
        },
    ]

    fig = plt.figure(figsize=(15, 9))
    plt.subplots_adjust(hspace=0.40, wspace=0.35)

    for idx, sc in enumerate(scenarios):
        angles, power = loc.localize(sc["audio"])
        gate_status = f"Gate 2: PASS ({angles}°)" if angles else "Gate 2: ABORT (No Peaks)"

        # ---------------------------------------------------------
        # Row 1: Cartesian Plot (Power vs 0°..360°)
        # ---------------------------------------------------------
        ax_cart = fig.add_subplot(2, 3, idx + 1)
        ax_cart.plot(loc.angles_deg, power, color="royalblue", lw=2, label="SRP Power P(θ)")

        # Mark ground truth angles
        for gt in sc["ground_truth"]:
            ax_cart.axvline(gt, color="green", ls="--", lw=1.8, label=f"True Source ({gt}°)")

        # Mark detected peaks
        for det in angles:
            peak_val = power[int(det)]
            ax_cart.plot(det, peak_val, "ro", markersize=8)
            ax_cart.annotate(
                f"{det}°",
                (det, peak_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="red",
            )

        ax_cart.set_title(f"{sc['title']}\n{gate_status}", fontsize=9.5, fontweight="bold", pad=8)
        ax_cart.set_xlabel("Azimuth Angle (°)")
        ax_cart.set_ylabel("Normalized Power")
        ax_cart.set_xlim(0, 360)
        ax_cart.set_ylim(-0.05, 1.1)
        ax_cart.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_cart.legend(loc="upper right", fontsize=8)

        # ---------------------------------------------------------
        # Row 2: Polar Plot (Directional Compass)
        # ---------------------------------------------------------
        ax_polar = fig.add_subplot(2, 3, idx + 4, projection="polar")
        theta_rad = np.deg2rad(loc.angles_deg)
        ax_polar.plot(theta_rad, power, color="royalblue", lw=1.5)
        ax_polar.fill(theta_rad, power, color="royalblue", alpha=0.15)

        # Mark physical 6-mic positions on the perimeter
        for m_idx, m_deg in enumerate(loc.mic_angles_deg):
            ax_polar.plot(np.deg2rad(m_deg), 1.05, "k^", markersize=6)
            if idx == 0:
                ax_polar.text(np.deg2rad(m_deg), 1.2, f"M{m_idx+1}", ha="center", fontsize=7)

        # Mark detected angles with red radial beams
        for det in angles:
            det_rad = np.deg2rad(det)
            ax_polar.plot([det_rad, det_rad], [0, 1.0], color="red", lw=2)

        ax_polar.set_theta_zero_location("E")  # 0° on the Right (MIC_6)
        ax_polar.set_theta_direction(1)        # Counter-clockwise
        ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax_polar.set_yticklabels([])
        ax_polar.grid(True, linestyle=":", alpha=0.6)

    # Save to canonical repo data output directory
    output_dir = repo_root / "data" / "dsp" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "srp_diagnostic_dashboard.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✅ Dashboard saved successfully to: {output_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    run_visual_diagnostic()


