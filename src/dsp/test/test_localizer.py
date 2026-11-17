"""
Level 1: Unit Tests for SRPLocalizer (Synthetic Data — No Hardware)
"""
import sys
from pathlib import Path
import numpy as np
import pytest
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from source.localizer import SRPLocalizer

# ============================================================
# Helpers
# ============================================================
def make_source(loc, angle_deg, snr_db=20.0):
    """Generate fake 6-mic audio with a source at a known angle."""
    n = int(0.5 * loc.sample_rate)
    t = np.arange(n) / loc.sample_rate
    angle_rad = np.deg2rad(angle_deg)
    # Broadband speech-like signal
    source = sum(
        np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
        for f in range(200, 3400, 100)
    )
    source /= np.max(np.abs(source))
    # Apply per-mic delays in frequency domain
    S = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, 1.0 / loc.sample_rate)
    audio = np.zeros((loc.num_mics, n), dtype=np.float64)
    for m in range(loc.num_mics):
        tau = -(loc.radius / loc.speed_of_sound) * np.cos(angle_rad - loc.mic_angles_rad[m])
        audio[m] = np.fft.irfft(S * np.exp(-1j * 2 * np.pi * freqs * tau), n=n)
    # Add noise at controlled SNR
    sig_power = np.mean(audio ** 2)
    audio += np.random.randn(*audio.shape) * np.sqrt(sig_power / 10 ** (snr_db / 10))
    audio /= np.max(np.abs(audio)) + 1e-12
    return audio.astype(np.float32)

def circ_dist(a, b):
    """Shortest angular distance on a 360° circle."""
    d = abs(a - b)
    return min(d, 360.0 - d)

@pytest.fixture
def loc():
    return SRPLocalizer()

# ============================================================
# 1. Input Validation
# ============================================================
def test_wrong_channels_raises(loc):
    with pytest.raises(ValueError):
        loc.localize(np.zeros((4, 8000), dtype=np.float32))
def test_1d_input_raises(loc):
    with pytest.raises(ValueError):
        loc.localize(np.zeros(8000, dtype=np.float32))
def test_too_short_returns_empty(loc):
    angles, power = loc.localize(np.zeros((6, 100), dtype=np.float32))
    assert angles == []
    assert power.shape == (360,)

# ============================================================
# 2. Silence & Noise Rejection
# ============================================================
def test_silence_returns_empty(loc):
    angles, _ = loc.localize(np.zeros((6, 8000), dtype=np.float32))
    assert angles == []

def test_diffuse_noise_rejected(loc):
    noise = np.random.randn(6, 8000).astype(np.float32)
    noise /= np.max(np.abs(noise))
    angles, _ = loc.localize(noise)
    assert len(angles) <= 2

# ============================================================
# 3. Output Shape & Range
# ============================================================
def test_output_types(loc):
    audio = make_source(loc, 60.0)
    angles, power = loc.localize(audio)
    assert isinstance(angles, list)
    assert isinstance(power, np.ndarray)
    assert power.shape == (360,)
    
def test_normalized_power_range(loc):
    audio = make_source(loc, 60.0)
    _, power = loc.localize(audio)
    assert np.min(power) >= -1e-6
    assert np.max(power) <= 1.0 + 1e-6


# ============================================================
# 4. Single Source Detection
# ============================================================
@pytest.mark.parametrize("angle", [0.0, 60.0, 90.0, 180.0, 270.0, 359.0])
def test_single_source(loc, angle):
    audio = make_source(loc, angle, snr_db=25.0)
    angles, _ = loc.localize(audio)
    assert len(angles) >= 1, f"No detection for source at {angle}°"
    closest = min(angles, key=lambda a: circ_dist(a, angle))
    assert circ_dist(closest, angle) <= 15.0, f"Source at {angle}°, got {closest}°"

# ============================================================
# 5. Boundary Wrap-Around (0° / 360°)
# ============================================================
def test_boundary_no_duplicate(loc):
    audio = make_source(loc, 2.0, snr_db=25.0)
    angles, _ = loc.localize(audio)
    near_zero = [a for a in angles if circ_dist(a, 2.0) <= 15.0]
    assert len(near_zero) == 1, f"Expected 1 near 0°, got {near_zero}"


# ============================================================
# 6. Multiple Sources
# ============================================================
def test_two_sources(loc):
    mixed = (make_source(loc, 60.0, snr_db=25.0) + make_source(loc, 180.0, snr_db=25.0)) / 2
    angles, _ = loc.localize(mixed)
    assert len(angles) >= 2, f"Expected 2 sources, got {angles}"
    assert any(circ_dist(a, 60.0) <= 15.0 for a in angles), f"60° missing: {angles}"
    assert any(circ_dist(a, 180.0) <= 15.0 for a in angles), f"180° missing: {angles}"

def test_close_sources_merged(loc):
    """Sources < 15° apart should merge into 1 detection (distance=15 limit)."""
    mixed = (make_source(loc, 60.0) + make_source(loc, 70.0)) / 2
    angles, _ = loc.localize(mixed)
    near_65 = [a for a in angles if circ_dist(a, 65.0) <= 15.0]
    assert len(near_65) == 1, f"Expected 1 merged peak near 65°, got {near_65}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])