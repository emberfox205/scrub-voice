"""
Level 1: Unit Tests for DASBeamformer (Synthetic Data — No Hardware)
"""
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest

dsp_root = Path(__file__).resolve().parent.parent
if str(dsp_root) not in sys.path:
    sys.path.insert(0, str(dsp_root))

from source.beamformer import DASBeamformer

# ============================================================
# Helpers
# ============================================================
def make_simulated_source(bf: DASBeamformer, angle_deg: float, num_samples: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic 6-channel audio with a broadband sound source
    arriving from azimuth angle_deg under far-field plane wave propagation.
    Returns:
        audio: np.ndarray of shape (6, num_samples), float32
        source: np.ndarray of shape (num_samples,), float32 (clean reference)
    """    
    t = np.arange(num_samples) / bf.sample_rate
    angle_rad = np.deg2rad(angle_deg)

    # Broadband signal (speech frequency range 200 Hz - 3500 Hz)
    source = sum(np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi)) for f in range(300, 3100, 200)).astype(np.float64)
    source /= np.max(np.abs(source))

    # Apply per-microphone phase delay in frequency domain
    S = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(num_samples, 1.0 / bf.sample_rate)

    audio = np.zeros((bf.num_mics, num_samples), dtype=np.float64)
    for m in range(bf.num_mics):
        tau = -(bf.radius / bf.speed_of_sound) * np.cos(angle_rad - bf.mic_angles_rad[m])
        audio[m] = np.fft.irfft(S * np.exp(-1j * 2 * np.pi * freqs * tau), n=num_samples)

    return audio.astype(np.float32), source.astype(np.float32)

@pytest.fixture
def bf():
    return DASBeamformer()


# ============================================================
# 1. Input Validation
# ============================================================
def test_wrong_channels_raises(bf):
    with pytest.raises(ValueError):
        bf.beamform(np.zeros((4, 8000), dtype=np.float32), target_angle_deg=0.0)

def test_1d_input_raises(bf):
    with pytest.raises(ValueError):
        bf.beamform(np.zeros(8000, dtype=np.float32), target_angle_deg=0.0)

def test_short_chunk_returns_zeros(bf):
    """Chunk shorter than 512 frame_size should return zero vector without crash."""
    out = bf.beamform(np.zeros((6, 200), dtype=np.float32), target_angle_deg=90.0)
    assert out.shape == (200,)
    assert np.all(out == 0.0)


# ============================================================
# 2. Output Contract
# ============================================================
def test_output_shape_and_dtype(bf):
    chunk = np.random.randn(6, 8000).astype(np.float32)
    out = bf.beamform(chunk, target_angle_deg=180.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8000,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_silence_input_gives_clean_silence(bf):
    silence = np.zeros((6, 8000), dtype=np.float32)
    out = bf.beamform(silence, target_angle_deg=45.0)
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()

# ============================================================
# 3. Spatial Selectivity (Constructive vs Destructive Gain)
# ============================================================
@pytest.mark.parametrize("target_angle", [0.0, 90.0, 180.0, 270.0])
def test_on_axis_vs_off_axis_attenuation(bf, target_angle):
    """
    Source at target_angle:
    - Steering at target_angle (on-axis) must have significantly higher energy
      than steering at opposite direction (off-axis, 180° away).
    """

    audio, _ = make_simulated_source(bf, angle_deg=target_angle)

    opposite_angle = (target_angle + 180.0) % 360.0

    on_axis_audio = bf.beamform(audio, target_angle_deg=target_angle)   
    off_axis_audio = bf.beamform(audio, target_angle_deg=opposite_angle)

    # Compute RMS energy excluding edge frames
    on_axis_energy = np.mean(on_axis_audio[512:-512] ** 2)
    off_axis_energy = np.mean(off_axis_audio[512:-512] ** 2)

    # On-axis must be substantially louder than opposite direction (at least 3x power / > 5 dB)
    assert on_axis_energy > off_axis_energy * 3.0, (
        f"Angle {target_angle}°: on-axis energy {on_axis_energy:.5f} not substantially larger "
        f"than off-axis energy {off_axis_energy:.5f}"
    )


# ============================================================
# 4. Waveform Preservation (Correlation with Clean Source)
# ============================================================
def test_on_axis_correlation(bf):
    """Reconstructed on-axis audio must preserve waveform shape of the source."""
    audio, clean_ref = make_simulated_source(bf, angle_deg=120.0)
    enhanced = bf.beamform(audio, target_angle_deg=120.0)

    # Normalize and compute Pearson correlation in steady-state region
    seg_ref = clean_ref[1000:7000]
    seg_enh = enhanced[1000:7000]

    corr = np.corrcoef(seg_ref, seg_enh)[0, 1]
    assert corr > 0.90, f"Expected high waveform correlation, got {corr:.3f}"

# ============================================================
# 5. Multi-Source Interface (beamform_multi)
# ============================================================
def test_beamform_multi_empty(bf):
    chunk = np.random.randn(6, 8000).astype(np.float32)
    res = bf.beamform_multi(chunk, target_angles_deg=[])
    assert res == []

def test_beamform_multi_multiple_targets(bf):
    chunk = np.random.randn(6, 8000).astype(np.float32)
    angles = [60.0, 180.0]
    res = bf.beamform_multi(chunk, target_angles_deg=angles)

    assert len(res) == 2
    assert res[0].shape == (8000,)
    assert res[1].shape == (8000,)
    assert res[0].dtype == np.float32
    assert res[1].dtype == np.float32

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])