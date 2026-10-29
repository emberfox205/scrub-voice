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
        bf.beamform(np.zeros((4, 8000), dtype=np.float32), target_angles=0.0)

def test_1d_input_raises(bf):
    with pytest.raises(ValueError):
        bf.beamform(np.zeros(8000, dtype=np.float32), target_angles=0.0)

def test_short_chunk_returns_zeros(bf):
    """Chunk shorter than 512 frame_size should return zero vector without crash."""
    out = bf.beamform(np.zeros((6, 200), dtype=np.float32), target_angles=90.0)
    assert out.shape == (200,)
    assert np.all(out == 0.0)


# ============================================================
# 2. Output Contract
# ============================================================
def test_output_shape_and_dtype(bf):
    chunk = np.random.randn(6, 8000).astype(np.float32)
    out = bf.beamform(chunk, target_angles=180.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8000,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_silence_input_gives_clean_silence(bf):
    silence = np.zeros((6, 8000), dtype=np.float32)
    out = bf.beamform(silence, target_angles=45.0)
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

    on_axis_audio = bf.beamform(audio, target_angles=target_angle)   
    off_axis_audio = bf.beamform(audio, target_angles=opposite_angle)

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
    enhanced = bf.beamform(audio, target_angles=120.0)

    # Normalize and compute Pearson correlation in steady-state region
    seg_ref = clean_ref[1000:7000]
    seg_enh = enhanced[1000:7000]

    corr = np.corrcoef(seg_ref, seg_enh)[0, 1]
    assert corr > 0.90, f"Expected high waveform correlation, got {corr:.3f}"

# ============================================================
# 5. Multi-Source Interface (beamform_multi)
# ============================================================
def test_beamform_empty_list(bf):
    """Passing an empty list of angles returns an empty list."""
    chunk = np.random.randn(6, 8000).astype(np.float32)
    res = bf.beamform(chunk, target_angles=[])
    assert res == []

def test_beamform_multiple_targets(bf):
    """Passing a list of 2 angles returns a list of 2 separated audio streams."""
    chunk = np.random.randn(6, 8000).astype(np.float32)
    angles = [60.0, 180.0]
    res = bf.beamform(chunk, target_angles=angles)

    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].shape == (8000,)
    assert res[1].shape == (8000,)
    assert res[0].dtype == np.float32
    assert res[1].dtype == np.float32

def test_wiener_masking_cross_talk_reduction(bf):
    """
    Simulate two simultaneous speakers at 60° and 180°.
    Verify that spatial masking suppresses speaker 2 from stream 1.
    """
    audio_spk1, _ = make_simulated_source(bf, angle_deg=60.0)
    audio_spk2, _ = make_simulated_source(bf, angle_deg=180.0)
    mixed = audio_spk1 + audio_spk2

    streams = bf.beamform(mixed, target_angles=[60.0, 180.0], apply_wiener_mask=True)
    assert len(streams) == 2
    assert streams[0].shape == (8000,)
    assert streams[1].shape == (8000,)
    assert not np.isnan(streams[0]).any()
    assert not np.isnan(streams[1]).any()  

def test_wiener_mask_actually_attenuates_interferer(bf):
    """
    Verify that enabling the Wiener mask strictly reduces 
    the energy of the interfering speaker in the target beam.
    """
    audio_spk1, _ = make_simulated_source(bf, angle_deg=60.0)
    audio_spk2, _ = make_simulated_source(bf, angle_deg=180.0)

    # When spk1 is silent, check how much of spk2 leaks through beam 1
    # Feed ONLY speaker 2 into beamformer steered at [60, 180]
    leak_raw = bf.beamform(audio_spk2, target_angles=[60.0, 180.0], apply_wiener_mask=False)[0]
    leak_wiener = bf.beamform(audio_spk2, target_angles=[60.0, 180.0], apply_wiener_mask=True)[0]

    raw_leak_energy = np.mean(leak_raw[512:-512] ** 2)
    wiener_leak_energy = np.mean(leak_wiener[512:-512] ** 2)

    # Wiener mask must reduce the leak energy by at least 3x (> 5 dB improvement over pure DAS)
    assert wiener_leak_energy <= raw_leak_energy / 3.0, (
        f"Wiener mask failed to reduce leakage by at least 3x: "
        f"raw={raw_leak_energy:.6f}, wiener={wiener_leak_energy:.6f}, "
        f"ratio={raw_leak_energy / (wiener_leak_energy + 1e-12):.2f}x"
    )

    def test_wiener_mask_floor_and_bounds(bf):
        """
        - Test the wiener_mask method directly with synthetic STFT tensors
        - Verify that gain values stay between floor and 1.0
        """
        # 2 targets, 257 freqs, 30 frames
        # Target 0 has high power, Target 1 has near-zero power   
        Y_stft = np.zeros((2, 257, 30), dtype=np.complex64)
        Y_stft[0, :, :] = 10.0 + 0j
        Y_stft[1, :, :] = 0.01 + 0j

        masked = bf.wiener_mask(Y_stft.copy())

        # Target 0 should be largely untouched (near 10.0)
        assert np.allclose(np.abs(masked[0]), 10.0, rtol=0.1)

        # Target 1 should be clamped at exactly floor (0.10 of its energy)
        assert np.all(np.abs(masked[1]) >= 0.01 * 0.10 - 1e-6)    

    def test_wiener_mask_three_speakers(bf):
        # Passing 3 distinct angles must return 3 valid streams
        chunk = np.random.randn(6, 8000).astype(np.float32)
        angles = [0.0, 120.0, 240.0]

        streams = bf.beamform(chunk, target_angles=angles, apply_wiener_mask=True)

        assert len(streams) == 3
        for s in streams:
            assert s.shape == (8000,)
            assert s.dtype == np.float32
            assert not np.isnan(s).any() 

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])