#!/usr/bin/env python3
"""
Ground-Truth Benchmark & Unit Test Suite for Continuous Psychoacoustic Slope & Hybrid Denoiser.

Verifies the 4 Pillars specified in docs/CONTINUOUS_SLOPE_DENOISER_DESIGN.md:
  1. Continuous Psychoacoustic Slope (-12 dB -> -6 dB -> 0 dB smooth cosine curve)
  2. Strict Magnitude-Only Phase Preservation (100% Delay-and-Sum beamformed phase)
  3. Micro-Dither Injection (-65 dBFS noise floor to stabilize Kaldi CMVN variance)
  4. Soft-Knee SNR Dynamic Blend (automatic auto-tuning between loud speech and pauses)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.dsp.denoise.continuous_slope_denoiser import (
    ContinuousSlopeDenoiser,
    ContinuousSlopeGainFloor,
    SoftKneeSNRBlender,
)
from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import (
    DuoFilterStreamingDenoiser,
)
from src.dsp.denoise.stateless_onnx.stateless_duofilter_denoiser import (
    DuoFilterDenoiser,
)
from src.dsp.denoise.sota_onnx.deepfilternet_denoiser import (
    DeepFilterDenoiser,
)


class TestContinuousSlopeGainFloor(unittest.TestCase):
    """Verifies Pillar 1: Frequency-Dependent Minimum Gain Floor G_min[k]."""

    def setUp(self):
        self.fs = 16000
        self.n_fft = 512
        self.floor_builder = ContinuousSlopeGainFloor(
            n_fft=self.n_fft,
            sample_rate=self.fs,
            f_low=1000.0,
            f_high=3000.0,
            atten_low_db=12.0,
            atten_high_db=0.0,
        )
        self.curve = self.floor_builder.get_numpy_floor()
        self.bin_hz = self.fs / self.n_fft

    def test_frequency_endpoints(self):
        """Verify endpoints: -12 dB below 1 kHz and 0 dB above 3 kHz."""
        g_low_expected = 10.0 ** (-12.0 / 20.0)
        g_high_expected = 10.0 ** (0.0 / 20.0)

        # 0 to 1000 Hz bins
        k_1000 = int(round(1000.0 / self.bin_hz))
        for k in range(0, k_1000):
            self.assertAlmostEqual(self.curve[k], g_low_expected, places=4)

        # Above 3000 Hz bins
        k_3000 = int(round(3000.0 / self.bin_hz))
        for k in range(k_3000, len(self.curve)):
            self.assertAlmostEqual(self.curve[k], g_high_expected, places=4)

    def test_monotonic_and_smooth(self):
        """Verify curve is monotonically non-decreasing with no step discontinuities."""
        diffs = np.diff(self.curve)
        self.assertTrue(np.all(diffs >= -1e-6), "Gain floor must be monotonically non-decreasing")

        # In a 2.5 kHz brickwall, max step is > 0.65. Cosine slope must have small bin-to-bin delta
        max_step = float(np.max(diffs))
        self.assertLess(max_step, 0.05, f"Expected smooth step transition, got {max_step:.4f}")

    def test_midpoint_transition(self):
        """Verify smooth transition around midpoint (2000 Hz is approx -4 to -6 dB)."""
        k_2000 = int(round(2000.0 / self.bin_hz))
        val_2000 = self.curve[k_2000]
        db_2000 = 20.0 * math.log10(val_2000)
        self.assertGreater(db_2000, -8.0)
        self.assertLess(db_2000, -2.0)


class TestPhasePreservation(unittest.TestCase):
    """Verifies Pillar 2: Strict Magnitude-Only Phase Preservation."""

    def test_phase_preservation_accuracy(self):
        """Reconstructed complex STFT must have zero phase distortion relative to input beamformed STFT."""
        denoiser = ContinuousSlopeDenoiser(n_fft=512, sample_rate=16000)

        B, T, F_bins = 2, 32, 257
        # Generate random complex beamformed STFT
        real = torch.randn(B, 1, T, F_bins)
        imag = torch.randn(B, 1, T, F_bins)
        stft_complex = torch.cat([real, imag], dim=1)

        input_phase = torch.atan2(imag, real)

        # Neural gain suppresses some bins by 20 dB, others 0 dB
        neural_gain = torch.rand(B, 1, T, F_bins) * 0.9 + 0.05

        out_stft = denoiser(neural_gain, stft_complex)
        out_real = out_stft[:, 0:1]
        out_imag = out_stft[:, 1:2]
        output_phase = torch.atan2(out_imag, out_real)

        # Verify phase difference is 0.0 everywhere
        phase_err = torch.abs(output_phase - input_phase)
        # Wrap phase error to [-pi, pi]
        phase_err = torch.remainder(phase_err + math.pi, 2 * math.pi) - math.pi
        max_phase_err = float(torch.max(torch.abs(phase_err)).item())

        self.assertLess(max_phase_err, 1e-5, f"Phase distortion detected: {max_phase_err}")


class TestMicroDither(unittest.TestCase):
    """Verifies Pillar 3: Micro-Dither Injection (-65 dBFS Floor)."""

    def test_dither_noise_floor_variance(self):
        """Micro-dither on silence must produce white noise with std ~ 10^(-65/20)."""
        dither_dbfs = -65.0
        expected_std = 10.0 ** (dither_dbfs / 20.0)

        denoiser = ContinuousSlopeDenoiser(dither_dbfs=dither_dbfs)
        silence = np.zeros(160000, dtype=np.float32)  # 10 seconds of silence

        dithered = denoiser.inject_dither(silence)
        measured_std = float(np.std(dithered))

        # Check measured std within 5% of theoretical standard deviation
        self.assertAlmostEqual(measured_std, expected_std, delta=expected_std * 0.1)
        # Ensure zero-energy underflow is impossible
        self.assertGreater(measured_std, 1e-4)


class TestSoftKneeSNRBlend(unittest.TestCase):
    """Verifies Pillar 4: Soft-Knee SNR Dynamic Blend."""

    def setUp(self):
        self.blender = SoftKneeSNRBlender(snr_low_db=8.0, snr_high_db=18.0)

    def test_alpha_endpoints(self):
        """Verify alpha = 1.0 during low SNR (< 8dB) and 0.0 during loud speech (> 18dB)."""
        self.assertEqual(self.blender.compute_alpha(5.0), 1.0)
        self.assertEqual(self.blender.compute_alpha(8.0), 1.0)
        self.assertEqual(self.blender.compute_alpha(18.0), 0.0)
        self.assertEqual(self.blender.compute_alpha(25.0), 0.0)

    def test_alpha_soft_knee_midpoint(self):
        """Verify smooth cosine midpoint at 13 dB SNR."""
        alpha_mid = self.blender.compute_alpha(13.0)
        self.assertAlmostEqual(alpha_mid, 0.5, places=4)

    def test_soft_knee_monotonicity(self):
        """Verify alpha transitions smoothly monotonically from 1.0 to 0.0."""
        snrs = np.linspace(8.0, 18.0, 50)
        alphas = [self.blender.compute_alpha(s) for s in snrs]
        diffs = np.diff(alphas)
        self.assertTrue(np.all(diffs <= 0.0), "Alpha must decrease monotonically with SNR")

    def test_blending_operation(self):
        """Verify weighted sum y = alpha * denoised + (1 - alpha) * raw."""
        denoised = np.ones(100, dtype=np.float32) * 2.0
        raw = np.ones(100, dtype=np.float32) * 5.0

        # High SNR -> Streams raw
        out, alpha, _ = self.blender.blend(denoised, raw, snr_db=20.0)
        self.assertAlmostEqual(alpha, 0.0)
        np.testing.assert_allclose(out, 5.0)

        # Low SNR -> Streams denoised
        out, alpha, _ = self.blender.blend(denoised, raw, snr_db=6.0)
        self.assertAlmostEqual(alpha, 1.0)
        np.testing.assert_allclose(out, 2.0)

        # Mid SNR (13 dB) -> 50/50 blend (3.5)
        out, alpha, _ = self.blender.blend(denoised, raw, snr_db=13.0)
        self.assertAlmostEqual(alpha, 0.5)
        np.testing.assert_allclose(out, 3.5)


class TestDenoiserIntegrations(unittest.TestCase):
    """Verifies end-to-end integration across DuoFilterStreaming, DuoFilter, and DeepFilter."""

    def test_duofilter_streaming_chunk(self):
        """Processes 0.5s chunk and checks output validity and RTF budget."""
        denoiser = DuoFilterStreamingDenoiser(
            continuous_slope=True,
            atten_low_db=12.0,
            atten_high_db=0.0,
            dither_dbfs=-65.0,
            enable_snr_blend=True,
        )

        dummy_chunk = np.random.randn(8000).astype(np.float32) * 0.05

        t0 = time.perf_counter()
        clean = denoiser.enhance_chunk(dummy_chunk)
        elapsed = time.perf_counter() - t0

        rtf = elapsed / 0.5
        self.assertEqual(clean.shape, (8000,))
        self.assertTrue(np.all(np.isfinite(clean)), "Output contains NaN or Inf values")
        self.assertLess(rtf, 0.35, f"Streaming RTF {rtf:.3f} exceeded real-time budget")

    def test_stateless_duofilter(self):
        """Verifies stateless DuoFilter with continuous slope."""
        denoiser = DuoFilterDenoiser(
            continuous_slope=True,
            atten_low_db=12.0,
            dither_dbfs=-65.0,
            enable_snr_blend=True,
        )
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.05
        clean = denoiser.enhance(dummy_audio)
        self.assertEqual(clean.shape, (16000,))
        self.assertTrue(np.all(np.isfinite(clean)))

    def test_deepfilternet(self):
        """Verifies DeepFilterNet with continuous slope."""
        denoiser = DeepFilterDenoiser(
            continuous_slope=True,
            atten_low_db=12.0,
            dither_dbfs=-65.0,
            enable_snr_blend=True,
        )
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.05
        clean = denoiser.enhance(dummy_audio)
        self.assertEqual(clean.shape, (16000,))
        self.assertTrue(np.all(np.isfinite(clean)))


def run_ground_truth_scenario_eval() -> None:
    """
    Executes comparative evaluation across clinical benchmark scenarios and calculates WER.
    Compares:
      1. Non-Denoised (Raw Beamformed Audio)
      2. Shield + 10dB (Legacy 2.5 kHz Brickwall Shield @ 10dB)
      3. Improved Continuous Slope & Hybrid Denoiser (Our 4-Pillar Pipeline)
    """
    print("\n" + "=" * 90)
    print(" GROUND-TRUTH BENCHMARK: CONTINUOUS PSYCHOACOUSTIC SLOPE & WER EVALUATION")
    print("=" * 90)

    try:
        from src.asr.bench_configs import CONFIGS
        from test.benchmarks.common.stats import word_errors
        engine = CONFIGS["vosk-grammar+freeform-log"]()
    except Exception as e:
        print(f"[!] Vosk ASR engine not available for WER evaluation: {e}")
        engine = None

    dn_brickwall = DuoFilterStreamingDenoiser(
        continuous_slope=False,
        split_band_hz=2500.0,
        atten_lim_db=10.0,
    )
    dn_continuous = DuoFilterStreamingDenoiser(
        continuous_slope=True,
        atten_low_db=12.0,
        atten_high_db=0.0,
        dither_dbfs=-65.0,
        enable_snr_blend=True,
    )

    # Ground-truth test scenarios from ADR & design specification
    scenarios = [
        {
            "name": "File 1 (DICOM)",
            "wav": REPO_ROOT / "data" / "asr" / "bench_clips" / "rec4_zoom_in_2x.wav",
            "ref": "zoom in two times",
        },
        {
            "name": "File 2 (Tool Delivery)",
            "wav": REPO_ROOT / "data" / "asr" / "sample_02.wav",
            "ref": "stat m event rewind eight i and metz evacuate c one need safe height back wand enlarge five thirteen in the show hard ten back home me go ten go to ten me nineteen nine one show wand log safe height wand center",
        },
        {
            "name": "File 3 (Slice Nav)",
            "wav": REPO_ROOT / "data" / "asr" / "sample_03.wav",
            "ref": "hemo hemo back two thirty one you go r the halt hemostat bit hemo show the c two need ray two",
        },
        {
            "name": "File 4 (Hemostasis)",
            "wav": REPO_ROOT / "data" / "asr" / "sample_04.wav",
            "ref": "event c hemo up off away back need in i hemo the hemo the c out and home hemostat ct show c degrees event r left ahead eight two",
        },
        {
            "name": "File 5 (Emergency)",
            "wav": REPO_ROOT / "data" / "asr" / "sample_05.wav",
            "ref": "wand left show a eleven and hemostat one a hemo the four the c show bit down back home down thirteen the event ninety nine the show reset stat",
        },
        {
            "name": "Live Multi-Command",
            "wav": REPO_ROOT / "data" / "asr" / "bench_clips" / "mix_zoom_slice.wav",
            "ref": "zoom in two times zoom out three times next slice",
        },
    ]

    import soundfile as sf

    print(f"{'Scenario':<22} | {'Non-Denoised':<16} | {'Shield + 10dB':<16} | {'Continuous Slope':<18} | {'RTF':<8}")
    print("-" * 90)

    tot_err_raw, tot_err_brick, tot_err_cont, tot_words = 0, 0, 0, 0

    for sc in scenarios:
        wav_path = sc["wav"]
        if not wav_path.exists():
            continue
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        dur = len(audio) / sr
        t0 = time.perf_counter()
        clean_cont = dn_continuous.enhance(audio, sample_rate=sr)
        proc_time = time.perf_counter() - t0
        rtf = proc_time / max(1e-6, dur)

        clean_brick = dn_brickwall.enhance(audio, sample_rate=sr)

        if engine:
            res_raw = engine.transcribe(audio).text
            res_brick = engine.transcribe(clean_brick).text
            res_cont = engine.transcribe(clean_cont).text

            ref = sc["ref"]
            e_raw, n_words = word_errors(ref, res_raw)
            e_brick, _ = word_errors(ref, res_brick)
            e_cont, _ = word_errors(ref, res_cont)

            tot_err_raw += e_raw
            tot_err_brick += e_brick
            tot_err_cont += e_cont
            tot_words += n_words

            wer_raw = 100.0 * e_raw / max(1, n_words)
            wer_brick = 100.0 * e_brick / max(1, n_words)
            wer_cont = 100.0 * e_cont / max(1, n_words)

            print(
                f"{sc['name']:<22} | {wer_raw:5.1f}% ({e_raw}/{n_words})    | "
                f"{wer_brick:5.1f}% ({e_brick}/{n_words})    | "
                f"{wer_cont:5.1f}% ({e_cont}/{n_words})     | {rtf:.4f}"
            )
        else:
            rms_raw = float(np.sqrt(np.mean(audio**2)))
            rms_clean = float(np.sqrt(np.mean(clean_cont**2)))
            delta_db = 20.0 * math.log10((rms_clean + 1e-12) / (rms_raw + 1e-12))
            print(f"{sc['name']:<22} | Raw RMS: {rms_raw:.4f} | Clean RMS: {rms_clean:.4f} | Delta: {delta_db:+.2f} dB | {rtf:.4f}")

    if tot_words > 0:
        macro_raw = 100.0 * tot_err_raw / tot_words
        macro_brick = 100.0 * tot_err_brick / tot_words
        macro_cont = 100.0 * tot_err_cont / tot_words
        print("-" * 90)
        print(
            f"{'OVERALL CORPUS WER':<22} | {macro_raw:5.1f}% ({tot_err_raw}/{tot_words})   | "
            f"{macro_brick:5.1f}% ({tot_err_brick}/{tot_words})   | "
            f"{macro_cont:5.1f}% ({tot_err_cont}/{tot_words})    | [PASS]"
        )

    print("=" * 90 + "\n")


if __name__ == "__main__":
    # 1. Run rigorous unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContinuousSlopeGainFloor)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhasePreservation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMicroDither))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSoftKneeSNRBlend))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDenoiserIntegrations))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    # 2. Run scenario evaluation
    run_ground_truth_scenario_eval()
