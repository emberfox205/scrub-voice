from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousSlopeGainFloor:
    """
    Computes and manages the Frequency-Dependent Minimum Gain Floor G_min[k].
    
    Replaces abrupt split-band brickwall shelf filters with a gradual 3-stage
    psychoacoustic curve matching human equal-loudness contours:
      - 0 to f_low (default 1000 Hz): atten_low_db (default -12 dB, rumble removal)
      - f_low to f_high (default 1000 - 3000 Hz): Smooth cosine transition (-12 dB -> 0 dB)
      - Above f_high (default > 3000 Hz): 0 dB (100% consonant pass-through)
    
    Mathematical Formulation:
      G_low  = 10^(-atten_low_db / 20)
      G_high = 10^(-atten_high_db / 20)
      G_min[k] = G_low for k < k_low
      G_min[k] = G_low + 0.5 * (G_high - G_low) * (1 - cos(pi * (k - k_low) / (k_high - k_low))) for k_low <= k <= k_high
      G_min[k] = G_high for k > k_high
    """

    def __init__(
        self,
        n_fft: int = 512,
        sample_rate: int = 16000,
        f_low: float = 1000.0,
        f_high: float = 3000.0,
        atten_low_db: float = 12.0,
        atten_high_db: float = 0.0,
    ):
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.f_low = f_low
        self.f_high = f_high
        self.atten_low_db = atten_low_db
        self.atten_high_db = atten_high_db

        self.n_bins = n_fft // 2 + 1
        self.bin_hz = sample_rate / n_fft

        self.k_low = int(round(f_low / self.bin_hz))
        self.k_high = int(round(f_high / self.bin_hz))
        self.k_low = max(0, min(self.k_low, self.n_bins - 1))
        self.k_high = max(self.k_low + 1, min(self.k_high, self.n_bins))

        self.g_low = 10.0 ** (-abs(atten_low_db) / 20.0)
        self.g_high = 10.0 ** (-abs(atten_high_db) / 20.0)

        curve = np.ones(self.n_bins, dtype=np.float32) * self.g_high
        curve[: self.k_low] = self.g_low

        if self.k_high > self.k_low:
            t = np.linspace(0.0, np.pi, self.k_high - self.k_low, endpoint=False)
            curve[self.k_low : self.k_high] = (
                self.g_low + (self.g_high - self.g_low) * 0.5 * (1.0 - np.cos(t))
            )
        curve[self.k_high :] = self.g_high

        self.curve_np = curve
        self.curve_torch = torch.from_numpy(curve).float()

    def get_torch_floor(self, view_shape: Tuple[int, ...] = (1, 1, 1, -1)) -> torch.Tensor:
        """Returns the precomputed gain floor as a PyTorch tensor reshaped for broadcasting."""
        return self.curve_torch.view(view_shape)

    def get_numpy_floor(self) -> np.ndarray:
        """Returns the precomputed gain floor as a NumPy 1D array."""
        return self.curve_np.copy()


class SoftKneeSNRBlender:
    """
    Pillar 4: Soft-Knee SNR Dynamic Blend.
    
    Dynamically blends enhanced (denoised) and authentic beamformed audio
    based on the frame / chunk instantaneous Signal-to-Noise Ratio (SNR):
      y(t) = alpha * x_denoised(t) + (1 - alpha) * x_beamformed(t)
    
    Behavior:
      - Loud speech peaks (> snr_high_db, e.g. 18 dB): alpha -> 0 (100% beamformed pass-through)
      - Pauses / low SNR (< snr_low_db, e.g. 8 dB): alpha -> 1 (100% denoiser suppression)
      - Transition zone (8 to 18 dB): Smooth C1 cosine soft-knee
    """

    def __init__(
        self,
        snr_low_db: float = 8.0,
        snr_high_db: float = 18.0,
        noise_floor_alpha: float = 0.95,
        min_noise_power: float = 1e-8,
    ):
        self.snr_low_db = snr_low_db
        self.snr_high_db = snr_high_db
        self.noise_floor_alpha = noise_floor_alpha
        self.min_noise_power = min_noise_power

        # Recursive tracking of background noise floor power
        self.noise_power_est: float = min_noise_power

    def reset(self) -> None:
        """Reset internal noise tracker to default baseline."""
        self.noise_power_est = self.min_noise_power

    def update_noise_estimate(self, chunk: Union[np.ndarray, torch.Tensor], is_speech: Optional[bool] = None) -> float:
        """
        Updates background noise power estimate during low-energy or non-speech frames.
        """
        if isinstance(chunk, torch.Tensor):
            chunk_power = float(torch.mean(chunk**2).item())
        else:
            chunk_power = float(np.mean(chunk**2))

        chunk_power = max(chunk_power, self.min_noise_power)

        # If speech status is known or power is very low, update noise tracker
        if is_speech is False or chunk_power < self.noise_power_est * 1.5:
            self.noise_power_est = (
                self.noise_floor_alpha * self.noise_power_est
                + (1.0 - self.noise_floor_alpha) * chunk_power
            )
        elif chunk_power < self.noise_power_est:
            self.noise_power_est = 0.5 * (self.noise_power_est + chunk_power)

        self.noise_power_est = max(self.noise_power_est, self.min_noise_power)
        return self.noise_power_est

    def compute_snr_db(self, chunk: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Computes instantaneous SNR (in dB) relative to the estimated noise floor.
        """
        if isinstance(chunk, torch.Tensor):
            sig_power = float(torch.mean(chunk**2).item())
        else:
            sig_power = float(np.mean(chunk**2))

        sig_power = max(sig_power, 1e-12)
        noise_power = max(self.noise_power_est, 1e-12)

        snr_linear = sig_power / noise_power
        snr_db = 10.0 * math.log10(max(snr_linear, 1e-6))
        return snr_db

    def compute_alpha(self, snr_db: float) -> float:
        """
        Computes the soft-knee blending factor alpha in [0.0, 1.0].
        """
        if snr_db <= self.snr_low_db:
            return 1.0
        if snr_db >= self.snr_high_db:
            return 0.0

        t = (snr_db - self.snr_low_db) / (self.snr_high_db - self.snr_low_db)
        alpha = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(np.clip(alpha, 0.0, 1.0))

    def blend(
        self,
        denoised: Union[np.ndarray, torch.Tensor],
        beamformed: Union[np.ndarray, torch.Tensor],
        snr_db: Optional[float] = None,
        is_speech: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
        """
        Blends denoised and beamformed audio chunks.
        
        Returns:
            blended_audio, alpha, snr_db
        """
        self.update_noise_estimate(beamformed, is_speech=is_speech)
        if snr_db is None:
            snr_db = self.compute_snr_db(beamformed)

        alpha = self.compute_alpha(snr_db)

        if isinstance(denoised, torch.Tensor) and isinstance(beamformed, torch.Tensor):
            blended = alpha * denoised + (1.0 - alpha) * beamformed
        else:
            blended = alpha * np.asarray(denoised) + (1.0 - alpha) * np.asarray(beamformed)

        return blended, alpha, snr_db


class ContinuousSlopeDenoiser(nn.Module):
    """
    Continuous Psychoacoustic Slope & Hybrid Denoiser PyTorch Module.
    
    Implements all 4 Pillars:
      1. Continuous Psychoacoustic Slope (-12 dB -> -6 dB -> 0 dB monotonic cosine transition)
      2. Strict Magnitude-Only Phase Preservation (preserves 100% of beamformed spatial phase)
      3. Micro-Dither Injection (-65 dBFS noise floor to stabilize Kaldi CMVN variance)
      4. Soft-Knee SNR Dynamic Blending (intelligently streams pure beamforming during loud speech)
    """

    def __init__(
        self,
        n_fft: int = 512,
        sample_rate: int = 16000,
        f_low: float = 1000.0,
        f_high: float = 3000.0,
        atten_low_db: float = 12.0,
        atten_high_db: float = 0.0,
        dither_dbfs: Optional[float] = -65.0,
        enable_snr_blend: bool = True,
        snr_low_db: float = 8.0,
        snr_high_db: float = 18.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.f_low = f_low
        self.f_high = f_high
        self.atten_low_db = atten_low_db
        self.atten_high_db = atten_high_db
        self.dither_dbfs = dither_dbfs
        self.enable_snr_blend = enable_snr_blend

        self.gain_floor_builder = ContinuousSlopeGainFloor(
            n_fft=n_fft,
            sample_rate=sample_rate,
            f_low=f_low,
            f_high=f_high,
            atten_low_db=atten_low_db,
            atten_high_db=atten_high_db,
        )

        n_bins = n_fft // 2 + 1
        floor_tensor = self.gain_floor_builder.get_torch_floor((1, 1, 1, n_bins))
        self.register_buffer("gain_floor", floor_tensor)

        self.dither_std: Optional[float] = (
            10.0 ** (dither_dbfs / 20.0) if dither_dbfs is not None else None
        )

        self.snr_blender = SoftKneeSNRBlender(
            snr_low_db=snr_low_db,
            snr_high_db=snr_high_db,
        )

    def forward(
        self,
        neural_gain_or_stft: torch.Tensor,
        stft_complex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply continuous slope gain floor and strict magnitude-only phase preservation.
        
        Args:
            neural_gain_or_stft: Raw neural gain [B, 1, T, F] OR enhanced complex STFT [B, 2, T, F]
            stft_complex: Raw Delay-and-Sum beamformed STFT [B, 2, T, F] (Real, Imag)
            
        Returns:
            enhanced_stft: Recombined complex STFT [B, 2, T, F] preserving 100% of beamformed phase.
        """
        noisy_real = stft_complex[:, 0:1]
        noisy_imag = stft_complex[:, 1:2]
        noisy_mag = torch.sqrt(noisy_real**2 + noisy_imag**2 + 1e-12)

        if neural_gain_or_stft.shape[1] == 1:
            neural_gain = neural_gain_or_stft
        elif neural_gain_or_stft.shape[1] == 2:
            enh_real = neural_gain_or_stft[:, 0:1]
            enh_imag = neural_gain_or_stft[:, 1:2]
            enh_mag = torch.sqrt(enh_real**2 + enh_imag**2 + 1e-12)
            neural_gain = enh_mag / noisy_mag
        else:
            raise ValueError(
                f"Unexpected channel dimension {neural_gain_or_stft.shape[1]}. Expected 1 (gain) or 2 (complex)."
            )

        # Pillar 1: Clamp suppression to the continuous psychoacoustic gain floor
        floor_aligned = self.gain_floor.to(device=neural_gain.device, dtype=neural_gain.dtype)
        if floor_aligned.shape[-1] != neural_gain.shape[-1]:
            # Accommodate [B, C, F, T] vs [B, C, T, F]
            if floor_aligned.shape[-1] == neural_gain.shape[-2]:
                floor_aligned = floor_aligned.transpose(-1, -2)

        effective_gain = torch.maximum(neural_gain, floor_aligned)
        enhanced_mag = effective_gain * noisy_mag

        # Pillar 2: Phase-preserving reconstruction (100% original beamformed phase angle)
        unit_real = noisy_real / noisy_mag
        unit_imag = noisy_imag / noisy_mag
        out_real = enhanced_mag * unit_real
        out_imag = enhanced_mag * unit_imag

        return torch.cat([out_real, out_imag], dim=1)

    def inject_dither(self, wav: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Pillar 3: Micro-dither injection (-65 dBFS floor).
        Prevents zero-energy filterbanks in Kaldi/Vosk CMVN.
        """
        if self.dither_std is None or self.dither_std <= 0:
            return wav

        if isinstance(wav, torch.Tensor):
            noise = torch.randn_like(wav) * self.dither_std
            return wav + noise
        else:
            noise = np.random.normal(0.0, self.dither_std, size=wav.shape).astype(np.float32)
            return wav + noise

    def blend(
        self,
        denoised_wav: Union[np.ndarray, torch.Tensor],
        beamformed_wav: Union[np.ndarray, torch.Tensor],
        is_speech: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
        """
        Pillar 4: Soft-knee SNR dynamic blend wrapper.
        """
        if not self.enable_snr_blend:
            return denoised_wav, 1.0, 0.0
        return self.snr_blender.blend(denoised_wav, beamformed_wav, is_speech=is_speech)
