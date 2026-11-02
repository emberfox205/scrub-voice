from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from src.dsp.denoise.continuous_slope_denoiser import (
    ContinuousSlopeGainFloor,
    SoftKneeSNRBlender,
)

CURRENT_DIR = Path(__file__).resolve().parent
DENOISE_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]
DEFAULT_MODEL_PATH = DENOISE_DIR / "onnx" / "duofilter_stateful.onnx"


class DuoFilterStreamingDenoiser:
    """
    Stateful streaming DuoFilter ONNX inference engine configured for 0.5s chunks (8,000 samples).
    Matches the system-wide chunk specification of scrub-voice (AudioStreamer, Silero VAD, DAS Beamformer).
    
    Enhanced with the 4-Pillar Continuous Psychoacoustic Slope & Hybrid Architecture:
      1. Continuous Psychoacoustic Slope (-12 dB -> -6 dB -> 0 dB smooth cosine curve)
      2. Strict Magnitude-Only Phase Preservation (100% Delay-and-Sum beamformed phase)
      3. Micro-Dither Injection (-65 dBFS noise floor to stabilize Kaldi CMVN variance)
      4. Soft-Knee SNR Dynamic Blend (automatic auto-tuning between loud speech and pauses)
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        chunk_samples: int = 8000,  # 0.5s @ 16 kHz
        threads: int = 1,
        atten_lim_db: Optional[float] = None,
        split_band_hz: Optional[float] = None,
        continuous_slope: bool = True,
        f_low: float = 1000.0,
        f_high: float = 3000.0,
        atten_low_db: float = 12.0,
        atten_high_db: float = 0.0,
        dither_dbfs: Optional[float] = -65.0,
        enable_snr_blend: bool = True,
        snr_low_db: float = 8.0,
        snr_high_db: float = 18.0,
        phase_preservation: bool = True,
    ):
        if model_path is None:
            self.model_path = DEFAULT_MODEL_PATH
        else:
            self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Stateful DuoFilter ONNX model not found at {self.model_path}. "
                f"Run export_duofilter_streaming_onnx.py first to generate it."
            )

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.chunk_samples = chunk_samples

        self.continuous_slope = continuous_slope
        self.f_low = f_low
        self.f_high = f_high
        self.atten_low_db = atten_low_db
        self.atten_high_db = atten_high_db
        self.phase_preservation = phase_preservation
        self.enable_snr_blend = enable_snr_blend

        # Legacy parameters support
        self.atten_lim_db = atten_lim_db if atten_lim_db is not None else 20.0
        self.split_band_hz = split_band_hz

        n_bins = self.n_fft // 2 + 1

        if self.continuous_slope:
            # Pillar 1: Precompute Continuous Psychoacoustic Slope curve
            slope_atten_low = self.atten_low_db
            if atten_lim_db is not None and atten_lim_db != 20.0 and atten_low_db == 12.0:
                slope_atten_low = atten_lim_db

            self.gain_floor_builder = ContinuousSlopeGainFloor(
                n_fft=self.n_fft,
                sample_rate=self.sample_rate,
                f_low=self.f_low,
                f_high=self.f_high,
                atten_low_db=slope_atten_low,
                atten_high_db=self.atten_high_db,
            )
            self.gain_floor = self.gain_floor_builder.get_torch_floor((1, 1, n_bins))
            self.split_weights = None
        else:
            self.gain_floor = None
            # Precompute legacy Split-Band Consonant Shield weights
            if self.split_band_hz is not None and 0 < self.split_band_hz < (self.sample_rate / 2):
                bin_hz = self.sample_rate / self.n_fft
                split_bin = int(round(self.split_band_hz / bin_hz))
                transition_bins = 8
                half_trans = max(1, transition_bins // 2)
                start_bin = max(0, split_bin - half_trans)
                end_bin = min(n_bins, split_bin + half_trans)

                weights = torch.ones(n_bins, dtype=torch.float32)
                if start_bin < end_bin:
                    weights[start_bin:end_bin] = torch.linspace(1.0, 0.0, end_bin - start_bin)
                weights[end_bin:] = 0.0
                self.split_weights: Optional[torch.Tensor] = weights.view(1, 1, 1, n_bins)
            else:
                self.split_weights = None

        # Pillar 3: Micro-dither floor parameter
        self.dither_dbfs = dither_dbfs
        self.dither_std: Optional[float] = (
            10.0 ** (dither_dbfs / 20.0) if dither_dbfs is not None else None
        )

        # Pillar 4: Soft-knee SNR blender
        self.snr_blender = SoftKneeSNRBlender(
            snr_low_db=snr_low_db,
            snr_high_db=snr_high_db,
        )

        # Square-root Hann window satisfies COLA condition with 50% overlap
        self.window = torch.hann_window(self.win_length).pow(0.5)

        # Configure ONNX runtime session
        torch.set_num_threads(threads)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        # Cache shapes for streaming convolutional and recurrent states
        self.cache_shapes = {
            "enc0_cache": (1, 9, 2, 128),
            "enc1_cache": (1, 64, 2, 128),
            "enc2_cache": (1, 64, 2, 64),
            "enc3_cache": (1, 64, 2, 32),
            "glu_cache": (1, 64, 2, 16),
            "gru_state": (1, 16, 64),
        }

        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal streaming caches and SNR estimator to zeros."""
        self.states: Dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.cache_shapes.items()
        }
        self.snr_blender.reset()

    def enhance_chunk(
        self,
        chunk: np.ndarray,
        is_speech: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Enhance a single 0.5s audio chunk (8,000 samples @ 16 kHz).
        Directly drop-in compatible with run_dsp.py streaming loop.
        
        Args:
            chunk: 1D float32 numpy array of length 8000.
            is_speech: Optional bool indicator from VAD for adaptive noise tracking.
        Returns:
            clean_chunk: 1D float32 numpy array of length 8000.
        """
        if chunk.ndim > 1:
            chunk = chunk[0, :] if chunk.shape[0] < chunk.shape[1] else chunk[:, 0]

        target_len = len(chunk)
        if target_len == 0:
            return np.zeros(0, dtype=np.float32)

        raw_chunk = chunk.astype(np.float32)
        wav_t = torch.from_numpy(raw_chunk).unsqueeze(0)

        # STFT on the chunk: 8000 samples -> [1, 257, 32]
        spec = torch.stft(
            wav_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        # Shape to ONNX input: [1, 2, T=32, 257]
        stft = torch.stack([spec.real, spec.imag], dim=1).transpose(-1, -2)

        ort_inputs = {"noisy_frame": stft.numpy().astype(np.float32), **self.states}
        outputs = self.session.run(None, ort_inputs)

        enh_stft = outputs[0]  # [1, 2, T=32, 257]
        self.states["enc0_cache"] = outputs[1]
        self.states["enc1_cache"] = outputs[2]
        self.states["enc2_cache"] = outputs[3]
        self.states["enc3_cache"] = outputs[4]
        self.states["glu_cache"] = outputs[5]
        self.states["gru_state"] = outputs[6]

        enh_stft_t = torch.from_numpy(enh_stft)

        if self.continuous_slope and self.gain_floor is not None:
            # Pillar 1 & 2: Continuous Psychoacoustic Slope + Strict Phase Preservation
            noisy_mag = torch.sqrt(stft[:, 0] ** 2 + stft[:, 1] ** 2 + 1e-12)
            enh_mag = torch.sqrt(enh_stft_t[:, 0] ** 2 + enh_stft_t[:, 1] ** 2 + 1e-12)

            floor_mag = self.gain_floor.to(device=enh_mag.device) * noisy_mag
            effective_mag = torch.maximum(enh_mag, floor_mag)

            if self.phase_preservation:
                # Recombine 100% authentic Delay-and-Sum beamformed phase
                unit_real = stft[:, 0] / noisy_mag
                unit_imag = stft[:, 1] / noisy_mag
                out_real = effective_mag * unit_real
                out_imag = effective_mag * unit_imag
                enh_stft_t = torch.stack([out_real, out_imag], dim=1)
            else:
                scale = effective_mag / (enh_mag + 1e-12)
                enh_stft_t = enh_stft_t * scale.unsqueeze(1)
        else:
            # Legacy branch: Minimum Gain Floor Clamping + Split-Band Consonant Shield
            if self.atten_lim_db is not None and self.atten_lim_db < 100.0:
                min_gain = 10.0 ** (-self.atten_lim_db / 20.0)
                noisy_mag = torch.sqrt(stft[:, 0] ** 2 + stft[:, 1] ** 2 + 1e-12)
                enh_mag = torch.sqrt(enh_stft_t[:, 0] ** 2 + enh_stft_t[:, 1] ** 2 + 1e-12)
                floor_mag = min_gain * noisy_mag

                below_floor = (enh_mag < floor_mag).unsqueeze(1)
                floor_stft = min_gain * stft
                enh_stft_t = torch.where(below_floor, floor_stft, enh_stft_t)

            if self.split_weights is not None:
                enh_stft_t = self.split_weights * enh_stft_t + (1.0 - self.split_weights) * stft

        # iSTFT synthesis
        enh_complex = torch.complex(
            enh_stft_t[:, 0, :, :],
            enh_stft_t[:, 1, :, :],
        ).transpose(-1, -2)  # [1, 257, T=32]

        enh_wav = torch.istft(
            enh_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=target_len,
        )

        clean_out = enh_wav.squeeze(0).cpu().numpy().astype(np.float32)

        # Pillar 3: Micro-Dither Injection (-65 dBFS floor)
        if self.dither_std is not None and self.dither_std > 0:
            dither = np.random.normal(0.0, self.dither_std, size=clean_out.shape).astype(np.float32)
            clean_out = clean_out + dither

        # Pillar 4: Soft-Knee SNR Dynamic Blend
        if self.enable_snr_blend:
            clean_out, alpha, snr_db = self.snr_blender.blend(
                denoised=clean_out,
                beamformed=raw_chunk,
                is_speech=is_speech,
            )

        return clean_out.astype(np.float32)

    def enhance(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Enhance 1D audio array by streaming in 0.5s (8,000 sample) chunks.
        """
        if audio.ndim > 1:
            audio = audio[0, :] if audio.shape[0] < audio.shape[1] else audio[:, 0]

        total_samples = len(audio)
        if sample_rate is not None and sample_rate != self.sample_rate:
            import torchaudio

            wav_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            resampled = torchaudio.functional.resample(
                wav_t, orig_freq=sample_rate, new_freq=self.sample_rate
            )
            audio = resampled.squeeze(0).numpy()
            total_samples = len(audio)

        self.reset_states()

        clean_chunks = []
        for start in range(0, total_samples, self.chunk_samples):
            end = min(start + self.chunk_samples, total_samples)
            chunk = audio[start:end]
            clean_chunk = self.enhance_chunk(chunk)
            clean_chunks.append(clean_chunk)

        if not clean_chunks:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(clean_chunks)

    def enhance_file(
        self, input_path: Union[Path, str], output_path: Union[Path, str]
    ) -> Tuple[Path, float]:
        """
        Enhance a WAV file from disk in 0.5s streaming chunks and compute RTF.
        """
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(in_p), dtype="float32")
        duration = (audio.shape[0] if audio.ndim == 1 else audio.shape[1]) / sr

        t0 = time.perf_counter()
        clean = self.enhance(audio, sample_rate=sr)
        proc_time = time.perf_counter() - t0

        rtf = proc_time / max(1e-6, duration)
        sf.write(str(out_p), clean, self.sample_rate)

        return out_p, rtf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stateful 0.5s Chunk DuoFilter with Continuous Psychoacoustic Slope & Hybrid Denoiser"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "asr" / "sample_02.wav"),
        help="Input noisy wav file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DENOISE_DIR / "audio" / "clean_duofilter_chunk_05s.wav"),
        help="Output enhanced wav file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to duofilter_stateful.onnx",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=1000.0,
        help="Low frequency boundary in Hz for deep cleaning (default: 1000 Hz)",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=3000.0,
        help="High frequency boundary in Hz for unmolested pass-through (default: 3000 Hz)",
    )
    parser.add_argument(
        "--atten-low-db",
        type=float,
        default=12.0,
        help="Attenuation limit below f_low in dB (default: 12.0 dB)",
    )
    parser.add_argument(
        "--dither-dbfs",
        type=float,
        default=-65.0,
        help="Micro-dither floor in dBFS (default: -65.0 dBFS)",
    )
    parser.add_argument(
        "--no-continuous-slope",
        dest="continuous_slope",
        action="store_false",
        help="Disable continuous slope and revert to legacy split-band filter",
    )
    parser.add_argument(
        "--no-snr-blend",
        dest="snr_blend",
        action="store_false",
        help="Disable soft-knee SNR dynamic blending",
    )
    parser.set_defaults(continuous_slope=True, snr_blend=True)
    args = parser.parse_args()

    denoiser = DuoFilterStreamingDenoiser(
        model_path=args.model,
        continuous_slope=args.continuous_slope,
        f_low=args.f_low,
        f_high=args.f_high,
        atten_low_db=args.atten_low_db,
        dither_dbfs=args.dither_dbfs,
        enable_snr_blend=args.snr_blend,
    )
    out_file, rtf = denoiser.enhance_file(args.input, args.output)
    print(f"[Continuous Slope DuoFilter] Enhanced audio saved to: {out_file}")
    print(f"[Continuous Slope DuoFilter] Real-Time Factor (RTF): {rtf:.4f}")
