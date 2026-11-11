from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

CURRENT_DIR = Path(__file__).resolve().parent
DENOISE_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]
DEFAULT_MODEL_PATH = DENOISE_DIR / "onnx" / "gtcrn_simple.onnx"
FALLBACK_MODEL_PATH = DENOISE_DIR / "onnx" / "gtcrn_simple.onnx"


class GTCRNDenoiser:
    """
    GTCRN (Grouped Temporal Convolutional Recurrent Network) ONNX inference engine.
    Designed for real-time streaming speech enhancement and noise suppression.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        threads: int = 1,
    ):
        if model_path is None:
            if DEFAULT_MODEL_PATH.exists():
                self.model_path = DEFAULT_MODEL_PATH
            elif FALLBACK_MODEL_PATH.exists():
                self.model_path = FALLBACK_MODEL_PATH
            else:
                self.model_path = DEFAULT_MODEL_PATH
        else:
            self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"GTCRN ONNX model not found at {self.model_path}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

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

        # Cache shapes for streaming recurrent state
        self.conv_cache_shape = (2, 1, 16, 16, 33)
        self.tra_cache_shape = (2, 3, 1, 1, 16)
        self.inter_cache_shape = (2, 1, 33, 16)

        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal streaming caches to zeros."""
        self.conv_cache = np.zeros(self.conv_cache_shape, dtype=np.float32)
        self.tra_cache = np.zeros(self.tra_cache_shape, dtype=np.float32)
        self.inter_cache = np.zeros(self.inter_cache_shape, dtype=np.float32)

    def process_frame(self, stft_frame: np.ndarray) -> np.ndarray:
        """
        Process a single STFT complex frame.
        Input: stft_frame of shape [1, 257, 1, 2] (Real, Imag)
        Returns: enhanced frame of shape [1, 257, 1, 2]
        """
        out, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
            {
                "mix": stft_frame,
                "conv_cache": self.conv_cache,
                "tra_cache": self.tra_cache,
                "inter_cache": self.inter_cache,
            },
        )
        return out

    def enhance(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Enhance a 1D or multi-channel audio array.
        Returns: enhanced 1D float32 numpy array sampled at 16000 Hz.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        orig_len = audio.shape[-1]
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        # Downmix multi-channel to mono
        if audio.ndim == 2:
            if audio.shape[0] < audio.shape[1]:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio.copy()

        wav_t = torch.from_numpy(audio_mono.astype(np.float32))

        # Resample to 16 kHz if necessary
        if sample_rate != self.sample_rate:
            wav_t = torchaudio.functional.resample(
                wav_t, orig_freq=sample_rate, new_freq=self.sample_rate
            )

        target_len = wav_t.shape[-1]
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)

        # STFT analysis: [1, n_fft//2 + 1, T] complex
        spec_complex = torch.stft(
            wav_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        # Convert to real/imag representation: [1, 257, T, 2]
        spec_real = torch.view_as_real(spec_complex).numpy().astype(np.float32)
        T = spec_real.shape[2]

        self.reset_states()
        enh_frames = []

        # Stream frame-by-frame through GTCRN
        for t in range(T):
            mix_frame = spec_real[:, :, t : t + 1, :]  # [1, 257, 1, 2]
            enh_frame = self.process_frame(mix_frame)
            enh_frames.append(enh_frame)

        enh_stft = np.concatenate(enh_frames, axis=2)  # [1, 257, T, 2]

        # iSTFT synthesis: [1, 257, T] complex
        enh_complex = torch.complex(
            torch.from_numpy(enh_stft[:, :, :, 0]),
            torch.from_numpy(enh_stft[:, :, :, 1]),
        )

        enh_wav = torch.istft(
            enh_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=target_len,
        )

        return enh_wav.squeeze(0).cpu().numpy().astype(np.float32)

    def enhance_file(
        self, input_path: Union[Path, str], output_path: Union[Path, str]
    ) -> Tuple[Path, float]:
        """
        Enhance an audio file from disk and save the clean output.
        Returns: (output_path, real_time_factor)
        """
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(in_p), dtype="float32")
        n_samples = audio.shape[0] if audio.ndim == 1 or audio.shape[0] > audio.shape[1] else audio.shape[1]
        duration = n_samples / sr

        t0 = time.perf_counter()
        clean = self.enhance(audio, sample_rate=sr)
        proc_time = time.perf_counter() - t0

        rtf = proc_time / max(1e-6, duration)
        sf.write(str(out_p), clean, self.sample_rate)

        return out_p, rtf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GTCRN Speech Enhancement / Denoising")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "asr" / "sample_02.wav"),
        help="Input noisy wav file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DENOISE_DIR / "audio" / "clean_gtcrn_sample_2.wav"),
        help="Output enhanced wav file path",
    )
    args = parser.parse_args()

    denoiser = GTCRNDenoiser()
    out_file, rtf = denoiser.enhance_file(args.input, args.output)
    print(f"[GTCRN] Enhanced audio saved to: {out_file}")
    print(f"[GTCRN] Real-Time Factor (RTF): {rtf:.4f}")
