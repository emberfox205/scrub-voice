from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

CURRENT_DIR = Path(__file__).resolve().parent
DENOISE_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]
DEFAULT_MODEL_PATH = DENOISE_DIR / "onnx" / "duofilter_stateful.onnx"


class DuoFilterStreamingDenoiser:
    """
    Stateful streaming DuoFilter ONNX inference engine configured for 0.5s chunks (8,000 samples).
    Matches the system-wide chunk specification of scrub-voice (AudioStreamer, Silero VAD, DAS Beamformer).
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
        """Reset internal streaming caches to zeros."""
        self.states: Dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.cache_shapes.items()
        }

    def enhance_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Enhance a single 0.5s audio chunk (8,000 samples @ 16 kHz).
        Directly drop-in compatible with run_dsp.py streaming loop.
        
        Args:
            chunk: 1D float32 numpy array of length 8000.
        Returns:
            clean_chunk: 1D float32 numpy array of length 8000.
        """
        if chunk.ndim > 1:
            chunk = chunk[0, :] if chunk.shape[0] < chunk.shape[1] else chunk[:, 0]

        target_len = len(chunk)
        if target_len == 0:
            return np.zeros(0, dtype=np.float32)

        wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)

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

        # iSTFT synthesis
        enh_complex = torch.complex(
            torch.from_numpy(enh_stft[:, 0, :, :]),
            torch.from_numpy(enh_stft[:, 1, :, :]),
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

        return enh_wav.squeeze(0).cpu().numpy().astype(np.float32)

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
            resampled = torchaudio.functional.resample(wav_t, orig_freq=sample_rate, new_freq=self.sample_rate)
            audio = resampled.squeeze(0).numpy()
            total_samples = len(audio)

        self.reset_states()

        clean_chunks = []
        for start in range(0, total_samples, self.chunk_samples):
            end = min(start + self.chunk_samples, total_samples)
            chunk = audio[start:end]

            # If tail chunk is shorter, it processes with dynamic ONNX axes seamlessly
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
    parser = argparse.ArgumentParser(description="Stateful 0.5s Chunk DuoFilter Speech Enhancement")
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
    args = parser.parse_args()

    denoiser = DuoFilterStreamingDenoiser(model_path=args.model)
    out_file, rtf = denoiser.enhance_file(args.input, args.output)
    print(f"[0.5s Chunk DuoFilter] Enhanced audio saved to: {out_file}")
    print(f"[0.5s Chunk DuoFilter] Real-Time Factor (RTF): {rtf:.4f}")
