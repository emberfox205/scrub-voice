from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torch import Tensor

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]


# ---------------------------------------------------------
# STFT & iSTFT Helper Functions (Square-Root Hann COLA)
# ---------------------------------------------------------
def wav_to_stft(
    wav: Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    window: Optional[Tensor] = None,
) -> Tensor:
    """
    Transforms audio waveform to STFT representation.
    Returns: Tensor of shape [B, 2, T, 257] (Real, Imag)
    """
    if window is None:
        window = torch.hann_window(win_length, device=wav.device).pow(0.5)

    # Complex STFT: [B, F=257, T]
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    # Transpose to [B, T, F] then stack real and imag -> [B, 2, T, F]
    spec = spec.transpose(-1, -2)
    return torch.stack([spec.real, spec.imag], dim=1)


def stft_to_wav(
    enhanced_stft: Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    window: Optional[Tensor] = None,
    length: Optional[int] = None,
) -> Tensor:
    """
    Reconstructs audio waveform from enhanced STFT [B, 2, T, 257].
    Returns: Tensor of shape [B, Audio_Samples]
    """
    if window is None:
        window = torch.hann_window(win_length, device=enhanced_stft.device).pow(0.5)

    # Reconstruct complex spectrogram: [B, T, F] -> [B, F, T]
    enh_complex = torch.complex(enhanced_stft[:, 0].float(), enhanced_stft[:, 1].float())
    spec_transposed = enh_complex.transpose(-1, -2).contiguous()

    return torch.istft(
        spec_transposed,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.float(),
        center=True,
        length=length,
    )


# ---------------------------------------------------------
# DuoFilter Denoiser Class
# ---------------------------------------------------------
class DuoFilterDenoiser:
    """DuoFilter ONNX inference engine optimized for Raspberry Pi / Edge CPU."""

    DEFAULT_ONNX_PATH = CURRENT_DIR / "onnx" / "duofilter.onnx"

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
            model_path = self.DEFAULT_ONNX_PATH
        self.model_path = Path(model_path).resolve()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if not self.model_path.exists():
            raise FileNotFoundError(f"DuoFilter ONNX model not found at {self.model_path}")

        torch.set_num_threads(threads)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.window = torch.hann_window(self.win_length).pow(0.5)

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance mono audio (1D float32 or [C, T]).
        Returns 1D float32 np.ndarray matching the input audio length.
        """
        orig_len = audio.shape[-1]
        wav_t = torch.from_numpy(audio.astype(np.float32))
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        elif wav_t.shape[0] > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)

        noisy_stft = wav_to_stft(
            wav_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        ort_inputs = {self.input_name: noisy_stft.numpy().astype(np.float32)}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        enhanced_stft = torch.from_numpy(ort_outputs[0])

        enh_wav = stft_to_wav(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=orig_len,
        )
        return enh_wav.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------
# Standalone CLI / Verification
# ---------------------------------------------------------
if __name__ == "__main__":
    input_wav = CURRENT_DIR / "audio" / "noise_audio.wav"
    output_wav = CURRENT_DIR / "audio" / "clean_audio.wav"
    if not input_wav.exists():
        input_wav = REPO_ROOT / "data" / "asr" / "sample_01.wav"

    print(f"Running DuoFilterDenoiser on {input_wav}...")
    denoiser = DuoFilterDenoiser()
    noisy_wav, sr = torchaudio.load(str(input_wav))
    if sr != denoiser.sample_rate:
        noisy_wav = torchaudio.functional.resample(noisy_wav, orig_freq=sr, new_freq=denoiser.sample_rate)

    enhanced = denoiser.enhance(noisy_wav.squeeze(0).numpy())
    torchaudio.save(str(output_wav), torch.from_numpy(enhanced).unsqueeze(0), denoiser.sample_rate)
    print(f"Saved enhanced audio to {output_wav}")
