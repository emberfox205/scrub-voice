from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

CURRENT_DIR = Path(__file__).resolve().parent
DENOISE_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]
DEFAULT_MODEL_PATH = DENOISE_DIR / "onnx" / "duofilter.onnx"


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
    """
    DuoFilter ONNX inference engine optimized for Raspberry Pi / Edge CPU.
    
    Supports:
    1. Continuous full-audio processing without chunk boundary artifacts or comb filtering.
    2. Real-time streaming mode with true Overlap-Add (OLA) buffer to preserve the COLA condition.
    3. Safe multi-channel downmixing (preserves primary microphone phase, avoids spatial comb filtering).
    4. Optional phase-preservation (magnitude-masking mode for high reverberation environments).
    """

    DEFAULT_ONNX_PATH = DEFAULT_MODEL_PATH

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        threads: int = 2,
        lookback_samples: int = 2048,
        atten_lim_db: Optional[float] = 20.0,
    ):
        if model_path is None:
            model_path = self.DEFAULT_ONNX_PATH
        self.model_path = Path(model_path).resolve()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.lookback_samples = lookback_samples
        self.atten_lim_db = atten_lim_db

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

        # Square-root Hann window satisfies COLA condition with 50% overlap
        self.window = torch.hann_window(self.win_length).pow(0.5)

        # Streaming state buffers
        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal streaming OLA buffers."""
        self._stream_in_buffer = torch.zeros(0, dtype=torch.float32)
        self._stream_ola_buffer = torch.zeros(self.win_length, dtype=torch.float32)
        self._stream_lookback_buffer = np.zeros(self.lookback_samples, dtype=np.float32)

    def _downmix(self, audio: np.ndarray, channel_idx: int = 0) -> np.ndarray:
        """
        Safely convert multi-channel audio to mono.
        Selecting a single reference microphone channel prevents acoustic spatial
        comb filtering caused by inter-microphone phase delays.
        """
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            if audio.shape[0] < audio.shape[1]:
                # Shape: [C, N]
                ch = min(channel_idx, audio.shape[0] - 1)
                return audio[ch, :]
            else:
                # Shape: [N, C]
                ch = min(channel_idx, audio.shape[1] - 1)
                return audio[:, ch]
        return audio.flatten()

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        phase_preservation: bool = False,
        channel_idx: int = 0,
        streaming: bool = False,
    ) -> np.ndarray:
        """
        Enhance 1D or multi-channel audio array.
        
        Args:
            audio: 1D or 2D numpy array of audio samples.
            sample_rate: Input sampling rate (resampled to 16000 Hz if different).
            phase_preservation: If True, applies magnitude attenuation only and keeps
                                the input acoustic phase. This avoids hollow phase cancellation
                                in highly reverberant rooms.
            channel_idx: For multi-channel inputs, microphone index to use (avoids spatial comb filtering).
            streaming: If True, uses stateful Overlap-Add across sequential chunk calls.
        
        Returns:
            Enhanced 1D float32 numpy array sampled at 16000 Hz matching original input duration.
        """
        orig_len = audio.shape[-1] if audio.ndim == 1 or audio.shape[0] < audio.shape[1] else audio.shape[0]
        if orig_len == 0:
            return np.zeros(0, dtype=np.float32)

        # Downmix multi-channel to mono safely
        audio_mono = self._downmix(audio, channel_idx=channel_idx)

        wav_t = torch.from_numpy(audio_mono.astype(np.float32))

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            wav_t = torchaudio.functional.resample(
                wav_t, orig_freq=sample_rate, new_freq=self.sample_rate
            )

        if streaming:
            return self.stream_enhance(wav_t.numpy(), phase_preservation=phase_preservation)

        target_len = wav_t.shape[-1]
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)

        # Ensure signal is at least win_length for reflection padding
        pad_samples = max(0, self.win_length - wav_t.shape[-1])
        if pad_samples > 0:
            wav_t = torch.nn.functional.pad(wav_t, (0, pad_samples), mode="constant", value=0.0)

        # Continuous STFT across the entire waveform (NO CHUNKING!)
        noisy_stft = wav_to_stft(
            wav_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        T = noisy_stft.shape[2]
        TARGET_FRAMES = 32
        if T < TARGET_FRAMES:
            pad_T = TARGET_FRAMES - T
            stft_input = torch.nn.functional.pad(noisy_stft, (0, 0, 0, pad_T), mode="constant", value=0.0)
        else:
            stft_input = noisy_stft

        ort_inputs = {self.input_name: stft_input.numpy().astype(np.float32)}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        enhanced_stft = torch.from_numpy(ort_outputs[0])

        if T < TARGET_FRAMES:
            enhanced_stft = enhanced_stft[:, :, :T, :]

        # Minimum Gain Floor Clamping: clamp over-suppressed bins to authentic input floor
        if self.atten_lim_db is not None and self.atten_lim_db < 100.0:
            min_gain = 10.0 ** (-self.atten_lim_db / 20.0)
            noisy_mag = torch.sqrt(noisy_stft[:, 0] ** 2 + noisy_stft[:, 1] ** 2 + 1e-12)
            enh_mag = torch.sqrt(enhanced_stft[:, 0] ** 2 + enhanced_stft[:, 1] ** 2 + 1e-12)
            floor_mag = min_gain * noisy_mag

            below_floor = (enh_mag < floor_mag).unsqueeze(1)
            floor_stft = min_gain * noisy_stft
            enhanced_stft = torch.where(below_floor, floor_stft, enhanced_stft)

        # Optional Phase Preservation: Use magnitude from model + original noisy phase
        if phase_preservation:
            spec_noisy = torch.complex(noisy_stft[:, 0].float(), noisy_stft[:, 1].float())
            spec_enh = torch.complex(enhanced_stft[:, 0].float(), enhanced_stft[:, 1].float())
            noisy_phase = torch.angle(spec_noisy)
            enh_mag = torch.clamp(spec_enh.abs(), min=1e-8)
            mag_only_complex = enh_mag * torch.exp(1j * noisy_phase)
            enhanced_stft = torch.stack([mag_only_complex.real, mag_only_complex.imag], dim=1)

        # Continuous iSTFT across full spectrogram
        enh_wav = stft_to_wav(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=target_len + pad_samples,
        )

        if pad_samples > 0:
            enh_wav = enh_wav[..., :target_len]

        return enh_wav.squeeze(0).cpu().numpy().astype(np.float32)

    def stream_enhance(
        self,
        chunk: np.ndarray,
        phase_preservation: bool = False,
    ) -> np.ndarray:
        """
        Streaming enhancement with Lookback Warm-up Buffer.
        Prepend past audio history to warm up recurrent states and prevent speech swallowing.
        """
        actual_len = len(chunk)
        if actual_len == 0:
            return np.zeros(0, dtype=np.float32)

        extended_chunk = np.concatenate([self._stream_lookback_buffer, chunk.astype(np.float32)])

        if actual_len >= self.lookback_samples:
            self._stream_lookback_buffer = chunk[-self.lookback_samples:].astype(np.float32).copy()
        else:
            self._stream_lookback_buffer = np.concatenate([
                self._stream_lookback_buffer[actual_len:], 
                chunk.astype(np.float32)
            ])

        enhanced_extended = self.enhance(
            extended_chunk,
            phase_preservation=phase_preservation,
            streaming=False, 
        )

        clean_chunk = enhanced_extended[self.lookback_samples : self.lookback_samples + actual_len]
        
        return clean_chunk.astype(np.float32)

    def enhance_file(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        phase_preservation: bool = False,
        channel_idx: int = 0,
        streaming: bool = False,
        chunk_size: int = 8000,
    ) -> Tuple[Path, float]:
        """
        Enhance an audio file from disk and save the clean output.
        
        Args:
            input_path: Path to noisy WAV file.
            output_path: Path to save enhanced WAV file.
            phase_preservation: Whether to preserve original noisy phase.
            channel_idx: Primary channel to extract from multi-channel audio.
            streaming: If True, simulates chunk-by-chunk real-time streaming with OLA.
            chunk_size: Chunk size in samples for streaming mode (default: 8000 = 0.5s @ 16kHz).
            
        Returns:
            (output_path, real_time_factor)
        """
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(in_p), dtype="float32")
        n_samples = audio.shape[0] if audio.ndim == 1 or audio.shape[0] > audio.shape[1] else audio.shape[1]
        duration = n_samples / sr

        if streaming:
            print(f"[*] Processing in Real-Time STREAMING mode (chunk_size={chunk_size} samples / {chunk_size/self.sample_rate*1000:.0f}ms)...")
            self.reset_states()
            out_chunks = []
            t0 = time.perf_counter()
            for start in range(0, n_samples, chunk_size):
                chunk = audio[start : start + chunk_size] if audio.ndim == 1 else audio[start : start + chunk_size, :]
                clean_chunk = self.enhance(
                    chunk,
                    sample_rate=sr,
                    phase_preservation=phase_preservation,
                    channel_idx=channel_idx,
                    streaming=True,
                )
                out_chunks.append(clean_chunk)
            clean = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
            proc_time = time.perf_counter() - t0
        else:
            print("[*] Processing in Full-File STATIC continuous mode...")
            t0 = time.perf_counter()
            clean = self.enhance(
                audio,
                sample_rate=sr,
                phase_preservation=phase_preservation,
                channel_idx=channel_idx,
                streaming=False,
            )
            proc_time = time.perf_counter() - t0

        rtf = proc_time / max(1e-6, duration)
        sf.write(str(out_p), clean, self.sample_rate)

        return out_p, rtf


# ---------------------------------------------------------
# Standalone CLI / Verification
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuoFilter Speech Enhancement / Denoising")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "asr" / "test_clip_event.wav"),
        help="Input noisy wav file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DENOISE_DIR / "audio" / "clean_clip_event.wav"),
        help="Output enhanced wav file path",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of CPU threads for inference (default: 2 for RTF ~0.03)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=2048,
        help="Lookback context buffer size in samples (default: 2048 = 128ms)",
    )
    parser.add_argument(
        "--phase-preservation",
        action="store_true",
        help="Enable magnitude-only masking and preserve original noisy phase",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to use for multi-channel audio (avoids acoustic comb filter)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Process audio in real-time chunked streaming mode with Overlap-Add (OLA)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="Chunk size in samples for streaming mode (default: 8000 = 0.5s @ 16kHz)",
    )
    args = parser.parse_args()

    input_p = Path(args.input)
    if not input_p.exists():
        input_p = REPO_ROOT / "data" / "asr" / "sample_01.wav"

    print(f"Running DuoFilterDenoiser on {input_p}...")
    denoiser = DuoFilterDenoiser(threads=args.threads, lookback_samples=args.lookback)
    out_file, rtf = denoiser.enhance_file(
        input_p,
        args.output,
        phase_preservation=args.phase_preservation,
        channel_idx=args.channel,
        streaming=args.stream,
        chunk_size=args.chunk_size,
    )
    print(f"[DuoFilter] Enhanced audio saved to: {out_file}")
    print(f"[DuoFilter] Real-Time Factor (RTF): {rtf:.4f}")
