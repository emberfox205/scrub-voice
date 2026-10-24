from typing import Optional
import numpy as np
import torch
from scipy.signal import resample_poly

from df.enhance import init_df, enhance

class DeepFilterPostFilter:
    """
    Module 4: Neural Spectral Post-Filter (DeepFilterNet3).
    Takes the beamformed mono audio stream, strips comb-filtering notches,
    reverberation tails, and background noise, restoring natural human voice
    before downstream speaker verification (ECAPA-TDNN) and ASR.
    """

    def __init__(
        self,
        sample_rate: int = 16000      
    ):
        self.sample_rate = sample_rate
        self.target_rate = 48000    # DeepFilterNet3 native model rate

        # Resampling ratio: 16,000 * 3 = 48,000 Hz
        # (Exact integer factor of 3 -> zero phase distortion)     
        self.up_factor = 3
        self.down_factor = 1

        # -------------------------------------------------------------
        # Load DeepFilterNet3 Model & Streaming State
        # -------------------------------------------------------------
        # model: Neural network (ERG conv layers + Deep Filtering GRU)
        # df_state: Internal state buffer (maintains continuity across chunks)
        self.model, self.df_state, _ = init_df()
        self.model.eval()

    def enhance(self, audio_mono: np.ndarray) -> np.ndarray:
        """
        Enhances a 1D mono audio chunk from the beamformer.
        Args:
            audio_mono: 1D numpy array of shape (num_samples,) float32 at 16 kHz
        Returns:
            enhanced_audio: 1D numpy array of shape (num_samples,) float32 at 16 kHz
        """
        # 1. Input Validataion
        if audio_mono.ndim != 1:
            raise ValueError(f"Expected 1D mono audio array, got shape {audio_mono.shape}")

        num_samples = len(audio_mono)
        if num_samples == 0:
            return np.zeros(0, dtype=np.float32)
        # -------------------------------------------------------------
        # 2. High-Speed Polyphase Up-sampling (16 kHz -> 48 kHz)
        # Ratio 3:1 (e.g., 8000 samples -> 24,000 samples)
        # -------------------------------------------------------------
        audio_48k = resample_poly(audio_mono, self.up_factor, self.down_factor).astype(np.float32)

        # -------------------------------------------------------------
        # 3. Format as PyTorch Tensor: Shape (1, num_samples_48k)
        # DeepFilterNet expects (batch/channels, time)
        # -------------------------------------------------------------
        tensor_in = torch.from_numpy(audio_48k).unsqueeze(0)

        # -------------------------------------------------------------
        # 4. Neural Deep Filtering Inference
        # Strips comb-filtering notches, room reflections, and noise
        # -------------------------------------------------------------
        with torch.no_grad():
            tensor_out = enhance(self.model, self.df_state, tensor_in)

        enhanced_48k = tensor_out.squeeze(0).cpu().numpy()

        # -------------------------------------------------------------
        # 5. Polyphase Down-sampling (48 kHz -> 16 kHz)
        # Ratio 1:3 (e.g., 24,000 samples -> 8000 samples)
        # -------------------------------------------------------------
        enhanced_16k = resample_poly(enhanced_48k, self.down_factor, self.up_factor).astype(np.float32)

        # -------------------------------------------------------------
        # 6. Ensure Exact Length Matching
        # Prevents frame accumulation drift over long streaming sessions
        # -------------------------------------------------------------
        result = np.zeros(num_samples, dtype=np.float32)
        copy_len = min(num_samples, len(enhanced_16k))
        result[:copy_len] = enhanced_16k[:copy_len]

        return result

    def enhance_multi(self, audio_streams: list[np.ndarray]) -> list[np.ndarray]:
        # Enhances multiple separated audio streams
        return [self.enhance(stream) for stream in audio_streams]
    