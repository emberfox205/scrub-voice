from typing import Optional, List, Union
import numpy as np

class DASBeamformer:
    """
    Delay-and-Sum (DAS) Beamformer with Competitive Spatial Masking.
    Module 3 in the DSP pipeline:
    - Single speaker: Steers constructive interference toward target angle.
    - Multiple speakers: Applies Soft Wiener Masking in the STFT domain
      to eliminate acoustic cross-talk between overlapping talkers.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        radius: float = 0.04625,
        mic_angles_deg: Optional[np.ndarray] = None,
        speed_of_sound: float = 343.0,
        frame_size: int = 512,
        hop_size: int = 256,
    ):
        self.sample_rate = sample_rate
        self.radius = radius
        self.speed_of_sound = speed_of_sound
        self.frame_size = frame_size
        self.hop_size = hop_size

        # MIC_6 at 0° (East), counter-clockwise: [300°, 240°, 180°, 120°, 60°, 0°]
        if mic_angles_deg is None:
            self.mic_angles_deg = np.array([300.0, 240.0, 180.0, 120.0, 60.0, 0.0])
        else:
            self.mic_angles_deg = np.asarray(mic_angles_deg, dtype=np.float64)

        self.num_mics = len(self.mic_angles_deg)
        self.mic_angles_rad = np.deg2rad(self.mic_angles_deg)

        # -------------------------------------------------------------
        # 1. Full-Band Frequency Grid (0 Hz to Nyquist 8000 Hz)
        # 512 frame_size -> 257 frequency bins
        # -------------------------------------------------------------
        self.freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        self.num_freqs = len(self.freqs)

        # 360-degree discrete grid [0°, 1°, ..., 359°]
        self.grid_angles_deg = np.arange(0, 360, 1)
        self.grid_angles_rad = np.deg2rad(self.grid_angles_deg)

        # -------------------------------------------------------------
        # 2. Precompute Full-Band Steering Weights Tensor
        # Shape: (360, num_mics, num_freqs) complex64 
        #
        # tau[theta, m] = (r / c) * cos(theta - phi_m)
        # w_m(f, theta) = (1 / M) * exp(-j * 2 * pi * f * tau)
        # -----------------------------------------------------------------
        tau = (self.radius / self.speed_of_sound) * np.cos(
            self.grid_angles_rad[:, None] - self.mic_angles_rad[None, :]
        )  # Shape: (360, num_mics)

        phase = -2.0 * np.pi * self.freqs[None, None, :] * tau[:, :, None]

        # Complex weights with 1/M normalization
        self.steering_weights_360 = (
            (1.0 / self.num_mics) * np.exp(1j * phase)
        ).astype(np.complex64)

        # -----------------------------------------------------------------
        # 3. Hann Window for STFT analysis and iSTFT overlap-add synthesis
        # -----------------------------------------------------------------
        self.window = np.hanning(self.frame_size).astype(np.float32)

    def wiener_mask(
            self,
            Y_stft: np.ndarray,
            alpha: float = 1.3,
            floor: float = 0.10,
            eps: float = 1e-10,
    ) -> np.ndarray :
        """
        Applies Multi-Beam Soft Wiener Masking to eliminate cross-talk between multiple speakers.
        Args:
            Y_stft: Complex STFT tensor of shape (num_targets, num_freqs, num_frames)
            alpha: Over-subtraction aggressiveness (1.0 to 1.5)
            floor: Safety attenuation floor (0.10 = max -20 dB suppression)
            eps: Epsilon to prevent division by zero
        Returns:
            Masked complex STFT tensor of shape (num_targets, num_freqs, num_frames)
        """
        # Shape: (num_targets, num_freqs, num_frames)
        beam_power = np.abs(Y_stft) ** 2

        # Shape: (1, freqs, frames)
        total_power = np.sum(beam_power, axis=0, keepdims=True)

        # Broadcasted subtraction:
        p_interf = total_power - beam_power

        # ADAPTIVE ALPHA
        alpha_eff = alpha / max(1, num_targets - 1)

        # Soft Wiener Gain for all targets simultaneously
        wiener_gain = beam_power / (beam_power + alpha_eff * p_interf + eps)
        wiener_gain = np.clip(wiener_gain, floor, 1.0).astype(np.float32)

        Y_stft *= wiener_gain

        return Y_stft

    def beamform(
        self, 
        audio_chunk: np.ndarray, 
        target_angles: Union[float, List[float]],
        apply_wiener_mask: bool = True,
        ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Steers beam(s) towards one or more target angles.
        - If target_angles is a float (e.g. 180.0): returns 1D array of shape (num_samples,).
        - If target_angles is a list (e.g. [60.0, 180.0]): returns List[1D array] with
          Soft Wiener cross-talk suppression applied between competing talkers
        """
        # 1. Normalize input to list and remember if input was single float
        is_single = isinstance(target_angles, (int, float))
        if is_single:
            angles_list = [float(target_angles)]
        else:
            angles_list = [float(a) for a in target_angles]
        if not angles_list:
            return []
        
        # 2. Input Validation
        if audio_chunk.ndim != 2 or audio_chunk.shape[0] != self.num_mics:
            raise ValueError(
                f"Expected audio_chunk of shape ({self.num_mics}, num_samples), "
                f"got {audio_chunk.shape}"                
            )       

        num_targets = len(angles_list)
        num_samples = audio_chunk.shape[1]
        num_frames = max(0, 1 + (num_samples - self.frame_size) // self.hop_size)

        if num_frames == 0:
            zeros = np.zeros(num_samples, dtype=np.float32)
            return zeros if is_single else [zeros for _ in range(num_targets)]

        # 3. Steering weights Retrieval for target angle (num_targets, num_mics, num_freqs)
        target_indices = [int(round(ang)) % 360 for ang in angles_list]
        weights = self.steering_weights_360[target_indices]      # Shape: (num_targets, 6, 257)

        # 4. Multi-Beam STFT Tensor: Shape (num_targets, num_freqs, num_frames)
        Y_stft = np.zeros((num_targets, self.num_freqs, num_frames), dtype=np.complex64)   

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            frame = audio_chunk[:, start : start + self.frame_size] * self.window

            # Multi-channel FFT: Shape (num_mics, num_freqs)
            X = np.fft.rfft(frame, axis = 1)

            # Delay-and-Sum across microphones for all targets simultaneously
            # weights: (K, 6, 257), X: (1, 6, 257) -> sum over mic axis (axis=1) -> (K, 257)
            Y_stft[:, :, frame_idx] = np.sum(weights * X[None, :, :], axis=1)

        # -------------------------------------------------------------
        # 5. Multi-Beam Soft Wiener Masking (Cross-Talk Suppression)
        # Automatically activates only when 2 or more targets exist
        # -------------------------------------------------------------
        if num_targets > 1 and apply_wiener_mask:
            Y_stft = self.wiener_mask(Y_stft)

        # -------------------------------------------------------------
        # 6. Overlap-Add (OLA) Synthesis via iSTFT
        # -------------------------------------------------------------
        output_len = (num_frames - 1) * self.hop_size + self.frame_size
        separated_audio = np.zeros((num_targets, output_len), dtype=np.float32)
        window_sum = np.zeros(output_len, dtype=np.float32)

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size

            # iSTFT across all targets: Shape (num_targets, frame_size)
            time_frames = np.fft.irfft(Y_stft[:, :, frame_idx], n=self.frame_size, axis=1)
            separated_audio[:, start : start + self.frame_size] += time_frames * self.window[None, :]
            window_sum[start : start + self.frame_size] += self.window ** 2

        # Safe window normalization
        safe_window_sum = np.maximum(window_sum, 0.1 * np.max(window_sum))
        separated_audio /= safe_window_sum[None, :]

        # 7. Exact length matching
        copy_len = min(num_samples, output_len)
        results = []
        for k in range(num_targets):
            out = np.zeros(num_samples, dtype=np.float32)
            out[:copy_len] = separated_audio[k, :copy_len]
            results.append(out)

        # If user passed a single float, return a single array; else return the list
        return results[0] if is_single else results
 

