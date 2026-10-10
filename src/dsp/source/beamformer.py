from typing import Optional, List, Tuple
import numpy as np

class DASBeamformer:
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

    def beamform(self, audio_chunk: np.ndarray, target_angle_deg: float) -> np.ndarray:
        """
        Steers a single beam towards target_angle_deg and synthesizes back to time domain.
        Args:
            audio_chunk: Multi-channel audio array of shape (num_mics, num_samples)
            target_angle_deg: Steered azimuth angle in degrees [0, 360)
        Returns:
            enhanced_audio: 1D mono float32 array of shape (num_samples,)
        """
        # 1. Input Validation
        if audio_chunk.ndim != 2 or audio_chunk.shape[0] != self.num_mics:
            raise ValueError(
                f"Expected audio_chunk of shape ({self.num_mics}, num_samples), "
                f"got {audio_chunk.shape}"                
            )       

        num_samples = audio_chunk.shape[1]
        num_frames = max(0, 1 + (num_samples - self.frame_size) // self.hop_size)

        if num_frames == 0:
            return np.zeros(num_samples, dtype=np.float32)

        # 2. Steering weights Retrieval for target angle
        angle_idx = int(round(target_angle_deg)) % 360
        weights = self.steering_weights_360[angle_idx]      # Shape: (6, 257)

        # Buffers for Overlap-Add (OLA) Synthesis
        output_len = (num_frames - 1) * self.hop_size + self.frame_size
        enhanced = np.zeros(output_len, dtype=np.float32)
        window_sum = np.zeros(output_len, dtype=np.float32)

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            frame = audio_chunk[:, start : start + self.frame_size] * self.window

            # Multi-channel STFT: shape (num_mics, num_freqs)
            X = np.fft.rfft(frame, axis=1)

            # Delay-and-Sum across microphones (axis 0)
            Y = np.sum(weights * X, axis=0)     # Shape: (num_freqs,)

            # Inverse Real FFT back to time domain
            time_frame = np.fft.irfft(Y, n=self.frame_size)

            # Overlap-add synthesis
            enhanced[start : start + self.frame_size] += time_frame * self.window
            window_sum[start : start + self.frame_size] += self.window ** 2

        # Overlap-add window normalization
        #safe_window_sum = np.maximum(window_sum, 1e-4)
        safe_window_sum = np.maximum(window_sum, 0.1 * np.max(window_sum))
        enhanced /= safe_window_sum

        # Pad / trim output to match exact input chunk length
        result = np.zeros(num_samples, dtype=np.float32)
        copy_len = min(num_samples, output_len)
        result[:copy_len] = enhanced[:copy_len]

        return result

    def beamform_multi(self, audio_chunk: np.ndarray, target_angles_deg: List[float]) -> List[np.ndarray]:
        """
        Steers multiple simultaneous beams toward all angles detected by Gate 2.
        Args:
            audio_chunk: Multi-channel audio array of shape (num_mics, num_samples)
            target_angles_deg: List of detected azimuth angles in degrees [0, 360)
        Returns:
            List of 1D mono float32 arrays, one per target angle.
        """
        if not target_angles_deg:
            return []

        return [self.beamform(audio_chunk, angle) for angle in target_angles_deg]

