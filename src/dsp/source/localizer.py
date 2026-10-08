from typing import Optional, Tuple, List
import numpy as np
from scipy.signal import find_peaks

class SRPLocalizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        radius: float = 0.04625,
        mic_angles_deg: Optional[np.ndarray] = None,
        speed_of_sound: float = 343.0,
        frame_size: int = 512,
        hop_size: int = 256,
        freq_range: Tuple[float, float] = (100.0, 3500.0),
        par_threshold: float = 1.25,
    ):
        self.sample_rate = sample_rate
        self.radius = radius
        self.speed_of_sound = speed_of_sound
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.freq_range = freq_range
        self.par_threshold = par_threshold

        # Default: 6 microphones in circle [0°, 60°, 120°, 180°, 240°, 300°]
        if mic_angles_deg is None:
            self.mic_angles_deg = np.array([300.0, 240.0, 180.0, 120.0, 60.0, 0.0])

        else:
            self.mic_angles_deg = np.asarray(mic_angles_deg, dtype=np.float64)

        self.num_mics = len(self.mic_angles_deg)
        self.mic_angles_rad = np.deg2rad(self.mic_angles_deg)

        # 360-degree search grid (1-degree resolution)
        self.angles_deg = np.arange(0, 360, 1)
        self.angles_rad = np.deg2rad(self.angles_deg)

        # Precompute STFT frequency bins
        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
        self.loc_mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        self.freqs_loc = freqs[self.loc_mask]

        # -------------------------------------------------------------
        # Precompute Steering Phase Tensor: Shape (360, num_mics, num_freqs)
        # tau[angle, mic] = (radius / c) * cos(angle - mic_angle)
        # -------------------------------------------------------------
        tau = (self.radius / self.speed_of_sound) * np.cos(
            self.angles_rad[:, None] - self.mic_angles_rad[None, :]
        )   

        # steering_phase[angle, mic, freq] = exp(-j * 2 * pi * f * tau)
        self.steering_phase = np.exp(
            -1j * 2 * np.pi * self.freqs_loc[None, None, :] * tau[:, :, None]
        ).astype(np.complex64)   

        # Hann window for STFT frames
        self.window = np.hanning(self.frame_size).astype(np.float32)

    def localize(self, audio_chunk: np.ndarray) -> Tuple[List[float], np.ndarray]:
        if audio_chunk.ndim != 2 or audio_chunk.shape[0] != self.num_mics:
            raise ValueError(
                f"Expected audio_chunk of shape ({self.num_mics}, num_samples), "
                f"got {audio_chunk.shape}"
            )

        num_samples = audio_chunk.shape[1]
        num_frames = max(0, 1 + (num_samples - self.frame_size) // self.hop_size)

        if num_frames == 0:
            return [], np.zeros(len(self.angles_deg), dtype=np.float32)
        
        # Accumulator for 360-degree steered power
        power_360 = np.zeros(len(self.angles_deg), dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Multi-frame STFT + PHAT Beamforming
        # -------------------------------------------------------------
        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            frame = audio_chunk[:, start : start + self.frame_size] * self.window

            # FFT -> human speech band (100 - 3500 Hz)
            X = np.fft.rfft(frame, axis=1)[:, self.loc_mask]

            # PHAT whitening
            X_phat = X / (np.abs(X) + 1e-12)

            # Steered response across 360 degrees
            Y = np.sum(self.steering_phase * X_phat[None, :, :], axis = 1)

            # Accumulate power
            power_360 += np.sum(np.abs(Y) ** 2, axis = 1)

        # -------------------------------------------------------------
        # 2. Normalize spatial power curve to [0.0, 1.0]
        # -------------------------------------------------------------
        min_p = np.min(power_360)
        max_p = np.max(power_360)
        mean_p = np.mean(power_360)
        par = max_p / (mean_p + 1e-12)
        norm_power = (power_360 - min_p) / (max_p - min_p + 1e-12)

        # If spatial energy is flat (diffuse noise / reverberant echo), abort Gate 2
        if par < self.par_threshold:
            return [], norm_power

        # -------------------------------------------------------------
        # 3. Circular Padding (30°) so 0°/360° boundary is not truncated
        # -------------------------------------------------------------
        pad_deg = 30
        power_padded = np.concatenate([norm_power[-pad_deg:], norm_power, norm_power[:pad_deg]])

        # -------------------------------------------------------------
        # 4. Multi-Peak Detection (Gate 2 Check)
        # distance=25°: minimum angular separation between two distinct speakers
        # prominence=0.15: peak must rise at least 15% above surrounding noise
        # height=0.25: peak must be above 25% of maximum energy
        # -------------------------------------------------------------  
        peaks_padded, _ = find_peaks(
            power_padded, 
            distance=15,
            prominence=0.15,
            height=0.25
        )

        # Map padded peak indices back to [0, 360] degrees and deduplicate
        detected_angles: List[float] = []
        for p in peaks_padded:
            ang = float((p - pad_deg) % 360)

            # Avoid duplicate wrap-around detections near 0/360
            if not any(min(abs(ang - a), 360.0 - abs(ang - a)) < 15.0 for a in detected_angles):
                detected_angles.append(round(ang, 1))

        # Gate 2: returns all detected speaker angles (or [] if pure noise / echo)
        return detected_angles, norm_power
