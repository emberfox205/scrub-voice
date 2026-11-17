"""
Unified Array Geometry and STFT Audio Framing

Acts as the Single Source of Truth for:
1. Physical Array Geometry: Radius, microphone angles, speed of sound (CircularMicrophoneArray).
2. Time-Frequency Representation: Sampling rate, STFT frame size, hop size (AudioFramingConfig).
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

@dataclass(frozen=True)
class AudioFramingConfig:
    """
    Time-Frequency STFT framing contract across all DSP modules.
    Guarantees that localizer and beamformer share the exact same frequency grid.
    """
    sample_rate: int = 16000
    frame_size: int = 512   # 32 ms window (257 STFT frequency bins)
    hop_size: int = 256     # 16 ms step (50% overlap, COLA compliant)

class CircularMicrophoneArray:
    """
    Physical circular microphone array geometry (Default: ReSpeaker 6-Mic).
    Computes theoretical acoustic propagation delays and steering phase tensors.
    """
    def __init__(
        self,
        radius: float = 0.04625,  # 46.25 mm radius (92.5 mm diameter)
        mic_angles_deg: Optional[np.ndarray] = None,
        speed_of_sound: float = 343.0,
    ):
        self.radius = float(radius)
        self.speed_of_sound = float(speed_of_sound)

        # Default ReSpeaker 6-Mic: [300°, 240°, 180°, 120°, 60°, 0°]
        if mic_angles_deg is None:
            self.mic_angles_deg = np.array([300.0, 240.0, 180.0, 120.0, 60.0, 0.0], dtype=np.float64)
        else:
            self.mic_angles_deg = np.asarray(mic_angles_deg, dtype=np.float64)

        self.num_mics = len(self.mic_angles_deg)
        self.mic_angles_rad = np.deg2rad(self.mic_angles_deg)

        # Precompute Cartesian (x, y) coordinates on the circle (useful for plotting/simulation)
        self.x = self.radius * np.cos(self.mic_angles_rad)
        self.y = self.radius * np.sin(self.mic_angles_rad)

    def compute_delays(self, angles_deg: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes plane-wave propagation time delays tau (in seconds):
        Formula: tau[theta, m] = (radius / speed_of_sound) * cos(theta - phi_m)
        """
        ang_rad = np.deg2rad(np.atleast_1d(angles_deg))
        tau = (self.radius / self.speed_of_sound) * np.cos(
            ang_rad[:, None] - self.mic_angles_rad[None, :]
        )
        return tau.squeeze() if np.ndim(angles_deg) == 0 else tau

    def compute_steering_phases(self, angles_deg: Union[float, np.ndarray], freqs: np.ndarray) -> np.ndarray:
        """
        Computes the steering phase tensor: exp(-j * 2 * pi * f * tau).
        Shape: (num_angles, num_mics, num_freqs)
        """
        tau = np.atleast_2d(self.compute_delays(angles_deg))
        phase = -2.0 * np.pi * freqs[None, None, :] * tau[:, :, None]
        steering = np.exp(1j * phase).astype(np.complex64)

        return steering.squeeze(axis=0) if np.ndim(angles_deg) == 0 else steering
    


