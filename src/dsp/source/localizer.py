from typing import Optional, Tuple, List
import numpy as np
from scipy.signal import find_peaks

from .circuit_framing_config import CircularMicrophoneArray, AudioFramingConfig


class SRPLocalizer:
    """
    Steered Response Power with Phase Transform and CLEAN Deflation (SRP-CLEAN).
    Module 2 in the DSP pipeline:
    - Multi-speaker acoustic source localization using dynamic spatial deflation.
    - Iteratively resolves up to M - 1 active speakers without mutual side-lobe masking.
    - Dynamically halts when the residual spatial power field drops below par_threshold.
    """
    def __init__(
        self,
        geometry: Optional[CircularMicrophoneArray] = None,
        framing: Optional[AudioFramingConfig] = None,
        freq_range: Tuple[float, float] = (100.0, 3500.0),
        par_threshold: float = 1.15,
        tolerance_deg: float = 15.0,
        max_sources: Optional[int] = None,
    ):
        self.geometry = geometry or CircularMicrophoneArray()
        self.framing = framing or AudioFramingConfig()

        self.freq_range = freq_range
        self.par_threshold = par_threshold
        self.tolerance_deg = tolerance_deg

        # An array of M microphones has at most M - 1 spatial degrees of freedom
        self.max_sources = max_sources if max_sources is not None else (self.geometry.num_mics - 1)

        # 360-degree search grid (1-degree resolution)
        self.angles_deg = np.arange(0, 360, 1)

        # Precompute STFT frequency bins
        freqs = np.fft.rfftfreq(self.framing.frame_size, 1.0 / self.framing.sample_rate)
        self.loc_mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        self.freqs_loc = freqs[self.loc_mask]

        # Precomputed steering phase tensor: Shape (360, num_mics, num_freqs_loc)
        self.steering_phase = self.geometry.compute_steering_phases(self.angles_deg, self.freqs_loc)

        # Hann window for STFT frames
        self.window = np.hanning(self.framing.frame_size).astype(np.float32)

    def _compute_stft_tensor(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Computes multi-channel STFT tensor for human speech band (100 - 3500 Hz).
        Returns:
            X: Complex tensor of shape (num_frames, num_mics, num_freqs_loc)
        """
        num_samples = audio_chunk.shape[1]
        num_frames = max(0, 1 + (num_samples - self.framing.frame_size) // self.framing.hop_size)
        if num_frames == 0:
            return np.zeros((0, self.geometry.num_mics, len(self.freqs_loc)), dtype=np.complex64)

        X = np.zeros((num_frames, self.geometry.num_mics, len(self.freqs_loc)), dtype=np.complex64)
        for frame_idx in range(num_frames):
            start = frame_idx * self.framing.hop_size
            frame = audio_chunk[:, start : start + self.framing.frame_size] * self.window
            X[frame_idx] = np.fft.rfft(frame, axis=1)[:, self.loc_mask]

        return X

    def _srp_from_stft(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Computes 360-degree steered response power directly from STFT tensor.
        Returns:
            norm_power: 1D array of shape (360,) normalized to [0.0, 1.0]
            par: Peak-to-Average Ratio (PAR)
        """
        num_frames = X.shape[0]
        if num_frames == 0:
            return np.zeros(len(self.angles_deg), dtype=np.float32), 0.0, 0.0

        # PHAT whitening
        X_phat = X / (np.abs(X) + 1e-12)

        # Steer across 360 degrees and accumulate power across all frames
        power_360 = np.zeros(len(self.angles_deg), dtype=np.float32)

        for t in range(num_frames):
            Y = np.sum(self.steering_phase * X_phat[t, None, :, :], axis=1)     # (360, num_freqs_loc)
            power_360 += np.sum(np.abs(Y) ** 2, axis=1)
        
        min_p = np.min(power_360)
        max_p = np.max(power_360)
        mean_p = np.mean(power_360)
        par = max_p / (mean_p + 1e-12)
        norm_power = (power_360 - min_p) / (max_p - min_p + 1e-12)

        return norm_power, float(par), float(max_p)

    def localize(self, audio_chunk: np.ndarray) -> Tuple[List[float], np.ndarray]:
        """
        Localizes active sound sources using pure frequency-domain SRP-CLEAN deflation.
        Args:
            audio_chunk: Multi-channel audio array of shape (num_mics, num_samples)
        Returns:
            detected_angles: List of resolved speaker azimuth angles in degrees
            pass1_power: Initial spatial power map (0° - 359°) for visualization
        """
        if audio_chunk.ndim != 2 or audio_chunk.shape[0] != self.geometry.num_mics:
            raise ValueError(
                f"Expected audio_chunk of shape ({self.geometry.num_mics}, num_samples), "
                f"got {audio_chunk.shape}"
            )

        if audio_chunk.shape[1] < self.framing.frame_size:
            return [], np.zeros(len(self.angles_deg), dtype=np.float32)

        # Compute multi-channel STFT exactly ONCE
        X = self._compute_stft_tensor(audio_chunk)

        detected_angles: List[float] = []
        pass1_power: Optional[np.ndarray] = None
        pass1_max = 0.0

        # Dynamic Spatial Deflation Loop
        for iteration in range(self.max_sources):
            norm_power, par, max_p = self._srp_from_stft(X)

            if iteration == 0:
                pass1_power = norm_power.copy()
                pass1_max = max_p

                # Stop if residual field is flat (diffuse noise / reverberant echo)
                if par < self.par_threshold:
                    break
            else:
                # Stop if residual power drops below 25% of Pass 1 or PAR is flat
                if max_p < 0.25 * pass1_max or par < self.par_threshold + 0.15:
                    break

            # Find the dominant direction in the current sound field
            dom_idx = int(np.argmax(norm_power))
            dom_angle = float(self.angles_deg[dom_idx])

            # Collision Check: If the strongest signal falls into an already
            #    detected speaker (< tolerance_deg), all genuine sources are resolved!
            if any(min(abs(dom_angle - a), 360.0 - abs(dom_angle - a)) < self.tolerance_deg for a in detected_angles):
                break

            if pass1_power is not None and pass1_power[dom_idx] < 0.35:
                break

            detected_angles.append(round(dom_angle, 1))

            # Spatial Nulling in STFT domain (Pure SRP-CLEAN)
            steer_dom = self.steering_phase[dom_idx]
            beam = np.mean(steer_dom[None, :, :] * X, axis=1)
            X = X - beam[:, None, :] * np.conj(steer_dom)[None, :, :]

        return detected_angles, pass1_power if pass1_power is not None else np.zeros(len(self.angles_deg), dtype=np.float32)
    


            



        