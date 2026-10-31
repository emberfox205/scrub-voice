import subprocess
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import soundfile as sf

# Streaming Module: Captures raw audio from the Seeed 8-mic array via arecord
# or simulates streaming from a WAV file for edge testing.
# strips the 2 AEC loopback channels (6, 7) when using hardware
# float32 numpy chunks of shape (6, 8000)

class AudioStreamer:
    """
    Output per read_chunk():
      shape  : (6, 8000)  — (mics, samples)
      dtype  : float32
      range  : [-1.0, 1.0]
    """
    BYTES_PER_SAMPLE = 2
    HARDWARE_CHANNELS = 8
    ACTIVE_MICS = 6

    def __init__(
            self,
            device: str = "plughw:seeed8micvoicec,0",
            sample_rate: int = 16000,
            chunk_duration: float = 0.5,
            wav_path: Optional[Union[str, Path]] = None,
            realtime: bool = True,
            loop: bool = False,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.wav_path = Path(wav_path) if wav_path else None
        self.realtime = realtime
        self.loop = loop

        self.samples_per_chunk = int(sample_rate * chunk_duration)
        self.bytes_per_chunk = self.samples_per_chunk * self.HARDWARE_CHANNELS * self.BYTES_PER_SAMPLE

        self._process: subprocess.Popen | None = None
        self._wav_data: np.ndarray | None = None
        self._wav_pos: int = 0
        self._last_read_time: float | None = None


    # ================
    # Lifecycle
    # ================

    def start(self) -> None:
        if self.wav_path is not None:
            if not self.wav_path.exists():
                raise FileNotFoundError(f"Simulation WAV file not found: {self.wav_path}")
            data, sr = sf.read(str(self.wav_path), dtype="float32")
            if sr != self.sample_rate:
                import torchaudio
                import torch
                t = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
                resampled = torchaudio.functional.resample(t, orig_freq=sr, new_freq=self.sample_rate)
                data = resampled.numpy().T if data.ndim > 1 else resampled.squeeze(0).numpy()

            # Ensure shape is (ACTIVE_MICS, total_samples)
            if data.ndim == 1:
                # Tile mono across 6 mics
                self._wav_data = np.tile(data, (self.ACTIVE_MICS, 1))
            elif data.shape[1] >= self.ACTIVE_MICS:
                self._wav_data = data[:, : self.ACTIVE_MICS].T
            else:
                self._wav_data = np.tile(data.mean(axis=1), (self.ACTIVE_MICS, 1))

            self._wav_pos = 0
            self._last_read_time = None
            return

        # Spawn the arecord process and start streaming raw PCM
        if self._process is not None:
            return      # arecord already running

        cmd = [
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", str(self.HARDWARE_CHANNELS),
            "-t", "raw",
            "--buffer-size=65536",         # Power-of-two ALSA DMA ring buffer prevents XRUNs
            "--period-size=8192",          # Power-of-two period aligned with ~0.5s DSP chunks
            "--quiet"
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        """Terminate the arecord process cleanly without driver deadlocks."""
        self._wav_data = None
        self._wav_pos = 0
        if self._process is not None:
            try:
                # 1. Close stdout pipe first so arecord knows nothing is listening
                if self._process.stdout is not None:
                    try:
                        self._process.stdout.close()
                    except Exception:
                        pass

                # 2. Send SIGINT (Ctrl+C signal) instead of SIGTERM.
                # arecord handles SIGINT natively by closing the ALSA PCM device cleanly!
                import signal
                self._process.send_signal(signal.SIGINT)

                # 3. Wait briefly, then force-kill if driver refuses to yield
                try:
                    self._process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=0.5)
            except Exception as e:
                pass
            finally:
                self._process = None

    # ============
    # Reading
    # ============

    def read_chunk(self) -> np.ndarray:
        # Read one 0.5s chunk of audio
        # returns: np.ndarray, shape (6, 8000), dtype float32
        # 1 row per mic, values in normalized range [-1.0, 1.0]

        if self.wav_path is not None:
            if self._wav_data is None:
                raise RuntimeError("AudioStreamer is not running. Call start() first!")

            if self._wav_pos >= self._wav_data.shape[1]:
                if self.loop:
                    self._wav_pos = 0
                else:
                    raise EOFError("Simulation WAV reached end of file.")

            end_pos = self._wav_pos + self.samples_per_chunk
            if end_pos <= self._wav_data.shape[1]:
                chunk = self._wav_data[:, self._wav_pos : end_pos]
                self._wav_pos = end_pos
            else:
                # Zero-pad remaining
                chunk = np.zeros((self.ACTIVE_MICS, self.samples_per_chunk), dtype=np.float32)
                avail = self._wav_data.shape[1] - self._wav_pos
                chunk[:, :avail] = self._wav_data[:, self._wav_pos :]
                self._wav_pos = self._wav_data.shape[1]

            if self.realtime:
                now = time.monotonic()
                if self._last_read_time is not None:
                    elapsed = now - self._last_read_time
                    sleep_time = self.chunk_duration - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                self._last_read_time = time.monotonic()

            return chunk.astype(np.float32)

        """
        Raises
        ------
        RuntimeError
            If the streamer has not been started or arecord exits early.
        EOFError
            If arecord's stdout closes unexpectedly (process died).
        """
        if self._process is None:
            raise RuntimeError("AudioStreamer is not running. Call start() first!")

        parts = []
        remaining = self.bytes_per_chunk

        while remaining > 0:
            part = self._process.stdout.read(remaining)
            if not part:
                raise EOFError("arecord stdout closed unexpectedly."
                               "Check that the device is connected and the driver is loaded.")
            parts.append(part)
            remaining -= len(part)

        raw = b"".join(parts)

        # Parse and reshape
        # interleaved S16_LE -> (samples, 8 channels)
        pcm = np.frombuffer(raw, dtype=np.int16)
        pcm = pcm.reshape(-1, self.HARDWARE_CHANNELS)

        # Drop AEC loopback channels(6, 7), keep mics 0-5
        mics = pcm[:, : self.ACTIVE_MICS]

        # Transpose to (6, samples) and normalize to [-1.0, 1.0]
        audio = mics.T.astype(np.float32) / 32768.0
        return audio

    # =================
    # Context Manager
    # =================

    def __enter__(self) -> "AudioStreamer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
