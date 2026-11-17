import sys
import time
from pathlib import Path 
from typing import Optional, Union
import numpy as np
import scipy.io.wavfile as wavfile

# Dynamic path resolution 
CURRENT_DIR = Path(__file__).resolve().parent
DSP_DIR = CURRENT_DIR.parent
SRC_DIR = DSP_DIR.parent
REPO_ROOT = SRC_DIR.parent 

if str(DSP_DIR) not in sys.path:
    sys.path.insert(0, str(DSP_DIR))

# Import existing DSP gate modules
from source.speaker_verifier import SpeakerVerifier
from source.vad import SileroVAD

# Default destination for the doctor's biometric vector
DEFAULT_PROFILE_PATH = DSP_DIR / "models" / "ecapa_tdnn" / "doctor_profile.npy"

class DoctorEnroller:
    """
    Lead Surgeon Voice Biometrics Enroller.
    Collects speech chunks, filters out non-speech using VAD,
    extracts a 192-D ECAPA-TDNN embedding, and persists it to disk.
    """
    def __init__(
        self,
        verifier: Optional[SpeakerVerifier] = None,
        vad: Optional[SileroVAD] = None,
        sample_rate: int = 16000,
        target_speech_sec: float = 3.5,
    ):
        self.sample_rate = sample_rate
        self.target_speech_sec = target_speech_sec
        self.target_samples = int(target_speech_sec * sample_rate)
        self.min_valid_samples = int (1.5 * sample_rate)    # Floor: 1.5s
        self.verifier = verifier if verifier is not None else SpeakerVerifier(sample_rate=sample_rate)
        self.vad = vad if vad is not None else SileroVAD(sample_rate=sample_rate)

    def _extract_from_chunks(self, speech_chunks: list[np.ndarray]) -> np.ndarray:
        """Validates duration, concatenates speech, and extracts 192-D embedding."""
        total_samples = sum(len(c) for c in speech_chunks)
        if total_samples < self.min_valid_samples:
            raise ValueError(
                f"Insufficient speech collected: {total_samples / self.sample_rate:.2f}s. "
                f"Need at least 1.5s of active speech for valid biometric enrollment."
            )
        full_audio = np.concatenate(speech_chunks)
        return self.verifier.extract_embedding(full_audio)

    def enroll_from_stream(
        self,
        streamer,
        max_timeout_sec: float = 20.0
    ) -> np.ndarray:
        """
        Enrolls using an ALREADY-RUNNING AudioStreamer.
        Does NOT open or close ALSA — leaves hardware stream intact.
        """
        collected_speech = []
        total_samples = 0
        max_chunks = int (max_timeout_sec / 0.5)

        print("Enroller listening to active stream. Ask the Doctor to speak...")

        for _ in range(max_chunks):
            chunk = streamer.read_chunk()

            if self.vad.is_speech_active(chunk):
                # Mic 0 reference channel (avoids phase cancellation)
                mono = chunk[0] if chunk.ndim > 1 else chunk
                collected_speech.append(mono.astype(np.float32))
                total_samples += len(mono)

                progress = min(100, int((total_samples / self.target_samples) * 100))
                sec = total_samples / self.sample_rate
                print(f"Enrolling: {sec:.1f}s / {self.target_speech_sec:.1f}s [{progress:3d}%]")

                if total_samples >= self.target_samples:
                    break

        return self._extract_from_chunks(collected_speech)

    def enroll_from_file(
        self,
        wav_path: Union[str, Path]
    ) -> np.ndarray:
        """Offline/testing enrollment from a WAV file."""
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        sr, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        if audio.ndim > 1:
            audio = audio[:, 0]

        chunk_size = int(self.sample_rate * 0.5)
        collected_speech = []
        total_samples = 0

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) < chunk_size:
                break

            if self.vad.is_speech_active(chunk):
                collected_speech.append(chunk)
                total_samples += len(chunk)
                if total_samples >= self.target_samples:
                    break

        return self._extract_from_chunks(collected_speech)

    @staticmethod
    def save_profile(
        embedding: np.ndarray,
        path: Union[str, Path] = DEFAULT_PROFILE_PATH
    ) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, embedding.astype(np.float32))
        print(f"Saved profile: {dest.resolve()}")
        return dest

    @staticmethod
    def load_profile(path: Union[str, Path] = DEFAULT_PROFILE_PATH) -> np.ndarray:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Doctor profile not found: {src}")

        emb = np.load(src).astype(np.float32)
        if emb.shape != (192,):
            raise ValueError(f"Corrupted profile: expected (192,), got {emb.shape}")

        return emb
    

        

        