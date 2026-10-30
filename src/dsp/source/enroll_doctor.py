import argparse
import sys
import time
from pathlib import Path 
from typing import Iterator, Optional, Union
import numpy as np
import scipy.io.wavefile as wavfile

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

    def enroll_from_chunks(
        self,
        chunk_iterator: Iterator[np.ndarray],
        max_wait_chunks: int = 40,
    ) -> np.ndarray:
        """
        Consumes audio chunks from any source (hardware streamer or mock generator),
        accumulates VAD-qualified speech, and produces the enrollment vector.
        Args:
            chunk_iterator: Yields float32 chunks of shape (num_samples,) or (mics, num_samples).
            max_wait_chunks: Watchdog timeout (e.g., 40 chunks * 0.5s = 20s) to prevent infinite loops.
        Returns:
            L2-normalized 192-D float32 numpy array.
        """
        collected_speech = []
        total_speech_samples = 0
        chunks_inspected = 0

        for chunk in chunk_iterator:
            chunks_inspected += 1

        # Only collect chunks with active human speech
        if self.vad.is_speech_active(chunk):
            mono_chunk = np.mean(chunk, )
