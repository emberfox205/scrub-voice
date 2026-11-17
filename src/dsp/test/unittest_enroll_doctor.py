import sys
import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Dynamic path resolution
TEST_DIR = Path(__file__).resolve().parent
DSP_DIR = TEST_DIR.parent
SRC_DIR = DSP_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(DSP_DIR) not in sys.path:
    sys.path.insert(0, str(DSP_DIR))

from source.enroll_doctor import DoctorEnroller
from source.speaker_verifier import SpeakerVerifier

class MockAudioStreamer:
    """
    Simulates hardware AudioStreamer emitting 0.5s chunks.
    Replicates the (6, 8000) multi-channel format of the ReSpeaker array.
    """
    def __init__(self, audio: np.ndarray, chunk_samples: int = 8000, num_mics: int = 6):
        self.audio = audio
        self.chunk_samples = chunk_samples
        self.num_mics = num_mics
        self.cursor = 0

    def read_chunk(self) -> np.ndarray:
        if self.cursor >= len(self.audio):
            # Return silence once audio is exhausted
            return np.zeros((self.num_mics, self.chunk_samples), dtype=np.float32)

        end = self.cursor + self.chunk_samples
        slice_1d = self.audio[self.cursor:end]
        if len(slice_1d) < self.chunk_samples:
            slice_1d = np.pad(slice_1d, (0, self.chunk_samples - len(slice_1d)))
        self.cursor = end

        # Replicate 1D slice across 6 channels: shape (6, 8000)
        return np.tile(slice_1d, (self.num_mics, 1)).astype(np.float32)

def load_wav_normalized(file_path: Path) -> np.ndarray:
    """Loads WAV as float32 [-1.0, 1.0] for the MockAudioStreamer."""
    sr, audio = wavfile.read(file_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio

def run_tests():
    print("=" * 70)
    print("🧪 UNIT TEST: Lead Surgeon Voice Enrollment (DoctorEnroller)")
    print("=" * 70)

    sample1_path = REPO_ROOT / "data" / "asr" / "sample_01.wav"
    sample2_path = REPO_ROOT / "data" / "asr" / "sample_02.wav"

    assert sample1_path.exists(), f"Missing: {sample1_path}"
    assert sample2_path.exists(), f"Missing: {sample2_path}"

    # Loaded once for MockAudioStreamer and verification slices
    audio_spk1 = load_wav_normalized(sample1_path)
    audio_spk2 = load_wav_normalized(sample2_path)

    # -----------------------------------------------------------------
    # Test 1: Live Stream Enrollment (Mock Hardware Streamer)
    # ---------------------------------------------------------------    
    print("⏳ [1/5] Testing enroll_from_stream (simulating 6-mic array)...")
    enroller = DoctorEnroller(target_speech_sec=3.5)
    mock_streamer = MockAudioStreamer(audio_spk1)

    emb_stream = enroller.enroll_from_stream(mock_streamer, max_timeout_sec=10.0)
    assert emb_stream.shape == (192,), f"Expected shape (192,), got {emb_stream.shape}"
    print(f"   ✓ enroll_from_stream produced 192-D embedding: {emb_stream.shape}")

    # -----------------------------------------------------------------
    # Test 2: File-based Enrollment (enroll_from_file)
    # -----------------------------------------------------------------
    print("⏳ [2/5] Testing enroll_from_file...")
    emb_file = enroller.enroll_from_file(sample1_path)
    assert emb_file.shape == (192,), f"Expected shape (192,), got {emb_file.shape}"
    print(f"   ✓ enroll_from_file produced 192-D embedding: {emb_file.shape}")

    # -----------------------------------------------------------------
    # Test 3: Disk Persistence & Round-Trip (save_profile / load_profile)
    # -----------------------------------------------------------------
    print("⏳ [3/5] Testing save_profile & load_profile (.npy)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_profile_path = Path(tmpdir) / "test_doctor_profile.npy"

        # Save profile
        saved_path = DoctorEnroller.save_profile(emb_stream, temp_profile_path)
        assert saved_path.exists(), "Profile file was not written to disk!"

        # Load profile
        loaded_emb = DoctorEnroller.load_profile(saved_path)
        assert np.allclose(emb_stream, loaded_emb, atol=1e-6), "Loaded profile differs from original!"

        # Exception check: missing file
        try:
            DoctorEnroller.load_profile(Path(tmpdir) / "non_existent.npy")
            assert False, "Should have raised FileNotFoundError for missing profile"
        except FileNotFoundError:
            pass
    print("   ✓ save_profile and load_profile round-trip verified.")

    # -----------------------------------------------------------------
    # Test 4: Edge Case — Insufficient Speech (< 1.5s)
    # -----------------------------------------------------------------
    print("⏳ [4/5] Testing rejection on insufficient speech (< 1.5s)...")
    short_audio = audio_spk1[: int(0.5 * 16000)]  # Only 0.5s speech
    short_streamer = MockAudioStreamer(short_audio)

    try:
        enroller.enroll_from_stream(short_streamer, max_timeout_sec=2.0)
        assert False, "Should have raised ValueError for speech under 1.5s"
    except ValueError as e:
        assert "Insufficient speech" in str(e), f"Unexpected error message: {e}"
        print(f"   ✓ Safely rejected short utterance: '{e}'")

    # -----------------------------------------------------------------
    # Test 5: Biometric Verification Against Doctor vs Auxiliary Staff
    # -----------------------------------------------------------------
    print("⏳ [5/5] Testing biometric identity verification...")
    verifier = SpeakerVerifier(similarity_threshold=0.65)

    # 3.0s slice of Doctor (Speaker 1)
    doc_test_slice = audio_spk1[int(6.0 * 16000) : int(9.0 * 16000)]
    # 3.0s slice of Nurse / Auxiliary (Speaker 2)
    other_test_slice = audio_spk2[int(1.0 * 16000) : int(4.0 * 16000)]

    is_match_doc, score_doc = verifier.verify(doc_test_slice, reference_embedding=emb_stream)
    is_match_other, score_other = verifier.verify(other_test_slice, reference_embedding=emb_stream)

    print(f"   👉 Enrolled Doctor Verification:  score={score_doc:0.4f} (Match: {is_match_doc})")
    print(f"   👉 Auxiliary Staff Verification:  score={score_other:0.4f} (Match: {is_match_other})")

    assert is_match_doc, f"Doctor was rejected! Score: {score_doc:.4f}"
    assert not is_match_other, f"Auxiliary staff was falsely accepted! Score: {score_other:.4f}"
    assert score_doc > score_other, "Doctor score must be higher than auxiliary staff score!"

    print("=" * 70)
    print("🎉 ALL 5 ENROLLMENT TESTS PASSED! Module is production-ready.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
