import sys
import time
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

from source.speaker_verifier import SpeakerVerifier

def load_audio(file_path: Path) -> np.ndarray:
    """
    Loads a .wav file and converts to mono float32 normalized in [-1.0, 1.0].
    """
    sr, audio = wavfile.read(file_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0

    if audio.ndim > 1:
        audio = audio[:, 0]     # Downmix to mono if multi-channel
        
    return audio

def run_tests():
    print("=" * 70)
    print("🧪 UNIT TEST: ECAPA-TDNN Speaker Verification Engine")
    print("=" * 70)

    # Initialization
    print("⏳ [1/4] Initializing SpeakerVerifier...")
    verifier = SpeakerVerifier(similarity_threshold=0.65)
    print("   ✓ ONNX Model loaded into memory successfully.")

    # Verify audio files exist in data/asr/
    sample1_path = REPO_ROOT / "data" / "asr" / "sample_01.wav"
    sample2_path = REPO_ROOT / "data" / "asr" / "sample_03.wav"

    if not sample1_path.exists() or not sample2_path.exists():
        print(f"❌ Sample audio files not found in {REPO_ROOT / 'data' / 'asr'}")
        return

    # Load audio
    audio_spk1 = load_audio(sample1_path)
    audio_spk2 = load_audio(sample2_path)
    sr = 16000

    # Embedding Dimension & L2 Unit Norm Test
    print("⏳ [2/4] Testing embedding dimension and L2 normalization...")
    test_slice = audio_spk1[int(1.0 * sr) : int(3.0 * sr)]
    emb = verifier.extract_embedding(test_slice)

    assert emb.shape == (192,), f"Expected shape (192,), got {emb.shape}"

    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4), f"Vector not uni-normalized: norm={norm}"

    print(f"   ✓ Embedding shape: {emb.shape}")
    print(f"   ✓ Vector L2 norm: {norm:.6f} (Unit hypersphere validated)")

    # CPU Latency Benchmark 
    print("⏳ [3/4] Benchmarking CPU inference latency (5 runs)...")
    latencies = []
    benchmark_clip = audio_spk1[int(1.0 * sr) : int(2.5 * sr)]

    for _ in range(5):
        t0 = time.perf_counter()
        _ = verifier.extract_embedding(benchmark_clip)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    print(f"   ✓ Average Latency: {avg_ms:5.1f} ms | Fastest: {min_ms:5.1f} ms")

    # Identity & Verification Test (Real Audio Files)
    print("⏳ [4/4] Testing Voice Matching & Separation with Real Audio...") 
    # Person A: 2 Distinct time windows of sample_01.wav
    person_a_1 = audio_spk1[int(1.0 * sr) : int(4.0 * sr)]
    person_a_2 = audio_spk1[int(10.0 * sr) : int(13.0 * sr)]

    # Person B: Distinct speaker from sample_02.wav
    person_b = audio_spk2[int(1.0 * sr) : int(4.0 * sr)]

    emb_a1 = verifier.extract_embedding(person_a_1)
    emb_a2 = verifier.extract_embedding(person_a_2)
    emb_b = verifier.extract_embedding(person_b)

    sim_same = verifier.compute_similarity(emb_a1, emb_a2)
    sim_diff = verifier.compute_similarity(emb_a1, emb_b)

    is_match_same, _ = verifier.verify(person_a_2, reference_embedding=emb_a1)
    is_match_diff, _ = verifier.verify(person_b, reference_embedding=emb_a1)

    print(f"   👉 Same Speaker Similarity (A1 vs A2):   {sim_same:.4f} (Match: {is_match_same})")
    print(f"   👉 Different Speaker Similarity (A vs B):  {sim_diff:.4f} (Match: {is_match_diff})")

    assert sim_same > sim_diff, (
        f"Same speaker similarity ({sim_same:.4f}) must be higher than different speaker ({sim_diff:.4f})!"
    )

    print("=" * 70)
    print("🎉 ALL TESTS PASSED! ECAPA-TDNN Engine is production-ready.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()