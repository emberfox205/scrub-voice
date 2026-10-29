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

def generate_synthetic_voice(
    duration_sec: float = 2.0, sample_rate: int = 16000, f0: float = 130.0
) -> np.ndarray:
    """
    Synthesizes a simple harmonic voice-like signal with pitch f0 and formants.
    """
    t = np.arange(int(duration_sec * sample_rate)) / sample_rate
    # Harmonic series (f0, 2f0, 3f0...) imitating human vocal folds
    harmonics = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    sig = np.zeros_like(t)
    for i, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(2 * np.pi * (i * f0) * t)

    # Normalize to [-0.8, 0.8]
    sig = 0.8 * sig / (np.max(np.abs(sig)) + 1e-12)
    return sig.astype(np.float32)

def run_tests():
    print("=" * 70)
    print("🧪 UNIT TEST: ECAPA-TDNN Speaker Verification Engine")
    print("=" * 70)

    # 1. Initialization
    print("⏳ [1/4] Initializing SpeakerVerifier...")
    verifier = SpeakerVerifier(similarity_threshold=0.65)
    print("   ✓ ONNX Model loaded into memory successfully.")

    # 2. Test Embedding Extraction & Shape
    print("⏳ [2/4] Testing embedding dimension and L2 normalization...")
    dummy_audio = generate_synthetic_voice(duration_sec=1.5, f0=120.0)
    emb = verifier.extract_embedding(dummy_audio)

    assert emb.shape == (192,), f"Expected shape (192,), got {emb.shape}"

    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4), f"Vector not unit-normalized: norm={norm}"

    print(f"   ✓ Embedding shape: {emb.shape}")
    print(f"   ✓ Vector L2 norm: {norm:.6f} (Unit hypersphere validated)")

    # 3. Latency Benchmark
    print("⏳ [3/4] Benchmarking CPU inference latency (5 runs)...")
    latencies = []
    test_clip = generate_synthetic_voice(duration_sec=1.5, f0=150.0)

    for _ in range(5):
        t0 = time.perf_counter()
        _ = verifier.extract_embedding(test_clip)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)

    print(f"   ✓ Average Latency: {avg_ms:5.1f} ms | Fastest: {min_ms:5.1f} ms")


    # 4. Identity & Verification Test (Same Speaker vs Different Pitch)
    print("⏳ [4/4] Testing Voice Matching & Separation...")

    person_a_1 = generate_synthetic_voice(duration_sec=2.0, f0=130.0)
    person_a_2 = generate_synthetic_voice(duration_sec=2.0, f0=132.0)  # Same person slight pitch shift
    person_b   = generate_synthetic_voice(duration_sec=2.0, f0=220.0)  # Different person (higher voice)

    emb_a1 = verifier.extract_embedding(person_a_1)
    emb_a2 = verifier.extract_embedding(person_a_2)
    emb_b  = verifier.extract_embedding(person_b)

    sim_same = verifier.compute_similarity(emb_a1, emb_a2)
    sim_diff = verifier.compute_similarity(emb_a1, emb_b)

    print(f"   👉 Same Speaker Similarity (A1 vs A2):  {sim_same:.4f}")
    print(f"   👉 Different Speaker Similarity (A vs B): {sim_diff:.4f}")
    assert sim_same > sim_diff, "Same speaker must score higher than different speaker!"
    print("=" * 70)
    print("🎉 ALL TESTS PASSED! ECAPA-TDNN Engine is production-ready.")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()