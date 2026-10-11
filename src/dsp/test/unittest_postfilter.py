"""
Level 1: Unit Tests for DeepFilterPostFilter (Module 4)
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Dynamic path resolution to find 'source'
dsp_root = Path(__file__).resolve().parent.parent
if str(dsp_root) not in sys.path:
    sys.path.insert(0, str(dsp_root))

from source.postfilter import DeepFilterPostFilter


@pytest.fixture(scope="module")
def pf():
    # Module-scoped fixture so the neural network loads only once for all tests
    return DeepFilterPostFilter()


# ============================================================
# 1. Input Validation
# ============================================================
def test_wrong_ndim_raises(pf):
    """Passing 2D multichannel audio instead of 1D mono must raise ValueError."""
    with pytest.raises(ValueError):
        pf.enhance(np.zeros((6, 8000), dtype=np.float32))

def test_empty_input_returns_empty(pf):
    """Passing empty array should return empty array without crashing."""
    out = pf.enhance(np.zeros(0, dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert len(out) == 0

# ============================================================
# 2. Output Contract (Shape, Dtype, Finite Values)
# ============================================================
def test_output_shape_and_dtype(pf):
    # Standard 0.5s chunk (8000 samples) must return exactly (8000,) float 32
    chunk = (np.random.randn(8000) * 0.1).astype(np.float32)
    out = pf.enhance(chunk)

    assert isinstance(out, np.ndarray)
    assert out.shape == (8000,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

def test_arbitrary_chunk_length(pf):
    """Must maintain exact length for any chunk size (e.g. 4000 samples)."""
    chunk = (np.random.randn(4000) * 0.1).astype(np.float32)
    out = pf.enhance(chunk)
    assert out.shape == (4000,)


# ============================================================
# 3. Streaming Continuity (Multiple Consecutive Chunks)
# ============================================================
def test_streaming_continuity(pf):
    """
    Feeds 5 consecutive chunks through the recurrent state.
    Ensures internal state updates smoothly without NaN corruption or memory leaks.
    """
    for _ in range(5):
        chunk = (np.random.randn(8000) * 0.05).astype(np.float32)
        out = pf.enhance(chunk)
        assert out.shape == (8000,)
        assert not np.isnan(out).any()


# ============================================================
# 4. Multi-Stream Interface (enhance_multi)
# ============================================================
def test_enhance_multi_empty(pf):
    out = pf.enhance_multi([])
    assert out == []

def test_enhance_multi_multiple_streams(pf):
    stream1 = (np.random.randn(8000) * 0.1).astype(np.float32)
    stream2 = (np.random.randn(8000) * 0.1).astype(np.float32)

    enhanced = pf.enhance_multi([stream1, stream2])
    assert len(enhanced) == 2
    assert enhanced[0].shape == (8000,)
    assert enhanced[1].shape == (8000,)
    assert enhanced[0].dtype == np.float32
    assert enhanced[1].dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])