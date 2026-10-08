import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock, patch 

dsp_root = Path(__file__).resolve().parent.parent
if str(dsp_root) not in sys.path:
    sys.path.insert(0, str(dsp_root))

try:
    from source.audio_streamer import AudioStreamer
except ImportError:
    from src.dsp.source.audio_streamer import AudioStreamer

def test_streamer_initialization():
    streamer = AudioStreamer(sample_rate=16000, chunk_duration=0.5)
    assert streamer.samples_per_chunk == 8000
    # 8000 samples * 8 channels * 2 bytes = 128 000 bytes
    assert streamer.bytes_per_chunk == 128000

def test_read_chunk_shape_and_normalization():
    streamer = AudioStreamer(sample_rate=16000, chunk_duration=0.5)

    # Create simulated raw 8-channel int 16 PCM data
    # Shape: (8000 samples, 8 channels)
    num_samples = 8000
    fake_samples = np.random.randint(-30000, 30000, size=(num_samples, 8), dtype=np.int16)
    fake_bytes = fake_samples.tobytes()

    # Mock subprocess.Popen stdout to return this fake byte stream
    mock_process = MagicMock()
    mock_process.stdout.read.side_effect = [fake_bytes]
    streamer._process = mock_process

    # Read chunk
    audio = streamer.read_chunk()

    # Verify specification
    assert audio.shape == (6, 8000), f"Expected shape (6, 8000), got {audio.shape}"
    assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"
    assert -1.0 <= audio.min() and audio.max() <= 1.0, "Values must be normalized to [-1.0, 1.0]"

def test_read_chunk_not_started_raises_error():
    streamer = AudioStreamer()
    # Should raise RuntimeError if read_chunk() called before start()
    with pytest.raises(RuntimeError):
        streamer.read_chunk()