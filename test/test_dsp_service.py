"""Unit tests for DSP Service and ZeroMQ audio streaming.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import orjson
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dsp.run_dsp import run_dsp_service


class TestDSPService(unittest.TestCase):
    def setUp(self):
        self.ipc_addr = f"ipc:///tmp/test_scrub_audio_{int(time.time()*1000)}.ipc"
        self.context = zmq.Context()
        self.pull = self.context.socket(zmq.PULL)
        self.pull.bind(self.ipc_addr)
        self.sample_wav = REPO_ROOT / "data" / "asr" / "sample_01.wav"

    def tearDown(self):
        self.pull.close(linger=0)
        self.context.term()
        # Clean up ipc file
        path = self.ipc_addr.replace("ipc://", "")
        if Path(path).exists():
            try:
                Path(path).unlink()
            except Exception:
                pass

    def test_dsp_streaming_to_pull_socket(self):
        """Test DSP streaming WAV chunks with VAD & Denoiser over ZeroMQ."""
        received_chunks = []
        received_ends = 0

        # Start DSP in background thread in fast mode
        dsp_thread = threading.Thread(
            target=run_dsp_service,
            kwargs={
                "audio_addr": self.ipc_addr,
                "wav_path": self.sample_wav,
                "use_denoiser": True,
                "use_spatial": False,  # fast test on mono WAV
                "vad_threshold": 0.35,
                "realtime": False,
                "loop": False,
            },
            daemon=True,
        )
        dsp_thread.start()

        # Listen on PULL socket
        start_t = time.monotonic()
        while time.monotonic() - start_t < 15.0:
            if self.pull.poll(timeout=500):
                parts = self.pull.recv_multipart()
                header = orjson.loads(parts[0])
                if header["type"] == "chunk":
                    chunk_bytes = parts[1]
                    samples = np.frombuffer(chunk_bytes, dtype=np.float32)
                    received_chunks.append(samples)
                elif header["type"] == "end":
                    received_ends += 1
                    break
            elif not dsp_thread.is_alive():
                break

        dsp_thread.join(timeout=3.0)

        # Assertions
        self.assertGreater(len(received_chunks), 0, "DSP should have streamed speech chunks")
        self.assertEqual(len(received_chunks[0]), 8000, "Each chunk must be exactly 8000 samples (0.5s at 16kHz)")
        self.assertEqual(received_chunks[0].dtype, np.float32, "Audio chunk must be float32")
        self.assertGreaterEqual(received_ends, 1, "DSP must send at least one 'end' marker")


if __name__ == "__main__":
    unittest.main()

