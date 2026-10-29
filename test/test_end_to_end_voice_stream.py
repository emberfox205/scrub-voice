"""End-to-End Voice Pipeline Stream Integration Test.

Tests the complete physical and logical pipeline:
[WAV File] -> [DSP: AudioStreamer + Silero VAD + DuoFilter]
           -> [ASR: UtteranceStream + Vosk Engine]
           -> [NLU: VoiceToIntentPipeline + SetFit + Normalizer]
           -> [Orchestrator: CommandDispatcher LVC Cache]
           -> [Monitor: FastAPI + WebSockets Live Broadcast]
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx
import orjson
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestrator.supervisor import Supervisor

TEST_PORT = 8933
TEST_WAV = REPO_ROOT / "data" / "asr" / "test_clip_event.wav"


class TestEndToEndVoiceStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_WAV.exists():
            raise FileNotFoundError(f"Test clip not found: {TEST_WAV}")

        # Clean existing IPC sockets
        for s in Path("/tmp").glob("scrub_*.ipc"):
            try:
                s.unlink(missing_ok=True)
            except Exception:
                pass

        cls.supervisor = Supervisor(
            wav_path=TEST_WAV,
            host="127.0.0.1",
            port=TEST_PORT,
            enable_dsp=True,
            single_pass=True,
            max_restarts=1,
            dsp_denoise=False,
            dsp_spatial=False,
        )
        cls.sup_thread = threading.Thread(target=cls._run_supervisor, daemon=True)
        cls.sup_thread.start()

        # Wait until all initial pipeline services (MONITOR, NLU, ASR) are spawned and warmed up
        is_ready = cls.supervisor.wait_until_ready(timeout=100.0)
        if not is_ready:
            raise RuntimeError("Supervisor pipeline services failed to reach ready state within 100s")

        # Wait for HTTP server to become responsive
        client = httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=5.0)
        started = False
        for _ in range(60):
            try:
                res = client.get("/api/health")
                if res.status_code == 200:
                    started = True
                    break
            except Exception:
                time.sleep(0.5)
        client.close()
        if not started:
            raise RuntimeError("Supervisor failed to start web server within 30 seconds")

    @classmethod
    def _run_supervisor(cls):
        cls.supervisor.start_all(start_dsp=False)
        cls.supervisor.run_watchdog()

    @classmethod
    def tearDownClass(cls):
        cls.supervisor.stop_all()
        cls.sup_thread.join(timeout=3.0)

    def test_01_health_check(self):
        with httpx.Client(base_url=f"http://127.0.0.1:{TEST_PORT}", timeout=3.0) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "healthy")

    def test_02_stream_audio_and_receive_command(self):
        """Connect WebSocket to monitor, trigger DSP stream, and wait for voice command."""
        async def _test_ws():
            uri = f"ws://127.0.0.1:{TEST_PORT}/ws"
            received_messages = []
            async with websockets.connect(uri) as ws:
                # 1. First message must be snapshot
                init_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                init_data = orjson.loads(init_msg)
                self.assertEqual(init_data.get("type"), "snapshot")

                # 2. Trigger DSP audio stream now that WebSocket listener is hot
                dsp_spec = self.supervisor.services.get("DSP")
                self.assertIsNotNone(dsp_spec)
                self.supervisor.start_service(dsp_spec)

                # 3. Wait up to 35 seconds for DSP -> ASR -> NLU -> Monitor broadcast
                t0 = time.time()
                while time.time() - t0 < 35.0:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = orjson.loads(msg)
                        received_messages.append(data)
                        if data.get("type") == "command":
                            break
                    except asyncio.TimeoutError:
                        continue
            return received_messages

        messages = asyncio.run(_test_ws())
        self.assertGreater(len(messages), 0, "Should have received at least one WebSocket message")

        # Check if command arrived
        command_msgs = [m for m in messages if m.get("type") == "command"]
        self.assertGreater(len(command_msgs), 0, "Did not receive routed command message over WebSocket")

        cmd = command_msgs[0]["command"]
        print(f"\n  [E2E STREAM VERIFIED] Audio -> DSP -> ASR -> NLU -> Monitor:")
        print(f"    Intent: {cmd.get('intent')} | Status: {cmd.get('status')} | Category: {cmd.get('category')}")
        print(f"    Raw Transcript: '{cmd.get('raw_transcript')}' | Latency: {cmd.get('latency_ms', 0):.1f}ms")

        self.assertIn("intent", cmd)
        self.assertIn("category", cmd)
        self.assertIn("status", cmd)


if __name__ == "__main__":
    unittest.main()
