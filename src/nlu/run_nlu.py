#!/usr/bin/env python3
"""
Scrub-Voice NLU Service Daemon.

Listens for transcribed speech text from the ASR subsystem via ZeroMQ (PULL),
executes the VoiceToIntentPipeline (lexical normalization, SetFit classification,
slot extraction, and 10-key schema validation), and routes validated command
envelopes to the central Orchestrator (dual PUSH: normal vs. priority).

Conforms strictly to:
  - schemas/command_schema.json (Draft 2020-12)
  - docs/COMMAND_SCHEMA_GUIDE.md
  - AGENTS.md (Dynamic paths, process isolation, sub-200ms recovery)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
import zmq

# Dynamic path resolution according to AGENTS.md
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlu.pipeline import VoiceToIntentPipeline

IS_WINDOWS = sys.platform == "win32"

DEFAULT_ASR_ADDR = os.getenv(
    "SCRUB_ASR_ADDR",
    "tcp://127.0.0.1:5555" if IS_WINDOWS else "ipc:///tmp/scrub_asr.ipc",
)
DEFAULT_NLU_ADDR = os.getenv(
    "SCRUB_NLU_ADDR",
    "tcp://127.0.0.1:5556" if IS_WINDOWS else "ipc:///tmp/scrub_nlu.ipc",
)
DEFAULT_PRIO_ADDR = os.getenv(
    "SCRUB_NLU_PRIORITY_ADDR",
    os.getenv(
        "SCRUB_PRIO_ADDR",
        "tcp://127.0.0.1:5557" if IS_WINDOWS else "ipc:///tmp/scrub_nlu_priority.ipc",
    ),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NLU] %(message)s",
)
logger = logging.getLogger("nlu")


def clean_ipc_socket(addr: str) -> None:
    """Removes stale Unix domain socket file if present."""
    if addr.startswith("ipc://"):
        sock_path = Path(addr[6:])
        sock_path.unlink(missing_ok=True)
        sock_path.parent.mkdir(parents=True, exist_ok=True)


class NLUService:
    """Standalone ZeroMQ NLU microservice daemon."""

    def __init__(
        self,
        asr_addr: str = DEFAULT_ASR_ADDR,
        nlu_addr: str = DEFAULT_NLU_ADDR,
        priority_addr: str = DEFAULT_PRIO_ADDR,
        model_dir: Optional[str] = None,
        num_threads: int = 1,
    ) -> None:
        self.asr_addr = asr_addr
        self.nlu_addr = nlu_addr
        self.priority_addr = priority_addr
        self.model_dir = model_dir
        self.num_threads = num_threads

        logger.info("Initializing VoiceToIntentPipeline (threads=%d)...", num_threads)
        t0 = time.perf_counter()
        self.pipeline = VoiceToIntentPipeline(
            model_dir=self.model_dir,
            num_threads=self.num_threads,
        )
        logger.info(
            "NLU Pipeline successfully initialized in %.2fs (classes: %d)",
            time.perf_counter() - t0,
            len(self.pipeline.labels),
        )

        self.context = zmq.Context()
        self.pull_sock: Optional[zmq.Socket] = None
        self.push_sock: Optional[zmq.Socket] = None
        self.priority_push_sock: Optional[zmq.Socket] = None
        self._running = False

    def setup_sockets(self) -> None:
        """Binds incoming ASR PULL socket and connects outgoing Orchestrator PUSH sockets."""
        clean_ipc_socket(self.asr_addr)

        # 1. Ingest from ASR (bind PULL)
        self.pull_sock = self.context.socket(zmq.PULL)
        self.pull_sock.bind(self.asr_addr)
        logger.info("ASR Ingest Socket bound to %s (PULL)", self.asr_addr)

        # 2. Outgoing Normal commands to Dispatcher (connect PUSH)
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.connect(self.nlu_addr)
        logger.info("Normal Command Socket connected to %s (PUSH)", self.nlu_addr)

        # 3. Outgoing Priority commands to Dispatcher (connect PUSH)
        self.priority_push_sock = self.context.socket(zmq.PUSH)
        self.priority_push_sock.connect(self.priority_addr)
        logger.info("Priority Command Socket connected to %s (PUSH)", self.priority_addr)

    def extract_transcript(self, raw_msg: bytes) -> str:
        """Parses incoming message bytes into a text utterance."""
        try:
            data = orjson.loads(raw_msg)
            if isinstance(data, dict):
                # Check standard ASRResult keys
                return (
                    data.get("text")
                    or data.get("raw_transcript")
                    or data.get("normalized")
                    or ""
                ).strip()
            return str(data).strip()
        except Exception:
            # Fallback to plain UTF-8 text string
            return raw_msg.decode("utf-8", errors="replace").strip()

    def process_and_route(self, raw_transcript: str) -> Optional[Dict[str, Any]]:
        """Processes utterance through pipeline and pushes envelope to appropriate channel."""
        if not raw_transcript:
            return None

        t0 = time.perf_counter()
        envelope = self.pipeline.process(raw_transcript)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        is_priority = envelope.get("priority") == "HIGH"
        dest_sock = self.priority_push_sock if is_priority else self.push_sock
        dest_name = "PRIORITY" if is_priority else "NORMAL"

        payload_bytes = orjson.dumps(envelope)
        if dest_sock:
            dest_sock.send(payload_bytes)

        slots = envelope.get("slots", {})
        slots_str = f" | slots={slots}" if slots else ""
        logger.info(
            "Routed [%s] -> %s (%s | %s) status=%s conf=%.2f in %.2fms%s",
            dest_name,
            envelope.get("command_id"),
            envelope.get("category"),
            envelope.get("intent"),
            envelope.get("status"),
            envelope.get("confidence", 0.0),
            pipeline_ms,
            slots_str,
        )

        return envelope

    def run(self) -> None:
        """Main service loop polling ASR transcripts and routing envelopes."""
        self.setup_sockets()
        self._running = True

        poller = zmq.Poller()
        if self.pull_sock:
            poller.register(self.pull_sock, zmq.POLLIN)

        logger.info("NLU Service Daemon started and awaiting utterances...")

        try:
            while self._running:
                try:
                    socks = dict(poller.poll(timeout=200))
                except zmq.ZMQError:
                    if not self._running:
                        break
                    raise
                if self.pull_sock and socks.get(self.pull_sock) == zmq.POLLIN:
                    raw_msg = self.pull_sock.recv()
                    transcript = self.extract_transcript(raw_msg)
                    if transcript:
                        try:
                            self.process_and_route(transcript)
                        except Exception as e:
                            logger.error("Error processing transcript %r: %s", transcript, e, exc_info=True)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Termination signal received. Shutting down NLU Service.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully closes all sockets and context."""
        self._running = False
        time.sleep(0.05)
        for s in (self.pull_sock, self.push_sock, self.priority_push_sock):
            if s is not None and not s.closed:
                try:
                    s.close(linger=0)
                except Exception:
                    pass
        if self.context and not self.context.closed:
            try:
                self.context.destroy(linger=0)
            except Exception:
                pass
        clean_ipc_socket(self.asr_addr)
        logger.info("NLU Service Daemon gracefully stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub-Voice NLU ZeroMQ Service Daemon")
    parser.add_argument("--asr-addr", default=DEFAULT_ASR_ADDR, help="ASR PULL socket address")
    parser.add_argument("--nlu-addr", default=DEFAULT_NLU_ADDR, help="Orchestrator normal PUSH address")
    parser.add_argument("--priority-addr", default=DEFAULT_PRIO_ADDR, help="Orchestrator priority PUSH address")
    parser.add_argument("--model-dir", default=None, help="Path to SetFit model directory")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU worker threads (default: 1)")
    parser.add_argument("--test-utterance", type=str, default=None, help="Process single utterance and exit")
    args = parser.parse_args()

    service = None

    def handle_signal(sig, frame):
        logger.info("Caught signal %d; initiating shutdown...", sig)
        if service is not None:
            service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        service = NLUService(
            asr_addr=args.asr_addr,
            nlu_addr=args.nlu_addr,
            priority_addr=args.priority_addr,
            model_dir=args.model_dir,
            num_threads=args.threads,
        )

        if args.test_utterance:
            print(f"Testing utterance: {args.test_utterance}")
            envelope = service.pipeline.process(args.test_utterance)
            print(json.dumps(envelope, indent=2))
            sys.exit(0)

        service.run()
    except (KeyboardInterrupt, SystemExit):
        if service is not None:
            service.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
