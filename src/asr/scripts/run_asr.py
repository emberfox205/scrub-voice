"""Run the ASR stage: DSP audio chunks in, one JSON per utterance out
(contract: src/asr/CLAUDE.md "Output schema"; message formats:
docs/ASR_GUIDE.md).

Two modes, from the repository root:

    # Service (pipeline on the Pi): ZeroMQ in from DSP, ZeroMQ out to NLU
    python src/asr/scripts/run_asr.py

    # Simulation (no microphone array needed): wav files are cut into 0.5 s
    # chunks, and only chunks with speech are passed on - like the DSP does.
    # Several files are played one after the other as one stream.
    python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav

    # Same, but also send the results to NLU (played in real time, so the
    # timestamps are real): tests ASR -> NLU without the DSP
    python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav --send

Addresses (environment variables, same style as src/orchestrator/):
    SCRUB_AUDIO_ADDR  ASR binds a PULL socket here; DSP connects a PUSH socket
                      (default ipc:///tmp/scrub_audio.ipc)
    SCRUB_ASR_ADDR    ASR connects a PUSH socket here; NLU binds a PULL socket
                      (default ipc:///tmp/scrub_asr.ipc)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # src/ on sys.path, like the other modules

from asr.audio import TARGET_SR, load_audio
from asr.bench_configs import CONFIGS, PRODUCTION_CONFIG
from asr.stream import UtteranceStream

DEFAULT_AUDIO_ADDR = os.getenv("SCRUB_AUDIO_ADDR", "ipc:///tmp/scrub_audio.ipc")
DEFAULT_ASR_ADDR = os.getenv("SCRUB_ASR_ADDR", "ipc:///tmp/scrub_asr.ipc")
CHUNK_S = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [ASR] %(message)s")
logger = logging.getLogger("asr")

DEBUG_LOG_DIR = REPO_ROOT / "results"
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG_FILE = DEBUG_LOG_DIR / "asr_transcripts.jsonl"


def log_result(result, seconds: float) -> None:
    words_detail = ", ".join(f"{w.word}:{w.confidence:.2f}" for w in result.words) if result.words else "none"
    logger.info(
        f"{result.text!r} -> {result.normalized!r} (conf: {result.confidence}, {seconds:.2f}s) | Tokens: [{words_detail}]"
    )
    try:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(result.to_json() + "\n")
    except Exception:
        pass


def open_push(asr_addr: str):
    """PUSH socket to NLU (NLU binds a PULL socket at asr_addr)."""
    import zmq

    push = zmq.Context.instance().socket(zmq.PUSH)
    push.connect(asr_addr)
    return push


def simulate(stream: UtteranceStream, wav_paths: list[Path], push=None) -> None:
    """Play wav files as a DSP chunk stream and print one JSON line per
    utterance. Speech chunks are found with the Silero voice detector that
    ships with faster-whisper (the DSP uses Silero too).

    With a push socket, every result is also sent to NLU, and the files are
    played in real time: otherwise segment_start would lie in the future and
    NLU's latency would come out negative."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = np.concatenate([load_audio(p) for p in wav_paths])
    # Short min silence: the DSP decides chunk by chunk and does not glue
    # speech across pauses (Silero's default would join pauses under 2 s).
    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=100, speech_pad_ms=0))
    chunk_len = int(CHUNK_S * TARGET_SR)
    clock = time.time()  # simulated capture time of the first chunk

    def finish(result, started: float) -> None:
        if result is not None:
            log_result(result, time.perf_counter() - started)
            print(result.to_json(), flush=True)
            if push is not None:
                push.send(result.to_json().encode("utf-8"))

    for i in range(0, len(audio) - chunk_len + 1, chunk_len):
        if push is not None:  # real time: wait until this chunk would be recorded completely
            time.sleep(max(0.0, clock + (i + chunk_len) / TARGET_SR - time.time()))
        has_speech = any(s["start"] < i + chunk_len and s["end"] > i for s in speech)
        started = time.perf_counter()
        if has_speech:
            finish(stream.feed(audio[i : i + chunk_len], clock + i / TARGET_SR), started)
        else:
            finish(stream.end_utterance(), started)  # the DSP stops sending: speech is over
    started = time.perf_counter()
    finish(stream.end_utterance(), started)


def serve(stream: UtteranceStream, audio_addr: str, asr_addr: str, end_timeout_s: float) -> None:
    """DSP message (multipart):  [header, audio]
         header = orjson {"type": "chunk", "capture_time": <epoch seconds of the first sample>}
         audio  = float32 mono 16 kHz samples (raw bytes), normally 8000 of them
       or [orjson {"type": "end"}] when the DSP's detector says speech stopped.
    Without an "end" message, the utterance ends after end_timeout_s
    without chunks."""
    import orjson
    import zmq

    context = zmq.Context.instance()
    pull = context.socket(zmq.PULL)
    pull.bind(audio_addr)
    push = open_push(asr_addr)
    logger.info(f"Listening for DSP chunks on {audio_addr}, sending results to {asr_addr}")

    def send(result, started: float) -> None:
        if result is not None:
            log_result(result, time.perf_counter() - started)
            push.send(result.to_json().encode("utf-8"))

    last_chunk = time.monotonic()
    try:
        while True:
            if not pull.poll(timeout=100):
                if stream.has_audio() and time.monotonic() - last_chunk > end_timeout_s:
                    started = time.perf_counter()
                    send(stream.end_utterance(), started)
                continue
            parts = pull.recv_multipart()
            header = orjson.loads(parts[0])
            started = time.perf_counter()
            if header.get("type") == "end":
                send(stream.end_utterance(), started)
            elif header.get("type") == "chunk" and len(parts) == 2:
                chunk = np.frombuffer(parts[1], dtype=np.float32)
                send(stream.feed(chunk, float(header["capture_time"])), started)
                last_chunk = time.monotonic()
            else:
                logger.warning(f"Ignored message with header {header}")
    except KeyboardInterrupt:
        logger.info("Stopping.")
    finally:
        pull.close(linger=0)
        push.close(linger=1000)
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR stage: DSP chunks in, one JSON per utterance out")
    parser.add_argument("--wav", type=Path, nargs="+", help="Simulate the DSP stream from wav file(s) and print JSON")
    parser.add_argument("--send", action="store_true", help="With --wav: also send the results to NLU (SCRUB_ASR_ADDR)")
    parser.add_argument("--config", default=PRODUCTION_CONFIG, choices=list(CONFIGS))
    parser.add_argument("--audio-addr", default=DEFAULT_AUDIO_ADDR)
    parser.add_argument("--asr-addr", default=DEFAULT_ASR_ADDR)
    parser.add_argument("--end-timeout", type=float, default=0.8,
                        help="Seconds without a chunk after which an utterance counts as finished")
    args = parser.parse_args()

    started = time.perf_counter()
    stream = UtteranceStream(CONFIGS[args.config]())
    logger.info(f"Engine '{args.config}' loaded in {time.perf_counter() - started:.1f} s")
    if args.wav:
        push = open_push(args.asr_addr) if args.send else None
        if push is not None:
            logger.info(f"Sending results to {args.asr_addr} (playing in real time)")
        simulate(stream, args.wav, push)
        if push is not None:
            push.close(linger=2000)  # give queued messages time to reach NLU
    else:
        serve(stream, args.audio_addr, args.asr_addr, args.end_timeout)


if __name__ == "__main__":
    main()
