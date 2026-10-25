"""Run the DSP stage: Microphone array or WAV stream in, cleaned speech chunks out.

Streams 0.5s mono float32 chunks (16 kHz, 8000 samples) to the ASR stage via ZeroMQ PUSH socket.

Architecture:
  Gate 1: Silero VAD (voice activity detection)
  Gate 2: SRP-PHAT localizer (speaker direction of arrival)
  Mod 3:  DAS beamformer (constructive spatial filtering)
  Mod 4:  DeepFilterNet ONNX denoiser (neural noise suppression)

Address:
  SCRUB_AUDIO_ADDR: ZeroMQ socket where ASR binds PULL and DSP connects PUSH.
                    (default ipc:///tmp/scrub_audio.ipc)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import orjson
import soundfile as sf
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "src" / "dsp") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "dsp"))

from dsp.denoise.pi_inference import DeepFilterDenoiser
from dsp.source.audio_streamer import AudioStreamer
from dsp.source.beamformer import DASBeamformer
from dsp.source.localizer import SRPLocalizer
from dsp.source.vad import SileroVAD

DEFAULT_AUDIO_ADDR = os.getenv("SCRUB_AUDIO_ADDR", "ipc:///tmp/scrub_audio.ipc")
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DSP] %(message)s")
logger = logging.getLogger("dsp")


def connect_push_socket(addr: str, max_retries: int = 10, retry_delay: float = 0.5) -> tuple[zmq.Context, zmq.Socket]:
    """Connect PUSH socket to ASR with retry for IPC endpoint existence."""
    context = zmq.Context.instance()
    push = context.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 50)
    push.setsockopt(zmq.LINGER, 500)

    for attempt in range(max_retries):
        try:
            push.connect(addr)
            logger.info(f"Connected PUSH socket to {addr}")
            return context, push
        except zmq.ZMQError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to connect to {addr} ({e}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise
    return context, push


def run_dsp_service(
    audio_addr: str = DEFAULT_AUDIO_ADDR,
    wav_path: Optional[Path] = None,
    device: str = "plughw:seeed8micvoicec,0",
    use_denoiser: bool = True,
    use_spatial: bool = True,
    vad_threshold: float = 0.35,
    realtime: bool = True,
    loop: bool = False,
    dump_dir: Optional[Path] = None,
) -> None:
    """Run the DSP streaming loop."""
    logger.info("Initializing DSP subsystems...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=vad_threshold)

    # Warmup VAD
    dummy_multichannel = np.zeros((6, 8000), dtype=np.float32)
    _ = vad.is_speech_active(dummy_multichannel)
    logger.info("Silero VAD initialized and warmed up.")

    localizer = None
    beamformer = None
    if use_spatial:
        localizer = SRPLocalizer(sample_rate=SAMPLE_RATE, radius=0.04625, par_threshold=1.15)
        _ = localizer.localize(dummy_multichannel)
        beamformer = DASBeamformer(sample_rate=SAMPLE_RATE, radius=0.04625)
        _ = beamformer.beamform(dummy_multichannel, target_angles=0.0)
        logger.info("Spatial localizer and DAS beamformer initialized.")

    denoiser = None
    if use_denoiser:
        denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "deepfilternet.onnx"
        denoiser = DeepFilterDenoiser(model_path=denoiser_model, threads=1)
        # Warmup denoiser
        _ = denoiser.enhance(dummy_multichannel[0])
        logger.info("DeepFilterNet ONNX denoiser initialized.")

    context, push = connect_push_socket(audio_addr)

    mode_desc = f"Simulating from WAV: {wav_path}" if wav_path else f"Live hardware: {device}"
    logger.info(f"DSP active. {mode_desc} -> Streaming to {audio_addr}")

    streamer = AudioStreamer(
        device=device,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        wav_path=wav_path,
        realtime=realtime,
        loop=loop,
    )

    in_speech = False
    chunk_index = 0
    utterance_count = 0
    current_utterance_chunks: list[np.ndarray] = []
    all_session_chunks: list[np.ndarray] = []

    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio dump enabled -> processed audio will be saved to {dump_dir}")

    try:
        with streamer:
            while True:
                try:
                    chunk = streamer.read_chunk()
                except EOFError:
                    logger.info("Stream source reached EOF.")
                    if in_speech:
                        push.send_multipart([orjson.dumps({"type": "end"})])
                        if dump_dir and current_utterance_chunks:
                            utterance_count += 1
                            utt_audio = np.concatenate(current_utterance_chunks)
                            utt_path = dump_dir / f"utterance_{utterance_count:03d}.wav"
                            sf.write(str(utt_path), utt_audio, SAMPLE_RATE)
                            logger.info(f"Saved processed audio utterance to {utt_path} ({len(utt_audio)/SAMPLE_RATE:.2f}s)")
                            current_utterance_chunks = []
                        in_speech = False
                    break

                chunk_index += 1
                capture_time = time.time()

                # Gate 1: Voice Activity Detection
                t0 = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                t_vad = (time.perf_counter() - t0) * 1000

                if not is_speech:
                    if in_speech:
                        # Speech just ended: send 'end' message to trigger final ASR transcription
                        push.send_multipart([orjson.dumps({"type": "end"})])
                        if dump_dir and current_utterance_chunks:
                            utterance_count += 1
                            utt_audio = np.concatenate(current_utterance_chunks)
                            utt_path = dump_dir / f"utterance_{utterance_count:03d}.wav"
                            sf.write(str(utt_path), utt_audio, SAMPLE_RATE)
                            logger.info(f"Saved processed audio utterance to {utt_path} ({len(utt_audio)/SAMPLE_RATE:.2f}s)")
                            current_utterance_chunks = []
                        in_speech = False
                        logger.info(f"Chunk #{chunk_index:04d} | VAD: SILENCE ({t_vad:.1f}ms) -> Sent END to ASR")
                    continue

                in_speech = True

                # Gate 2 & Mod 3: Spatial Localizer & DAS Beamformer
                mono_audio = None
                angle_desc = "omni"
                if use_spatial and localizer is not None and beamformer is not None:
                    detected_angles, _ = localizer.localize(chunk)
                    if detected_angles:
                        angle_desc = f"{detected_angles[0]:.1f}°"
                        mono_audio = beamformer.beamform(chunk, target_angles=detected_angles[0])
                    else:
                        angle_desc = "diffuse"
                        mono_audio = np.mean(chunk, axis=0)
                else:
                    mono_audio = np.mean(chunk, axis=0)

                # Mod 4: Neural Denoiser
                if denoiser is not None:
                    clean_chunk = denoiser.enhance(mono_audio)
                else:
                    clean_chunk = mono_audio

                if dump_dir:
                    current_utterance_chunks.append(clean_chunk.copy())
                    all_session_chunks.append(clean_chunk.copy())

                header = {"type": "chunk", "capture_time": capture_time}
                push.send_multipart([orjson.dumps(header), clean_chunk.astype(np.float32).tobytes()])
                logger.info(
                    f"Chunk #{chunk_index:04d} | VAD: SPEECH ({t_vad:.1f}ms) | Angle: {angle_desc} -> Pushed 8000 samples to ASR"
                )

    except KeyboardInterrupt:
        logger.info("DSP service interrupted by user.")
    finally:
        if dump_dir:
            if current_utterance_chunks:
                utterance_count += 1
                utt_audio = np.concatenate(current_utterance_chunks)
                utt_path = dump_dir / f"utterance_{utterance_count:03d}.wav"
                sf.write(str(utt_path), utt_audio, SAMPLE_RATE)
                logger.info(f"Saved final audio utterance to {utt_path}")
            if all_session_chunks:
                session_path = dump_dir / "session_full_cleaned.wav"
                sf.write(str(session_path), np.concatenate(all_session_chunks), SAMPLE_RATE)
                logger.info(f"Saved full session audio to {session_path}")
        if in_speech:
            try:
                push.send_multipart([orjson.dumps({"type": "end"})], flags=zmq.NOBLOCK)
            except Exception:
                pass
        push.close(linger=500)
        context.term()
        logger.info("DSP service stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DSP stage: Multi-mic/WAV in, cleaned speech chunks out")
    parser.add_argument("--wav", type=Path, default=None, help="Simulate audio stream from WAV file")
    parser.add_argument("--device", type=str, default="plughw:seeed8micvoicec,0", help="ALSA device name")
    parser.add_argument("--audio-addr", type=str, default=DEFAULT_AUDIO_ADDR, help="ZeroMQ audio address")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", help="Bypass DeepFilterNet denoiser")
    parser.add_argument("--no-spatial", dest="spatial", action="store_false", help="Bypass localizer and beamformer")
    parser.add_argument("--vad-threshold", type=float, default=0.35, help="Silero VAD threshold")
    parser.add_argument("--fast", dest="realtime", action="store_false", help="Simulate faster than real time")
    parser.add_argument("--loop", action="store_true", help="Loop WAV file continuously")
    parser.add_argument("--dump-audio", type=Path, default=None, help="Directory to save cleaned audio utterances and full session")
    parser.set_defaults(denoise=True, spatial=True, realtime=True)

    args = parser.parse_args()
    run_dsp_service(
        audio_addr=args.audio_addr,
        wav_path=args.wav,
        device=args.device,
        use_denoiser=args.denoise,
        use_spatial=args.spatial,
        vad_threshold=args.vad_threshold,
        realtime=args.realtime,
        loop=args.loop,
        dump_dir=args.dump_audio,
    )


if __name__ == "__main__":
    main()

