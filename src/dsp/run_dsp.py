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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "src" / "dsp") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "dsp"))

from src.dsp.denoise.sota_onnx.deepfilternet_denoiser import DeepFilterDenoiser
from src.dsp.denoise.sota_onnx.gtcrn_denoiser import GTCRNDenoiser
from src.dsp.denoise.stateless_onnx.stateless_duofilter_denoiser import DuoFilterDenoiser
from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import DuoFilterStreamingDenoiser
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
    denoiser_type: str = "duofilter",
    use_spatial: bool = True,
    vad_threshold: float = 0.5,
    realtime: bool = True,
    loop: bool = False,
    atten_lim_db: Optional[float] = 20.0,
    split_band_hz: Optional[float] = 2500.0,
    silence_hangover_sec: float = 1.5,
    continuous_slope: bool = True,
    f_low: float = 1000.0,
    f_high: float = 3000.0,
    atten_low_db: float = 12.0,
    atten_high_db: float = 0.0,
    dither_dbfs: Optional[float] = -65.0,
    enable_snr_blend: bool = True,
    snr_low_db: float = 8.0,
    snr_high_db: float = 18.0,
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
        dtype = denoiser_type.lower()
        if dtype in ["duofilter", "duofilter_stateful"]:
            denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "onnx" / "duofilter_stateful.onnx"
            denoiser = DuoFilterStreamingDenoiser(
                model_path=denoiser_model,
                threads=1,
                atten_lim_db=atten_lim_db,
                split_band_hz=split_band_hz,
                continuous_slope=continuous_slope,
                f_low=f_low,
                f_high=f_high,
                atten_low_db=atten_low_db,
                atten_high_db=atten_high_db,
                dither_dbfs=dither_dbfs,
                enable_snr_blend=enable_snr_blend,
                snr_low_db=snr_low_db,
                snr_high_db=snr_high_db,
            )
            # Warmup denoiser
            _ = denoiser.enhance_chunk(dummy_multichannel[0])
            logger.info(
                f"Stateful DuoFilter ONNX denoiser initialized (continuous_slope={continuous_slope}, atten_low_db={atten_low_db} dB, dither={dither_dbfs} dBFS, snr_blend={enable_snr_blend})."
            )
        elif dtype == "duofilter_stateless":
            denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "onnx" / "duofilter.onnx"
            denoiser = DuoFilterDenoiser(
                model_path=denoiser_model,
                threads=1,
                atten_lim_db=atten_lim_db,
                split_band_hz=split_band_hz,
                continuous_slope=continuous_slope,
                f_low=f_low,
                f_high=f_high,
                atten_low_db=atten_low_db,
                atten_high_db=atten_high_db,
                dither_dbfs=dither_dbfs,
                enable_snr_blend=enable_snr_blend,
                snr_low_db=snr_low_db,
                snr_high_db=snr_high_db,
            )
            # Warmup denoiser
            _ = denoiser.enhance(dummy_multichannel[0])
            logger.info(
                f"Stateless DuoFilter ONNX denoiser initialized (continuous_slope={continuous_slope}, atten_low_db={atten_low_db} dB, dither={dither_dbfs} dBFS, snr_blend={enable_snr_blend})."
            )
        elif dtype == "gtcrn":
            denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "onnx" / "gtcrn_simple.onnx"
            denoiser = GTCRNDenoiser(model_path=denoiser_model, threads=1)
            # Warmup denoiser
            _ = denoiser.enhance(dummy_multichannel[0])
            logger.info("GTCRN ONNX denoiser initialized.")
        elif dtype == "deepfilter":
            denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "onnx" / "deepfilternet.onnx"
            if not denoiser_model.exists():
                denoiser_model = REPO_ROOT / "src" / "dsp" / "denoise" / "deepfilternet.onnx"
            denoiser = DeepFilterDenoiser(
                model_path=denoiser_model,
                threads=1,
                atten_lim_db=atten_lim_db,
                split_band_hz=split_band_hz,
                continuous_slope=continuous_slope,
                f_low=f_low,
                f_high=f_high,
                atten_low_db=atten_low_db,
                atten_high_db=atten_high_db,
                dither_dbfs=dither_dbfs,
                enable_snr_blend=enable_snr_blend,
                snr_low_db=snr_low_db,
                snr_high_db=snr_high_db,
            )
            # Warmup denoiser
            _ = denoiser.enhance(dummy_multichannel[0])
            logger.info(
                f"DeepFilterNet ONNX denoiser initialized (continuous_slope={continuous_slope}, atten_low_db={atten_low_db} dB, dither={dither_dbfs} dBFS, snr_blend={enable_snr_blend})."
            )
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}. Supported: 'duofilter', 'duofilter_stateless', 'gtcrn', 'deepfilter'")

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
    silence_chunks = 0
    hangover_silence_chunks = max(1, int(round(silence_hangover_sec / CHUNK_DURATION)))
    chunk_index = 0

    try:
        with streamer:
            while True:
                try:
                    chunk = streamer.read_chunk()
                except EOFError:
                    logger.info("Stream source reached EOF.")
                    if in_speech:
                        push.send_multipart([orjson.dumps({"type": "end"})])
                        in_speech = False
                    break

                chunk_index += 1
                capture_time = time.time()

                # Gate 1: Voice Activity Detection
                t0 = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                t_vad = (time.perf_counter() - t0) * 1000

                vad_status_desc = "SPEECH"
                if not is_speech:
                    if in_speech:
                        silence_chunks += 1
                        if silence_chunks > hangover_silence_chunks:
                            # Speech genuinely ended after sustained silence: send 'end' message to trigger final ASR transcription
                            push.send_multipart([orjson.dumps({"type": "end"})])
                            in_speech = False
                            silence_chunks = 0
                            logger.info(
                                f"Chunk #{chunk_index:04d} | VAD: SILENCE ({t_vad:.1f}ms) -> Sent END to ASR (silence >= {silence_hangover_sec:.1f}s)"
                            )
                            continue
                        else:
                            # Within silence hangover window: keep utterance alive and stream audio chunk
                            vad_status_desc = f"SILENCE-HANGOVER ({silence_chunks}/{hangover_silence_chunks})"
                    else:
                        continue
                else:
                    in_speech = True
                    silence_chunks = 0

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
                    if hasattr(denoiser, "enhance_chunk"):
                        try:
                            clean_chunk = denoiser.enhance_chunk(mono_audio, is_speech=is_speech)
                        except TypeError:
                            clean_chunk = denoiser.enhance_chunk(mono_audio)
                    else:
                        clean_chunk = denoiser.enhance(mono_audio)
                else:
                    clean_chunk = mono_audio

                # Compute dB attenuation applied by DeepFilterNet
                rms_in = float(np.sqrt(np.mean(mono_audio**2)))
                rms_out = float(np.sqrt(np.mean(clean_chunk**2)))
                chunk_db = 20 * np.log10((rms_out + 1e-9) / (rms_in + 1e-9))

                header = {"type": "chunk", "capture_time": capture_time}
                push.send_multipart([orjson.dumps(header), clean_chunk.astype(np.float32).tobytes()])
                logger.info(
                    f"Chunk #{chunk_index:04d} | VAD: {vad_status_desc} ({t_vad:.1f}ms) | Angle: {angle_desc} | Denoise: {chunk_db:+.1f}dB -> Pushed 8000 samples to ASR"
                )

    except KeyboardInterrupt:
        logger.info("DSP service interrupted by user.")
    finally:
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
    parser.add_argument(
        "--denoiser-type",
        choices=["duofilter", "duofilter_stateful", "duofilter_stateless", "gtcrn", "deepfilter"],
        default="duofilter",
        help="Select denoiser model: 'duofilter' (stateful, default), 'duofilter_stateless', 'gtcrn', or 'deepfilter'",
    )
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", help="Bypass AI neural denoiser")
    parser.add_argument("--no-spatial", dest="spatial", action="store_false", help="Bypass localizer and beamformer")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument("--fast", dest="realtime", action="store_false", help="Simulate faster than real time")
    parser.add_argument("--loop", action="store_true", help="Loop WAV file continuously")
    parser.add_argument(
        "--atten-lim-db",
        type=float,
        default=20.0,
        help="Attenuation limit in dB (default: 20.0 to preserve natural acoustic floor for ASR, or >= 100 for full suppression)",
    )
    parser.add_argument(
        "--split-band-hz",
        type=float,
        default=2500.0,
        help="Split-band frequency in Hz (default: 2500.0 to shield consonants for ASR, or 0 / >= 8000 to disable)",
    )
    parser.add_argument(
        "--silence-hangover-sec",
        type=float,
        default=1.5,
        help="Sustained silence duration (seconds) before DSP signals utterance end to ASR (default: 1.5s)",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=1000.0,
        help="Low frequency transition boundary (Hz) for continuous psychoacoustic slope (default: 1000.0 Hz)",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=3000.0,
        help="High frequency pass-through boundary (Hz) for consonant preservation (default: 3000.0 Hz)",
    )
    parser.add_argument(
        "--atten-low-db",
        type=float,
        default=12.0,
        help="Low-frequency attenuation limit in dB for rumble cleanup (default: 12.0 dB)",
    )
    parser.add_argument(
        "--atten-high-db",
        type=float,
        default=0.0,
        help="High-frequency attenuation limit in dB (default: 0.0 dB for 100%% pass-through)",
    )
    parser.add_argument(
        "--dither-dbfs",
        type=float,
        default=-65.0,
        help="Micro-dither floor in dBFS for Kaldi CMVN stabilization (default: -65.0 dBFS)",
    )
    parser.add_argument(
        "--no-continuous-slope",
        dest="continuous_slope",
        action="store_false",
        help="Disable continuous slope and revert to legacy split-band filter",
    )
    parser.add_argument(
        "--no-snr-blend",
        dest="snr_blend",
        action="store_false",
        help="Disable soft-knee SNR dynamic blending",
    )
    parser.set_defaults(denoise=True, spatial=True, realtime=True, continuous_slope=True, snr_blend=True)

    args = parser.parse_args()
    run_dsp_service(
        audio_addr=args.audio_addr,
        wav_path=args.wav,
        device=args.device,
        use_denoiser=args.denoise,
        denoiser_type=args.denoiser_type,
        use_spatial=args.spatial,
        vad_threshold=args.vad_threshold,
        realtime=args.realtime,
        loop=args.loop,
        atten_lim_db=args.atten_lim_db,
        split_band_hz=args.split_band_hz,
        silence_hangover_sec=args.silence_hangover_sec,
        continuous_slope=args.continuous_slope,
        f_low=args.f_low,
        f_high=args.f_high,
        atten_low_db=args.atten_low_db,
        atten_high_db=args.atten_high_db,
        dither_dbfs=args.dither_dbfs,
        enable_snr_blend=args.snr_blend,
    )


if __name__ == "__main__":
    main()

