import sys
import time
from pathlib import Path
import numpy as np

# Dynamic path resolution to find 'source'
test_dir = Path(__file__).resolve().parent
dsp_dir = test_dir.parent
if str(dsp_dir) not in sys.path:
    sys.path.insert(0, str(dsp_dir))

from source.audio_streamer import AudioStreamer
from source.vad import SileroVAD
from source.localizer import SRPLocalizer

# Hardware configuration constants
DEVICE = "plughw:seeed8micvoicec,0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # 500 ms chunks

def initialize_system():
    print("=" * 65)
    print("⏳ [1/3] Initializing Silero VAD (Gate 1)...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=0.35)

    # Cold-start warmup for PyTorch
    dummy_chunk = np.zeros((6, 8000), dtype=np.float32)
    _ = vad.is_speech_active(dummy_chunk)
    print("   ✓ Silero VAD ready & warmed up.")

    print("⏳ [2/3] Initializing SRP Localizer (Gate 2)...")
    localizer = SRPLocalizer(
        sample_rate=SAMPLE_RATE,
        radius=0.04625,
        par_threshold=1.15
    )

    # Cold-start warmup for FFT / NumPy
    _ = localizer.localize(dummy_chunk)
    print("   ✓ SRP Localizer ready (steering phases precomputed).")

    print("🎙️ [3/3] Ready to open hardware audio stream.")
    print("=" * 65)
    return vad, localizer

def run_live_pipeline():
    # 1. Warm up all models before starting the audio hardware
    vad, localizer = initialize_system()

    print("\n" + "=" * 75)
    print("🟢 REAL-TIME SURGICAL AUDIO PIPELINE ACTIVE (0.5s Chunks)")
    print("   Gate 1: Silero-VAD  (Temporal Filter)")
    print("   Gate 2: SRP-PHAT    (Spatial Multi-Speaker Filter)")
    print("👉 Speak into the 6-mic array to test. Press Ctrl+C to stop cleanly.")
    print("=" * 75 + "\n")

    frame_count = 0

    try:
        with AudioStreamer(device=DEVICE, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION) as streamer:
            # Discardd the very first power-on click from the ADC chips
            _ = streamer.read_chunk()

            while True:
                # -------------------------------------------------------------
                # 0. Ingestion: Read 0.5s multi-channel chunk (6, 8000)
                # -------------------------------------------------------------
                chunk = streamer.read_chunk()
                frame_count += 1
                timestamp = time.strftime("%H:%M:%S")

                # Measure volume for the visual peak meter
                peak_amp = int(np.max(np.abs(chunk)) * 32768)
                bars = min(int(peak_amp / 600), 12)
                meter = "█" * bars + "░" * (12 - bars)          

                # -------------------------------------------------------------
                # Gate 1: Voice Activity Detection (Silero VAD)
                # -------------------------------------------------------------
                t_vad = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                vad_ms = (time.perf_counter() - t_vad) * 1000

                # 🎙️ ADD THIS LINE: Check if all 6 mics are hearing sound
                if is_speech:
                    rms_mics = np.round(np.sqrt(np.mean(chunk**2, axis=1)), 3)
                    print(f"      🎙️ [Mics 1..6 RMS]: {rms_mics}")

                # EARLY-EXIT GATING: If silence/noise, abort now to save CPU!
                if not is_speech:
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: ⚪ SILENCE ({vad_ms:4.1f}ms) ──► 🛑 Gate 1 ABORT (Save CPU & Sleep)"
                    )
                    continue

                # -------------------------------------------------------------
                # Gate 2: Direction of Arrival & Multi-Speaker Localization
                # Only reached when human speech is genuinely detected!
                # -------------------------------------------------------------
                t_loc = time.perf_counter()
                detected_angles, _ = localizer.localize(chunk)
                loc_ms = (time.perf_counter() - t_loc) * 1000
                total_latency_ms = vad_ms + loc_ms

                # Gate 2 Evaluation
                if not detected_angles:
                    # Voice detected, but sound is diffuse echo or multipath reverberation
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: 🗣️ SPEECH  ({vad_ms:4.1f}ms) ──► ⚠️ Gate 2 ABORT: Diffuse/Echo ({loc_ms:4.1f}ms)"
                    )
                else:
                    # Valid direct point-source speaker(s) located!
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: 🗣️ SPEECH  ({vad_ms:4.1f}ms) ──► ✅ Gate 2 PASS: 📍 Speaker at {detected_angles}° "
                        f"(Total: {total_latency_ms:4.1f}ms)"
                    )                   

    except KeyboardInterrupt:
        print("\n🛑 Stopping live pipeline (Ctrl+C caught)...")
    except Exception as e:
        print(f"\n❌ Pipeline runtime error: {e}")
    finally:
        print("✅ ALSA hardware audio stream closed cleanly.")


if __name__ == "__main__":
    run_live_pipeline()