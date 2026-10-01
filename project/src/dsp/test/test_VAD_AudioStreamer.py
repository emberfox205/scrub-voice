import sys
import time
from pathlib import Path
import numpy as np

# Ensure Python finds 'source' folder regardless of execution path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from source.audio_streamer import AudioStreamer
from source.vad import SileroVAD

def main():
    print("⏳ Initializing Silero VAD...")
    vad = SileroVAD(sample_rate=16000, threshold=0.35)
    print("✅ Silero VAD initialized.")

    print("\n" + "=" * 70)
    print("🟢 REAL-TIME LISTENING STARTED")
    print("Speak into the 6-mic array to test. Press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    frame_count = 0
    try:
        with AudioStreamer(device="plughw:seeed8micvoicec,0", sample_rate=16000, chunk_duration=0.5) as streamer:
            while True:
                # Read 0.5s chunk -> shape: (6, 8000), dtype: floa32, range: [-1.0, 1.0]
                mics_float32 = streamer.read_chunk()

                # Convert peak amplitude to int16 scale [0, 32768] for the plotting
                peak_amp = int(np.max(np.abs(mics_float32)) * 32768)

                # Run SileroVAD gate
                t0 = time.perf_counter()
                is_speech = vad.is_speech_active(mics_float32)
                infer_ms = (time.perf_counter() - t0) * 1000

                # Print status
                frame_count += 1
                timestamp = time.strftime("%H:%M:%S")
                status = "🗣️ SPEECH DETECTED" if is_speech else "⚪ silence"

                bars = min(int(peak_amp / 600), 15)
                meter = "█" * bars + "░" * (15 - bars)       

                print(
                    f"[{timestamp}] #{frame_count:04d} | "
                    f"Peak: {peak_amp:5d} [{meter}] | "
                    f"VAD: {status:<18} ({infer_ms:4.1f}ms)"
                )                         

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("✅ Live stream closed cleanly.")

if __name__ == "__main__":
    main()