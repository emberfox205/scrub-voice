import sys
import time
import subprocess
import numpy as np
from vad import SileroVAD

# ---------------- Audio & Hardware Configuration ----------------
DEVICE = "plughw:seeed8micvoicec,0"
SAMPLE_RATE = 16000
HARDWARE_CHANNELS = 8    # The AC108 hardware driver streams 8 channels
ACTIVE_MICS = 6          # Channels 0 to 5 are the 6 microphones
CHUNK_DURATION = 0.5     # Process audio every 0.5 seconds

# 16000 samples/sec * 0.5s = 8000 samples
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

# 8000 samples * 8 channels * 2 bytes (16-bit) = 128,000 bytes per chunk
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * HARDWARE_CHANNELS * 2

def main():
    print("⏳ Initializing Silero VAD...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=0.5)
    print("✅ Silero VAD initialized.")

    # Launch arecord in raw streaming mode (pipes pure PCM bytes directly to memory)
    cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(HARDWARE_CHANNELS),
        "-t", "raw"
    ]

    print(f"🎙️ Opening live audio stream from {DEVICE}...")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=BYTES_PER_CHUNK
        )
    except Exception as e:
        print(f"❌ Failed to start arecord: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("🟢 REAL-TIME LISTENING STARTED (0.5s chunks)")
    print("Speak into the 6-mic array to test. Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    frame_count = 0
    try:
        while True:
            # 1. Read 0.5s of audio directly from RAM buffer
            raw_bytes = process.stdout.read(BYTES_PER_CHUNK)
            if len(raw_bytes) < BYTES_PER_CHUNK:
                print("⚠️ Stream interrupted or buffer underrun.")
                break

            # 2. Convert bytes to 16-bit integer array
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)

            # 3. Reshape to (8000 samples, 8 channels)
            audio_data = audio_data.reshape(-1, HARDWARE_CHANNELS)

            # 4. Take only the 6 active microphones (drop channels 6 & 7)
            mics_6ch = audio_data[:, :ACTIVE_MICS]

            # 5. Transpose to (6 mics, 8000 samples) to match your vad.py input
            mics_6ch = mics_6ch.T

            # 6. Normalize int16 [-32768, 32767] to float32 [-1.0, 1.0]
            mics_float32 = mics_6ch.astype(np.float32) / 32768.0

            # 7. Check peak amplitude for visual monitoring
            peak_amp = np.max(np.abs(mics_6ch))

            # 8. Run Silero VAD
            t0 = time.perf_counter()
            is_speech = vad.is_speech_active(mics_float32)
            infer_ms = (time.perf_counter() - t0) * 1000

            # 9. Print real-time status
            frame_count += 1
            timestamp = time.strftime("%H:%M:%S")
            status = "🗣️ SPEECH DETECTED" if is_speech else "⚪ silence"

            # Quick visual audio meter
            bars = min(int(peak_amp / 600), 15)
            meter = "█" * bars + "░" * (15 - bars)

            print(f"[{timestamp}] #{frame_count:04d} | Peak: {peak_amp:5d} [{meter}] | VAD: {status:<18} ({infer_ms:4.1f}ms)")

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        process.terminate()
        process.wait()
        print("✅ Live stream closed cleanly.")

if __name__ == "__main__":
    main()
