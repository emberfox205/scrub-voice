import sys
import time
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from collections import defaultdict

# 1. Directory Structure Resolution
test_dir  = Path(__file__).resolve().parent   # .../src/dsp/test
dsp_dir   = test_dir.parent                  # .../src/dsp
src_dir   = dsp_dir.parent                   # .../src
repo_root = src_dir.parent                   # .../scrub-voice

# Add dsp_dir to sys.path so Python finds the 'source' package
if str(dsp_dir) not in sys.path:
    sys.path.insert(0, str(dsp_dir))

# 2. Imports from DSP source modules
from source.audio_streamer import AudioStreamer
from source.vad import SileroVAD
from source.localizer import SRPLocalizer
from source.beamformer import DASBeamformer

# 3. Hardware & Pipeline Constants
DEVICE = "plughw:seeed8micvoicec,0"  # ReSpeaker 6-mic ALSA device name on Pi
SAMPLE_RATE = 16000                  # 16 kHz audio
CHUNK_DURATION = 0.5                 # 500 ms per chunk (8000 samples)
OUTPUT_DIR = repo_root / "data" / "dsp" / "output"

def initialize_system():
    """
    Initializes and warms up all DSP models before opening the audio hardware.
    """
    print("=" * 70)

    # 1. Gate 1: Silero VAD
    print("⏳ [1/3] Initializing Silero VAD (Gate 1)...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=0.35)

    # PyTorch cold-start warmup
    dummy_chunk = np.zeros((6, 8000), dtype=np.float32)
    _ = vad.is_speech_active(dummy_chunk)
    print("   ✓ Silero VAD ready & warmed up.")

    # 2. Gate 2: SRP Localizer
    print("⏳ [2/3] Initializing SRP Localizer (Gate 2)...")
    localizer = SRPLocalizer(sample_rate=SAMPLE_RATE, radius=0.04625, par_threshold=1.15)
    _ = localizer.localize(dummy_chunk)
    print("   ✓ SRP Localizer ready (steering phases precomputed).")

    # 3. Module DAS Beamformer
    print("⏳ [3/3] Initializing DAS Beamformer (Module 3)...")
    beamformer = DASBeamformer(sample_rate=SAMPLE_RATE, radius=0.04625)
    _ = beamformer.beamform(dummy_chunk, target_angle_deg=0.0)
    print("   ✓ DAS Beamformer ready (360° weights precomputed).")

    print("=" * 70)
    print("🎙️ All models hot in memory. Ready to open ALSA microphone stream.")
    print("=" * 70)
    
    return vad, localizer, beamformer

def run_live_pipeline():
    # 1. Warmup models
    vad, localizer, beamformer = initialize_system()

    # Ensures data/dsp/output/ directory exists on the Pi!
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🟢 REAL-TIME DSP PIPELINE ACTIVE (0.5s Chunks)")
    print("   Gate 1: Silero-VAD   (Temporal Speech Detector)")
    print("   Gate 2: SRP-PHAT     (Direction & Coherence Filter)")
    print("   Mod  3: DAS-Beam     (Multi-Target Spatial Isolation)")
    print("👉 Speak into the array. Press Ctrl+C to stop and save audio.")
    print("=" * 80 + "\n")

    frame_count = 0
    raw_recording = []

    # Fully dynamic dictionary: handles Stream 0, 1, 2, ... automatically
    beam_recordings = defaultdict(list)

    try:
        # 2. Open live ALSA audio capture stream
        with AudioStreamer(device=DEVICE, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION) as streamer:
            # Discard initial ADC power-on transient click
            _ = streamer.read_chunk()

            while True:
                # -------------------------------------------------------------
                # Ingestion: Read live 0.5s multi-channel chunk (6, 8000)
                # -------------------------------------------------------------
                chunk = streamer.read_chunk()
                frame_count += 1
                timestamp = time.strftime("%H:%M:%S")      

                # Visual Volume Meter
                peak_amp = int(np.max(np.abs(chunk)) * 32768)
                bars = min(int(peak_amp / 600), 12)
                meter = "█" * bars + "░" * (12 - bars)

                # -------------------------------------------------------------
                # Gate 1: Voice Activity Detection (Silero VAD)
                # -------------------------------------------------------------

                t_vad = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                vad_ms = (time.perf_counter() - t_vad) * 1000    

                # EARLY-EXIT: Silence -> Save CPU
                if not is_speech:
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: ⚪ SILENCE ({vad_ms:4.1f}ms) ──► 🛑 Gate 1 ABORT"
                    )
                    continue

                # -------------------------------------------------------------
                # Gate 2: Multi-Speaker Localization (SRP-PHAT)
                # -------------------------------------------------------------
                t_loc = time.perf_counter()
                detected_angles, _ = localizer.localize(chunk)
                loc_ms = (time.perf_counter() - t_loc) * 1000

                # DIFFUSE ECHO EXIT: Reverberation / Flat energy -> Discard
                if not detected_angles:
                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: 🗣️ SPEECH  ({vad_ms:4.1f}ms) ──► ⚠️ Gate 2 ABORT: Diffuse/Echo ({loc_ms:4.1f}ms)"
                    )
                    continue

                # -------------------------------------------------------------
                # Module 3: Dynamic Multi-Target Beamforming (DAS)
                # Steers simultaneous beams for ALL detected angles
                # -------------------------------------------------------------
                t_beam = time.perf_counter()
                enhanced_streams = beamformer.beamform_multi(chunk, target_angles_deg=detected_angles)
                beam_ms = (time.perf_counter() - t_beam) * 1000

                total_latency_ms = vad_ms + loc_ms + beam_ms
                angles_str = ", ".join(f"{a:.0f}°" for a in detected_angles)

                # Live Multi-Beam Telemetry
                print(
                    f"[{timestamp}] #{frame_count:04d} | "
                    f"Peak: {peak_amp:5d} [{meter}] | "
                    f"VAD: 🗣️ ({vad_ms:4.1f}ms) | "
                    f"LOC: 📍 [{angles_str}] ({loc_ms:4.1f}ms) | "
                    f"BEAM: 🎧 {len(enhanced_streams)} Stream(s) ({beam_ms:4.1f}ms) ──► Total: {total_latency_ms:4.1f}ms"
                )

                # Accumulate raw reference (Mic 0)
                raw_recording.append(chunk[0].copy())

                # Dynamically buffer each isolated beam stream
                for idx, stream in enumerate(enhanced_streams):
                    beam_recordings[idx].append(stream.copy())

    except KeyboardInterrupt:
        print("\n🛑 Stopping live pipeline (Ctrl+C caught)...")
    except Exception as e:
        print(f"\n❌ Pipeline runtime error: {e}")
    finally:
        # Check if any speech frames were captured
        if raw_recording:
            raw_full = np.concatenate(raw_recording)
            duration_sec = len(raw_full) / SAMPLE_RATE

            raw_path = OUTPUT_DIR / "live_raw.wav"
            # Scale float32 [-1.0, 1.0] to int16 PCM for standard WAV players
            wavfile.write(str(raw_path), SAMPLE_RATE, (raw_full * 32767).astype(np.int16))

            print("\n" + "=" * 70)
            print(f"💾 Captured {duration_sec:.1f}s of active speech audio:")
            print(f"   👉 Raw Reference:     {raw_path}")

            # Dynamically export every active beamformed stream
            for stream_idx, chunks in beam_recordings.items():
                if chunks:
                    beam_full = np.concatenate(chunks)
                    beam_path = OUTPUT_DIR / f"live_beam_stream{stream_idx}.wav"
                    wavfile.write(str(beam_path), SAMPLE_RATE, (beam_full * 32767).astype(np.int16))
                    print(f"   👉 Beamformed Stream #{stream_idx}:       {beam_path}")
            print("=" * 70)
        else:
            print("\nℹ️ No active speech was detected during this run.")

        print("✅ ALSA hardware audio stream closed cleanly.")

if __name__ == "__main__":
    run_live_pipeline()