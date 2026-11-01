import sys
import time
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Dynamic path resolution
SOURCE_DIR = Path(__file__).resolve().parent
DSP_DIR = SOURCE_DIR.parent
SRC_DIR = DSP_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(DSP_DIR) not in sys.path:
    sys.path.insert(0, str(DSP_DIR))

# DSP Gate Modules
from source.audio_streamer import AudioStreamer
from source.vad import SileroVAD
from source.localizer import SRPLocalizer
from source.beamformer import DASBeamformer
from source.speaker_verifier import SpeakerVerifier
from source.enroll_doctor import DoctorEnroller

# Hardware & Pipeline Constants
DEVICE = "plughw:seeed8micvoicec,0"  # ReSpeaker 6-Mic array on Raspberry Pi
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5                 # 500ms chunks (8000 samples)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_TTL_FRAMES = 4      # 4 frames * 0.5s = 2.0s tracking memory
TOLERANCE_DEG = 15.0    # Tolerance for the difference between 2 distinguishable angles

DOCTOR_PROFILE_PATH = DSP_DIR / "models" / "ecapa_tdnn" / "doctor_profile.npy"
OUTPUT_DIR = REPO_ROOT / "data" / "dsp" / "output"

def initialize_system():
    """
    Initializes and warms up all DSP models in memory BEFORE opening
    the ALSA hardware stream, guaranteeing zero startup lag on the Pi.
    """
    print("=" * 70)
    print("[1/4] Initializing Silero VAD (Gate 1)...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=0.50)
    dummy_multi = np.zeros((6, CHUNK_SAMPLES), dtype=np.float32)
    _ = vad.is_speech_active(dummy_multi)
    print("   ✓ Silero VAD ready & warmed up.")

    print("[2/4] Initializing SRP Localizer (Gate 2)...")
    localizer = SRPLocalizer(sample_rate=SAMPLE_RATE, radius=0.04625, par_threshold=1.15)
    _ = localizer.localize(dummy_multi)
    print("   ✓ SRP Localizer ready (360° steering phases precomputed).")

    print("[3/4] Initializing DAS Beamformer (Module 3)...")
    beamformer = DASBeamformer(sample_rate=SAMPLE_RATE, radius=0.04625)
    _ = beamformer.beamform(dummy_multi, target_angles=0.0)
    print("   ✓ DAS Beamformer ready (center-padding COLA active).")

    print("[4/4] Warming up ECAPA-TDNN Speaker Verifier (Gate 4)...")
    verifier = SpeakerVerifier(sample_rate=SAMPLE_RATE, similarity_threshold=0.50)
    dummy_mono=np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    _ = verifier.extract_embedding(dummy_mono)
    print("   ✓ Speaker Verifier ready (ONNX CPU session warm).")

    print("=" * 70)
    print("All 4 DSP stages hot in RAM. Ready for audio stream.")
    print("=" * 70)

    return vad, localizer, beamformer, verifier

def get_or_enroll_doctor(streamer: AudioStreamer, vad: SileroVAD, verifier: SpeakerVerifier) -> np.ndarray:
    """
    Ensures a valid 192-D Doctor biometric profile is ready in RAM.
    - If doctor_profile.npy exists: loads it in < 1ms.
    - If missing: triggers enrollment on the active stream (zero ALSA restarts).
    """
    if DOCTOR_PROFILE_PATH.exists():
        print(f"Loading Doctor profile from: {DOCTOR_PROFILE_PATH}")
        doctor_embedding = DoctorEnroller.load_profile(DOCTOR_PROFILE_PATH)
        print("   ✓ Doctor profile loaded into RAM successfully.")
        return doctor_embedding

    print("\nNo Doctor profile found on disk!")
    print("Starting live enrollment on active audio stream...")
    enroller = DoctorEnroller(verifier=verifier, vad=vad, sample_rate=SAMPLE_RATE, target_speech_sec=3.5)
    doctor_embedding = enroller.enroll_from_stream(streamer)
    DoctorEnroller.save_profile(doctor_embedding, DOCTOR_PROFILE_PATH)
    print("   ✓ Doctor enrolled and profile persisted to disk.\n")
    return doctor_embedding

def run_live_pipeline():
    # Warm up all DSP models in RAM before opening hardware
    vad, localizer, beamformer, verifier = initialize_system()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TARGET SPEAKER EXTRACTION (TSE) PIPELINE ACTIVE (0.5s Chunks)")
    print("=" * 80 + "\n")

    frame_count = 0
    raw_recording = []
    doctor_recording = []   # Retains ONLY the confirmed Lead Surgeon audio
    active_speakers = {}    # {track_angle: remaining_ttl_frames}
    candidate_buffers = {}  # {track_angle: [chunks]} (1.5s buffer while searching for doctor)
    doctor_angle = None     # Points to the Doctor's angle (e.g. 45.0°), or None

    try:
        # Open ALSA audio capture stream
        with AudioStreamer(device=DEVICE, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION) as streamer:
            # Discard ADC startup power-on transient
            _ = streamer.read_chunk()

            # Ensure Doctor is enrolled; auto-enrolls on this active stream if profile is missing
            doctor_embedding = get_or_enroll_doctor(streamer, vad, verifier)

            while True:
                # Ingestion: Read live 0.5s multi-channel chunk (6, 8000)
                chunk = streamer.read_chunk()
                frame_count += 1
                timestamp = time.strftime("%H:%M:%S")

                # Visual Volume Meter
                peak_amp = int(np.max(np.abs(chunk)) * 32768)
                bars = min(int(peak_amp / 600), 12)
                meter = "█" * bars + "░" * (12 - bars)

                # --------------------------------
                # Gate 1: Voice Activity Detection
                # --------------------------------
                t_vad = time.perf_counter()
                is_speech = vad.is_speech_active(chunk)
                vad_ms = (time.perf_counter() - t_vad) * 1000

                if not is_speech:
                    # On silence, decremet tracks, if doctor paused > 2.0s, reset lock
                    for tracked_ang in list(active_speakers.keys()):
                        active_speakers[tracked_ang] -= 1
                        if active_speakers[tracked_ang] <= 0:
                            del active_speakers[tracked_ang]
                            candidate_buffers.pop(tracked_ang, None)
                            if doctor_angle == tracked_ang:
                                doctor_angle = None

                    print(
                        f"[{timestamp}] #{frame_count:04d} | "
                        f"Peak: {peak_amp:5d} [{meter}] | "
                        f"VAD: SILENCE ({vad_ms:4.1f}ms) ──► Gate 1 ABORT"
                    )
                    continue 

                # ---------------------------------------
                # Gate 2: Direction of Arrival (SRP-PHAT)
                # ---------------------------------------
                t_loc = time.perf_counter()
                detected_angles, _ = localizer.localize(chunk)
                loc_ms = (time.perf_counter() - t_loc) * 1000

                # Update active speaker tracks (±15° tolerance for walking)
                matched_tracks = set()

                for ang in detected_angles:
                    matched_track = None
                    for tracked_ang in active_speakers:
                        circ_diff = min(abs(ang - tracked_ang), 360.0 - abs(ang - tracked_ang))
                        if circ_diff <= TOLERANCE_DEG:
                            matched_track = tracked_ang
                            break 

                    if matched_track is not None:
                        # Existing speaker: refresh TTL to 4 (2.0s)
                        active_speakers[matched_track] = MAX_TTL_FRAMES
                        matched_tracks.add(matched_track)
                    else:
                        # New speaker angle: create new track and ECAPA candidate buffer
                        active_speakers[ang] = MAX_TTL_FRAMES
                        candidate_buffers[ang] = []
                        matched_tracks.add(ang)

                # Decrement TTL for tracks that were silent this chunk
                for tracked_ang in list(active_speakers.keys()):
                    if tracked_ang not in matched_tracks:
                        active_speakers[tracked_ang] -= 1
                        if active_speakers[tracked_ang] <= 0:
                            del active_speakers[tracked_ang]
                            candidate_buffers.pop(tracked_ang, None)
                            if doctor_angle == tracked_ang:
                                doctor_angle = None

                # If no speakers are currently active, skip beamforming
                if not active_speakers:
                    continue

                # -------------------------------------------------------------
                # Module 3: Dynamic Multi-Target Beamforming (DAS + Soft Wiener)
                # -------------------------------------------------------------
                t_beam = time.perf_counter()
                target_angles_to_steer = list(active_speakers.keys())
                enhanced_streams = beamformer.beamform(chunk, target_angles=target_angles_to_steer)
                beam_ms = (time.perf_counter() - t_beam) * 1000

                # Save raw reference channel (Mic 0) for diagnostics
                raw_recording.append(chunk[0].copy())

                # ---------------------------------
                # Gate 4: Target Speaker Extraction
                # ---------------------------------
                t_veri = time.perf_counter()
                ecapa_checked = False

                for angle, stream in zip(target_angles_to_steer, enhanced_streams):
                    if angle in candidate_buffers:
                        # Accumulate candidate stream (aiming for 1.5s = 3 chunks)
                        candidate_buffers[angle].append(stream.copy())

                        if len(candidate_buffers[angle]) >= 3:
                            cand_audio = np.concatenate(candidate_buffers[angle])
                            is_doc, score = verifier.verify(cand_audio, doctor_embedding)
                            ecapa_checked = True 

                            if is_doc:
                                # Doctor confirmed! Lock angle & flush buffered audio
                                doctor_angle = angle
                                doctor_recording.extend(candidate_buffers[angle])

                                print(
                                    f"[{timestamp}] DOCTOR LOCKED at {angle:.0f}° "
                                    f"(Score: {score:.3f} >= {verifier.threshold}) ──► Audio Flushed"
                                )
                            else:
                                print(
                                    f"[{timestamp}] AUXILIARY STAFF at {angle:.0f}° "
                                    f"(Score: {score:.3f} < {verifier.threshold}) ──► Suppressed"
                                )

                            # Evict from candidates so ECAPA does NOT run again on this track
                            candidate_buffers.pop(angle, None)

                    elif angle == doctor_angle:
                        # Confirmed Doctor Stream: zero ECAPA overhead
                        doctor_recording.append(stream.copy())

                veri_ms = (time.perf_counter() - t_veri) * 1000 if ecapa_checked else 0.0

                # Live Pipeline Telemetry
                total_latency_ms = vad_ms + loc_ms + beam_ms + veri_ms
                doc_status = f"{doctor_angle:.0f}°" if doctor_angle is not None else "Searching..."
                angles_str = ", ".join(f"{a:.0f}°" for a in target_angles_to_steer)

                print(
                    f"[{timestamp}] #{frame_count:04d} | "
                    f"Peak: {peak_amp:5d} [{meter}] | "
                    f"LOC: [{angles_str}] ({loc_ms:4.1f}ms) | "
                    f"BEAM: {len(enhanced_streams)} stream(s) ({beam_ms:4.1f}ms) | "
                    f"DOCTOR: {doc_status} | "
                    f"Latency: {total_latency_ms:4.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n🛑 Stopping live pipeline (Ctrl+C caught)...")
    except Exception as e:
        print(f"\n❌ Pipeline runtime error: {e}")
    finally:
        print("\n" + "=" * 70)
        print("PIPELINE SHUTDOWN — SAVING DIAGNOSTIC AUDIO")
        print("=" * 70)

        # Save raw reference audio (Mic 0)
        if raw_recording:
            raw_full = np.concatenate(raw_recording)
            raw_path = OUTPUT_DIR / "live_raw.wav"
            wavfile.write(str(raw_path), SAMPLE_RATE, (raw_full * 32767).astype(np.int16))
            raw_dur = len(raw_full) / SAMPLE_RATE 
            print(f"   ✓ Raw Reference:        {raw_path} ({raw_dur:.1f}s)")

        # Save isolated Lead Surgeon audio
        if doctor_recording:
            doc_full = np.concatenate(doctor_recording)
            doc_path = OUTPUT_DIR / "live_target_doctor.wav"
            wavfile.write(str(doc_path), SAMPLE_RATE, (doc_full * 32767).astype(np.int16))
            doc_dur = len(doc_full) / SAMPLE_RATE
            print(f"   ✓ Target Doctor Audio:  {doc_path} ({doc_dur:.1f}s)")
        else:
            print("   ⚠️ No Doctor audio was captured during this session.")

        print("=" * 70 + "\n")

if __name__ == "__main__":
    run_live_pipeline()











                
                




