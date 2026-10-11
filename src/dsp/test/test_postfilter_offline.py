"""
Offline Listening Test: Enhance recorded beamformer audio with DeepFilterNet3
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Path resolution
test_dir = Path(__file__).resolve().parent
dsp_dir = test_dir.parent
repo_root = dsp_dir.parent.parent
if str(dsp_dir) not in sys.path:
    sys.path.insert(0, str(dsp_dir))

from source.postfilter import DeepFilterPostFilter

output_dir = repo_root / "data" / "dsp" / "output"

# The two stream files to enhance
FILES_TO_ENHANCE = [
    "live_beam_stream0.wav",
    "live_beam_stream1.wav"
]

def enhance_file(pf: DeepFilterPostFilter, filename: str):
    input_path = output_dir/filename
    if not input_path.exists():
        print(f"⚠️ Skipping {filename}: File not found at {input_path}")
        return

    # Output filename: e.g. live_beam_stream0_deepfiltered.wav
    stem = input_path.stem
    output_path = output_dir / f"{stem}_deepfiltered.wav"

    # 1. Read audio
    sr, audio_int16 = wavfile.read(str(input_path))
    audio_float32 = (audio_int16 / 32768.0).astype(np.float32)
    duration_sec = len(audio_float32) / sr
    print(f"\n🎧 Processing '{filename}' ({duration_sec:.2f}s, {sr} Hz)...")

    # 2. Process in 0.5s chunks (exactly like in real-time streaming)
    chunk_size = int(0.5 * sr)
    cleaned_chunks = []

    for start in range(0, len(audio_float32), chunk_size):
        chunk = audio_float32[start : start + chunk_size]

        # If the final chunk is shorter than 8000 samples, pad it with silence!
        if len(chunk) < chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=np.float32)
            padded_chunk[:len(chunk)] = chunk
            chunk = padded_chunk

        cleaned_chunk = pf.enhance(chunk)
        cleaned_chunks.append(cleaned_chunk)

    cleaned_audio = np.concatenate(cleaned_chunks)

    # 3. Save enhanced WAV
    wavfile.write(str(output_path), sr, (cleaned_audio * 32767).astype(np.int16))
    print(f"   ✓ Saved: {output_path.name}")

def main():
    print("=" * 70)
    print("⏳ Initializing DeepFilterNet3 model...")
    pf = DeepFilterPostFilter(sample_rate=16000)
    print("✓ DeepFilterNet3 ready.")
    print("=" * 70)

    # Enhance each stream file
    for filename in FILES_TO_ENHANCE:
        enhance_file(pf, filename)

    print("🎉 Finished enhancing all streams!")

if __name__ == "__main__":
    main()