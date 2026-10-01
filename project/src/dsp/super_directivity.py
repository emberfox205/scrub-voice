import os 
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# SETUP & ARRAY GEOMETRY
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "dsp"
DATA_FOLDER = DATA_DIR / "direction_60_120_0_0SNR_0Diffuse"
OUTPUT_FOLDER = DATA_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


RADIUS = 0.04          # 4 cm radius
SPEED_OF_SOUND = 343.0 # m/s

# 8 Mics: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
mic_angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])

# Load all 8 microphone .wav files
audio_channels = []
fs = None 

for mic_id in range(1, 9):
    file_path = os.path.join(DATA_FOLDER, f"mic_0{mic_id}.wav")
    fs, data = wav.read(file_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    audio_channels.append(data)

audio_matrix = np.array(audio_channels)
num_mics, total_samples = audio_matrix.shape
duration_sec = total_samples / fs

print(f"✅ Successfully loaded {num_mics} microphone channels at {fs} Hz!")

# ==========================================
# PRECOMPUTE STEERING PHASES (100 - 3500 Hz)
# ==========================================
FRAME_SIZE = 2048  # Fine frequency resolution for 44.1 kHz
HOP_SIZE = 512
angles_deg = np.arange(0, 360, 1)   # Search grid: 1° resolution
angles_rad = np.deg2rad(angles_deg)

# Physical delay compensation
tau = (RADIUS / SPEED_OF_SOUND) * np.cos(angles_rad[:, None] - mic_angles[None, :])
freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / fs)

# Coherent speech band for localization
LOC_MASK = (freqs >= 100) & (freqs <= 3500)
freqs_loc = freqs[LOC_MASK]

steering_phase = np.exp(-1j * 2 * np.pi * freqs_loc[None, None, :] * tau[:, :, None])

# ==========================================
# MULTI-SOURCE DETECTION PER FRAME (Find ALL Peaks)
# ==========================================
num_frames = max(0, 1 + (total_samples - FRAME_SIZE) // HOP_SIZE)
detected_angles_list = []

# Adaptive VAD: only evaluate active speech frames (top 60% energy)
frame_energies = np.array([np.mean(audio_matrix[0, f*HOP_SIZE:f*HOP_SIZE+FRAME_SIZE]**2) for f in range(num_frames)])
energy_threshold = np.percentile(frame_energies, 40)

print("Scanning audio frames for MULTIPLE simultaneous speakers...")
for frame_idx in range(num_frames):
    if frame_energies[frame_idx] < energy_threshold:
        continue
    
    start = frame_idx * HOP_SIZE
    frame = audio_matrix[:, start:start + FRAME_SIZE]
    
    # 1. FFT on localization band
    X = np.fft.rfft(frame, axis=1)[:, LOC_MASK]

    # 2. PHAT Whitening
    X_phat = X / (np.abs(X) + 1e-12)
    
    # 3. Steered beamforming: Y shape (360 angles, N_loc_freqs)
    Y = np.sum(steering_phase * X_phat[None, :, :], axis=1)
    
    # 4. Steered Response Power
    frame_power = np.sum(np.abs(Y)**2, axis=1)
    
    # FIX 1: Circularly pad frame_power by 30° so 0° is not on the edge
    power_padded = np.concatenate([frame_power[-30:], frame_power, frame_power[:30]])
    
    # FIX 2: Detect ALL peaks in this frame (allows multiple speakers per frame!)
    f_peaks, _ = find_peaks(power_padded, distance=25, prominence=np.max(frame_power)*0.08)
    
    # Record every detected speaker in this frame
    for p in f_peaks:
        ang = (p - 30) % 360
        detected_angles_list.append(ang)

detected_angles_list = np.array(detected_angles_list)
print(f"Total candidate source detections across all frames: {len(detected_angles_list)}")

# =========
# PLOTTING
# =========
# Group detected angles into 5-degree discrete bins
angle_bins = np.arange(0, 365, 5)
counts, bin_edges = np.histogram(detected_angles_list, bins=angle_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Circular boundary unification (merge 355°-360° with 0°-5° so 0° reaches full height!)
total_0deg_votes = counts[0] + counts[-1]
counts[0] = total_0deg_votes
counts[-1] = total_0deg_votes

# Normalize power between 0 and 1.0
normalized_votes = counts / np.max(counts)

plt.figure(figsize=(10, 6))

# Plot discrete dots connected by a line
plt.plot(bin_centers, normalized_votes, 'o-', color='darkblue', linewidth=1.5, markersize=6, label="P(q) Direction Power")

# Mark Ground Truth Lines
plt.axvline(x=60, color='orange', linestyle='--', linewidth=2, label="Source 1 (60°)")
plt.axvline(x=120, color='purple', linestyle='--', linewidth=2, label="Source 2 (120°)")
plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label="Source 3 (0°)")
plt.axvline(x=360, color='green', linestyle='--', linewidth=2)

# Find top peaks across the full circle (using circular wrap for peak detector)
votes_padded = np.concatenate([normalized_votes[-6:], normalized_votes, normalized_votes[:6]])
peaks_padded, _ = find_peaks(votes_padded, distance=5, height=0.20, prominence=0.10)

detected_peaks = []
ax = plt.gca()

for p in peaks_padded:
    idx = (p - 6) % len(bin_centers)
    ang = bin_centers[idx]
    if not any(abs(ang - existing) < 15 or abs(ang - existing) > 345 for existing in detected_peaks):
        detected_peaks.append(ang)
        # Draw red rectangle around the peak
        rect = plt.Rectangle((ang - 6, 0.0), 12, 1.05, linewidth=2.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.title("Sound Source Localization (Multi-Source Detection per Frame)", fontsize=14, fontweight='bold')
plt.xlabel("Azimuth Angle (Degrees)", fontsize=12)
plt.ylabel("Normalized Power P(q)", fontsize=12)
plt.xlim(-10, 370)
plt.ylim(0.0, 1.15)
plt.xticks(np.arange(0, 361, 30))
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

print("\n🎯 Final Sound Sources Localized:")
for ang in sorted(detected_peaks):
    print(f"   👉 {ang:.0f}°")

# =========================================
# STEP 3: BEAMFORMING (Super-Directive/DAS)
# =========================================

# target directions that we want to steer towards
target_angles_deg = sorted(detected_peaks)
target_angles_rad = np.deg2rad(target_angles_deg)

# For listening and audio reconstruction, use ALL frequency bins (From 0 Hz up to Nyquist 22,050 hz)
full_freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / fs)  # 1025 frequency bins

num_targets = len(target_angles_rad)

# Microphone coordinates
mic_x = RADIUS * np.cos(mic_angles)
mic_y = RADIUS * np.sin(mic_angles)

# Distance matrix btw every mic pair i and j: shape(8, 8)
dist_matrix = np.sqrt((mic_x[:, None] - mic_x[None, :]) ** 2 + (mic_y[:, None] - mic_y[None, :]) ** 2)

# WNG Threshold from Slide 13 (e.g. gamma = -6 dB to prevent mic noise amplification)
GAMMA_DB = -6.0
GAMMA_LINEAR = 10.0 ** (GAMMA_DB / 10.0)  # Minimum allowed WNG
I = np.eye(num_mics, dtype=np.complex64)

# Target arrival delays: shape (3 targets, 8 mics)
target_tau = (RADIUS / SPEED_OF_SOUND) * np.cos(target_angles_rad[:, None] - mic_angles[None, :])

# Array to store Super-Directive weights: shape (3 targets, 8 mics, 1025 frequencies)
beam_weights = np.zeros((num_targets, num_mics, len(full_freqs)), dtype=np.complex64)

def compute_mvdr_weights(Gamma_mat, d_vec, eps):
  """Computes MVDR weights with diagonal loading epsilon."""
  inv_R = np.linalg.inv(Gamma_mat + eps * I)
  inv_R_d = inv_R @ d_vec
  w = inv_R_d / (np.conj(d_vec) @ inv_R_d)
  return w

for k_freq, f in enumerate(full_freqs):
  # Diffuse coherence matrix Gamma(f) = sinc(2*pi*f*d/c)
  Gamma = np.sinc(2.0 * f * dist_matrix / SPEED_OF_SOUND).astype(np.complex64)
  for k_target in range(num_targets):
    d = np.exp(-1j * 2 * np.pi * f * target_tau[k_target, :])
    # 1. Try unconstrained (epsilon = 0)
    w = compute_mvdr_weights(Gamma, d, eps=1e-6)
    wng = 1.0 / (np.real(np.conj(w) @ w) + 1e-12)
    # 2. If WNG < gamma, find the optimal epsilon via bisection search
    if wng < GAMMA_LINEAR:
      low_eps, high_eps = 1e-6, 1.0
      for _ in range(8):  # 8 iterations reaches exact precision in 0.001s
        mid_eps = 0.5 * (low_eps + high_eps)
        w_test = compute_mvdr_weights(Gamma, d, eps=mid_eps)
        wng_test = 1.0 / (np.real(np.conj(w_test) @ w_test) + 1e-12)
        if wng_test < GAMMA_LINEAR:
          low_eps = mid_eps  # Needs more loading
        else:
          high_eps = mid_eps  # Can use less loading
      w = compute_mvdr_weights(Gamma, d, eps=high_eps)
    # Store conjugate weights for w^H * X
    beam_weights[k_target, :, k_freq] = np.conj(w)


# Hanning window for smooth audio reconstruction (no clicks / pops)
window = np.hanning(FRAME_SIZE)

# Tensor to store the 3 filtered beams in Time-Frequency domain
# Shape (3 beams, 1025 frequencies, num_frames)
Y_stft = np.zeros((num_targets, len(full_freqs), num_frames), dtype=np.complex64)

print("Applying beamforming filters across all frames...")

for frame_idx in range(num_frames):
    start = frame_idx * HOP_SIZE
    frame = audio_matrix[:, start:start + FRAME_SIZE]

    # 1. Apply smooth window to all 8 microphones
    frame_windowed = frame * window[None, :]

    # 2. Real FFT (0 to 22050 hz) - Natural speech, no PHAT
    X = np.fft.rfft(frame_windowed, axis=1)     # Shape (8 mics, 1025 freqs)

    # 3. Apply the filter: Y = sum(w * X) across 8 mics
    # beam_weights shape: (3, 8, 1025), X[None, :, :] shape: (1, 8, 1025)
    # Summing over axis=1 (the 8 mics) gives shape: (3 beams, 1025 freqs)
    Y_frame = np.sum(beam_weights * X[None, :, :], axis=1)

    # Store in output tensor
    Y_stft[:, :, frame_idx] = Y_frame

output_len = (num_frames - 1) * HOP_SIZE + FRAME_SIZE 
reconstructed_audio = np.zeros((num_targets, output_len), dtype=np.float32)
window_sum = np.zeros(output_len, dtype=np.float32)

print("Synthesizing audio back to time domain (iSTFT)...")

# 1. Overlap-Add reconstruction
for frame_idx in range(num_frames):
    start = frame_idx * HOP_SIZE

    # Convert 1025 freq bins back into 2048 time samples (iFFT)
    time_frame = np.fft.irfft(Y_stft[:, :, frame_idx], n=FRAME_SIZE, axis=1)

    # Add to output buffer with windowing
    reconstructed_audio[:, start : start + FRAME_SIZE] += time_frame * window
    window_sum[start : start + FRAME_SIZE] += window**2

# Normalize by window sum for perfect smooth reconstruction
reconstructed_audio /= np.maximum(window_sum, 1e-12)

# 2. Save the 3 Output .wav files
print("\n--- Step 3 Complete! Saving 3 Audio Beams ---")
saved_files = []

for k in range(num_targets):
    ang = target_angles_deg[k]
    filename = f"SD_{ang:.0f}deg.wav"
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    # Normalize audio amplitude to avoid clipping
    audio_stream = reconstructed_audio[k]
    max_peak = np.max(np.abs(audio_stream))
    if max_peak > 0:
        audio_stream = (audio_stream / max_peak) * 0.95

    # Save as 16-bit PCM WAV (playable in any audio player)
    wav.write(file_path, fs, (audio_stream * 32767).astype(np.int16))
    saved_files.append(file_path)
    print(f"   💾 Saved Beam {k+1} ({ang:.0f}°): {filename}")
