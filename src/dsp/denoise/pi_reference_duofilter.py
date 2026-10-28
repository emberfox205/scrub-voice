import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from torch import Tensor

# ---------------------------------------------------------
# STFT & iSTFT Helper Functions (Square-Root Hann COLA)
# ---------------------------------------------------------
def wav_to_stft(wav: Tensor, n_fft: int = 512, hop_length: int = 256, win_length: int = 512, window: Tensor = None) -> Tensor:
    """
    Transforms audio waveform to STFT representation.
    Returns: Tensor of shape [B, 2, T, 257] (Real, Imag)
    """
    if window is None:
        window = torch.hann_window(win_length, device=wav.device).pow(0.5)

    # Complex STFT: [B, F=257, T]
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )
    # Transpose to [B, T, F] then stack real and imag -> [B, 2, T, F]
    spec = spec.transpose(-1, -2)
    return torch.stack([spec.real, spec.imag], dim=1)


def stft_to_wav(enhanced_stft: Tensor, n_fft: int = 512, hop_length: int = 256, win_length: int = 512, window: Tensor = None, length: int = None) -> Tensor:
    """
    Reconstructs audio waveform from enhanced STFT [B, 2, T, 257].
    Returns: Tensor of shape [B, Audio_Samples]
    """
    if window is None:
        window = torch.hann_window(win_length, device=enhanced_stft.device).pow(0.5)

    # Reconstruct complex spectrogram: [B, T, F] -> [B, F, T]
    enh_complex = torch.complex(enhanced_stft[:, 0].float(), enhanced_stft[:, 1].float())
    spec_transposed = enh_complex.transpose(-1, -2).contiguous()

    return torch.istft(
        spec_transposed,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.float(),
        center=True,
        length=length
    )


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ONNX_MODEL_PATH = "src/dsp/denoise/onnx/duofilter.onnx"
INPUT_WAV = "src/dsp/denoise/audio/live_beam_stream_148.0deg.wav"
OUTPUT_WAV = "src/dsp/denoise/audio/clean_audio.wav"

# DuoFilter Audio & STFT Parameters
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256    # 16 ms hop (50% overlap)
WIN_LENGTH = 512

# Prevent CPU thrashing on Raspberry Pi
torch.set_num_threads(1)

# 1. Load ONNX Session (configured for ARM Cortex-A CPU)
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=session_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 2. Square-Root Hann Window for analysis & synthesis
window = torch.hann_window(WIN_LENGTH).pow(0.5)

# 3. Load & Preprocess Audio
noisy_wav, sr = torchaudio.load(INPUT_WAV)
if sr != SAMPLE_RATE:
    noisy_wav = torchaudio.functional.resample(noisy_wav, orig_freq=sr, new_freq=SAMPLE_RATE)

if noisy_wav.shape[0] > 1:
    noisy_wav = noisy_wav.mean(dim=0, keepdim=True)

orig_length = noisy_wav.shape[-1]

# 4. Feature Extraction: Causal STFT -> [1, 2, T, 257]
# Channel 0: Yr (Real), Channel 1: Yi (Imag)
noisy_stft = wav_to_stft(noisy_wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window)

# 5. Run DuoFilter ONNX Inference
# All ERB Band Merging (BM), SFE, U-Net processing, Amplified CRM, and Band Splitting (BS) run inside the model
ort_inputs = {input_name: noisy_stft.numpy().astype(np.float32)}
ort_outputs = session.run([output_name], ort_inputs)

enhanced_stft = torch.from_numpy(ort_outputs[0])  # [1, 2, T, 257]

# 6. Reconstruct Time-Domain Waveform via iSTFT
enh_wav = stft_to_wav(
    enhanced_stft,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window=window,
    length=orig_length
)

# 7. Save Enhanced Output
torchaudio.save(OUTPUT_WAV, enh_wav, SAMPLE_RATE)
print(f"Saved enhanced audio to {OUTPUT_WAV}")