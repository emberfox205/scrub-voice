import onnxruntime as ort
import torchaudio
import torch
import numpy as np
from torch import Tensor

def hz_to_erb(freq):
    return 21.4 * torch.log10(1.0 + 0.00437 * freq)

def erb_to_hz(erb):
    return (10 ** (erb / 21.4) - 1.0) / 0.00437

def make_erb_filterbank(sample_rate, n_fft, erb_bins, low_freq=0.0, high_freq=None):
    if high_freq is None:
        high_freq = sample_rate / 2

    erb_edges = torch.linspace(
        hz_to_erb(torch.tensor(low_freq)),
        hz_to_erb(torch.tensor(high_freq)),
        erb_bins + 2,
    )

    hz_edges = erb_to_hz(erb_edges)
    fft_freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)

    filterbank = torch.zeros(erb_bins, n_fft // 2 + 1)

    for i in range(erb_bins):
        left = hz_edges[i]
        center = hz_edges[i + 1]
        right = hz_edges[i + 2]

        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)

        filterbank[i] = torch.minimum(rising, falling).clamp_min(0.0)
    return filterbank

def erb_synthesis_matrix(filterbank: Tensor, eps: float = 1e-8) -> Tensor:
    return filterbank.T / filterbank.sum(dim=0, keepdim=True).T.clamp_min(eps)

def apply_erb_gains(input_spec: Tensor, gains: Tensor, synthesis_matrix: Tensor) -> Tensor:
    if gains.ndim != 4 or gains.shape[1] != 1:
        raise ValueError("gains must have shape [B, 1, T, E]")
        
    frequency_gain = gains[:, 0] @ synthesis_matrix.T    # -> [B, T, F]
    
    return input_spec.transpose(1, 2) * frequency_gain.to(input_spec.dtype)

def apply_deep_filter(stage1_spec: Tensor, coefficients: Tensor, alpha: Tensor, df_bins: int = 96, df_order: int = 5, lookahead: int = 0) -> Tensor:
    B, T, F = stage1_spec.shape

    # Complex coefficients
    c = torch.view_as_complex(coefficients.contiguous())

    # Low-frequency bands of Stage 1 output (Y^G)
    yg_low = stage1_spec[:, :, :df_bins]

    # Casual/lookahead padding
    pad_front = df_order - 1 - lookahead
    pad_back = lookahead
    pad_front_tensor = torch.zeros((B, pad_front, df_bins), dtype=yg_low.dtype, device=yg_low.device)
    pad_back_tensor = torch.zeros((B, pad_back, df_bins), dtype=yg_low.dtype, device=yg_low.device) if pad_back > 0 else None

    if pad_back_tensor is not None:
        padded = torch.cat([pad_front_tensor, yg_low, pad_back_tensor], dim=1)
    else:
        padded = torch.cat([pad_front_tensor, yg_low], dim=1)

    # Gather delayed frames: tap n corresponds to lag n
    frames = torch.stack([padded[:, df_order - 1 - n : df_order - 1 - n + T, :] for n in range(df_order)], dim=-1)  # [B, T, df_bins, df_order=5]

    # Filtered spectrogram Y^DF0
    y_df0 = (c * frames).sum(dim=-1)  # [B, T, df_bins]

    # Convex combination with alpha: α · Y^DF0 + (1 - α) · Y^G
    y_df = alpha * y_df0 + (1.0 - alpha) * yg_low

    # Full spectrum: low-frequency bins enhanced, high-frequency bins pass through from Y^G
    final_spec = stage1_spec.clone()
    final_spec[:, :, :df_bins] = y_df

    return final_spec

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ONNX_MODEL_PATH = "src/dsp/denoise/deepfilternet.onnx"
INPUT_WAV = "src/dsp/denoise/audio/noise_audio.wav"
OUTPUT_WAV = "src/dsp/denoise/audio/clean_audio.wav"

# Ensure PyTorch uses only 1 thread to prevent Raspberry Pi CPU thrashing
torch.set_num_threads(1)

# 1. Load ONNX Session (using CPU provider optimized for ARM)
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=options, providers=['CPUExecutionProvider'])

# 2. Setup DSP
sample_rate = 16000
n_fft = 512
hop_length = 128
win_length = 512
erb_bins = 32
df_bins = 96

window = torch.hann_window(win_length)
filterbank = make_erb_filterbank(sample_rate=sample_rate, n_fft=n_fft, erb_bins=erb_bins) # [32, 257]
synthesis_matrix = erb_synthesis_matrix(filterbank) # [257, 32]

# 3. Load Audio
noisy_wav, sr = torchaudio.load(INPUT_WAV)
if sr != sample_rate:
    noisy_wav = torchaudio.functional.resample(noisy_wav, orig_freq=sr, new_freq=sample_rate)

if noisy_wav.shape[0] > 1:
    noisy_wav = noisy_wav.mean(dim=0, keepdim=True)

# 4. Feature Extraction (STFT -> ERB & Complex)
# STFT
input_spec = torch.stft(noisy_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

# ERB
input_erb = torch.matmul(input_spec.abs().square().transpose(1, 2), filterbank.T)
input_erb = torch.log10(input_erb + 1e-7).unsqueeze(1) # [1, 1, T, 32]

# Complex Input
complex_raw = torch.view_as_real(input_spec[:, :df_bins, :]).permute(0, 3, 2, 1) # [1, 2, T, 96]
# Note: Apply CausalEmaUnitNorm here if your model expects it:
# complex_in = complex_ema_norm(complex_raw)
complex_in = complex_raw # Assuming raw for demo

# 5. Run ONNX Inference
inputs = {
    'input_erb': input_erb.numpy().astype(np.float32),
    'complex_in': complex_in.numpy().astype(np.float32)
}
outputs = session.run(None, inputs)

erb_gains = torch.from_numpy(outputs[0])
df_coefficients = torch.from_numpy(outputs[1])
alpha = torch.from_numpy(outputs[2])

# 6. Apply Filter & Inverse STFT
stage1_spec = apply_erb_gains(input_spec, erb_gains, synthesis_matrix)
final_spec = apply_deep_filter(stage1_spec=stage1_spec, coefficients=df_coefficients, alpha=alpha, df_bins=df_bins, df_order=5, lookahead=1)
enh_wav = torch.istft(final_spec.transpose(1, 2), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

# 7. Save
torchaudio.save(OUTPUT_WAV, enh_wav, sample_rate)
print(f"Saved enhanced audio to {OUTPUT_WAV}")
