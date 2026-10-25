from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

CURRENT_DIR = Path(__file__).resolve().parent

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

class DeepFilterDenoiser:
    """DeepFilterNet ONNX inference engine optimized for Raspberry Pi / Edge CPU."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        erb_bins: int = 32,
        df_bins: int = 96,
        threads: int = 1,
    ):
        if model_path is None:
            model_path = CURRENT_DIR / "deepfilternet.onnx"
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.erb_bins = erb_bins
        self.df_bins = df_bins

        torch.set_num_threads(threads)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        self.window = torch.hann_window(self.win_length)
        self.filterbank = make_erb_filterbank(
            sample_rate=self.sample_rate, n_fft=self.n_fft, erb_bins=self.erb_bins
        )
        self.synthesis_matrix = erb_synthesis_matrix(self.filterbank)

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """Enhance mono audio (1D float32 or [C, T]). Returns 1D float32 np.ndarray."""
        orig_len = audio.shape[-1]
        wav_t = torch.from_numpy(audio.astype(np.float32))
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        elif wav_t.shape[0] > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)

        input_spec = torch.stft(
            wav_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        input_erb = torch.matmul(input_spec.abs().square().transpose(1, 2), self.filterbank.T)
        input_erb = torch.log10(input_erb + 1e-7).unsqueeze(1)
        complex_raw = torch.view_as_real(input_spec[:, : self.df_bins, :]).permute(0, 3, 2, 1)

        inputs = {
            "input_erb": input_erb.numpy().astype(np.float32),
            "complex_in": complex_raw.numpy().astype(np.float32),
        }
        outputs = self.session.run(None, inputs)

        erb_gains = torch.from_numpy(outputs[0])
        df_coefficients = torch.from_numpy(outputs[1])
        alpha = torch.from_numpy(outputs[2])

        stage1_spec = apply_erb_gains(input_spec, erb_gains, self.synthesis_matrix)
        final_spec = apply_deep_filter(
            stage1_spec=stage1_spec,
            coefficients=df_coefficients,
            alpha=alpha,
            df_bins=self.df_bins,
            df_order=5,
            lookahead=1,
        )
        enh_wav = torch.istft(
            final_spec.transpose(1, 2),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=orig_len,
        )
        return enh_wav.squeeze(0).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Standalone Demo / Benchmark
    # ---------------------------------------------------------
    ONNX_MODEL_PATH = str(CURRENT_DIR / "deepfilternet.onnx")
    INPUT_WAV = str(CURRENT_DIR / "audio" / "noise_audio.wav")
    OUTPUT_WAV = str(CURRENT_DIR / "audio" / "clean_audio.wav")

    denoiser = DeepFilterDenoiser(model_path=ONNX_MODEL_PATH)
    audio_data, sr = sf.read(INPUT_WAV, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sr != 16000:
        noisy_t = torchaudio.functional.resample(torch.from_numpy(audio_data), orig_freq=sr, new_freq=16000)
        audio_data = noisy_t.numpy()

    out_audio = denoiser.enhance(audio_data)
    sf.write(OUTPUT_WAV, out_audio, 16000)
    print(f"Saved enhanced audio to {OUTPUT_WAV}")
