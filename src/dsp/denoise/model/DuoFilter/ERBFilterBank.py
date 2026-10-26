import torch
from torch import Tensor, nn
import math
from typing import Optional

class ERBFilterBank(nn.Module):
    """
    Psychoacoustically exact ERB Filterbank for Band Merging (BM) and Splitting (BS).
    - Low-band  (0 - 2 kHz, bins 0..63): 1:1 Identity mapping (64 linear bins).
    - High-band (2 - 8 kHz, bins 64..256): 193 linear bins -> 64 ERB filters.
    - Total Subbands: 128 (64 linear + 64 ERB).
    """
    def __init__(self, n_fft: int = 512, fs: int = 16000, n_linear: int = 64, n_erb: int = 64):
        super().__init__()
        self.n_linear = n_linear
        total_bins = n_fft // 2 + 1
        high_bins = total_bins - n_linear

        # Anchor frequencies with 1-bin overhang so linear bins 64 and 256 (Nyquist) are fully covered
        low_freq = ((n_linear - 1) / (n_fft // 2.0)) * (fs / 2.0)
        high_freq = ((total_bins) / (n_fft // 2.0)) * (fs / 2.0)

        erb_weights = self._build_true_erb_matrix(fs, n_fft, n_linear, high_bins, n_erb, low_freq, high_freq)
        self.register_buffer("bm_matrix", erb_weights)   # [64, 193]

        bs_weights = erb_weights.t() / (torch.sum(erb_weights.t(), dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("bs_matrix", bs_weights)    # [193, 64]

    def _freq_to_erb(self, freq: float) -> float:
        return 21.4 * math.log10(4.37 * (freq / 1000.0) + 1.0)

    def _erb_to_freq(self, erb: float) -> float:
        return (10.0 ** (erb / 21.4) - 1.0) / 0.00437

    def _build_true_erb_matrix(self, fs: int, n_fft: int, n_linear: int, n_high: int, n_filters: int, f_min: float, f_max: float) -> Tensor:
        erb_min = self._freq_to_erb(f_min)
        erb_max = self._freq_to_erb(f_max)
        erb_pts = torch.linspace(erb_min, erb_max, n_filters + 2)
        freq_pts = torch.tensor([self._erb_to_freq(e.item()) for e in erb_pts])

        bin_pts = (freq_pts * n_fft / fs) - n_linear
        weights = torch.zeros(n_filters, n_high)

        for i in range(n_filters):
            b_prev, b_curr, b_next = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
            for b in range(n_high):
                if b_prev <= b <= b_curr and b_curr > b_prev:
                    weights[i, b] = (b - b_prev) / (b_curr - b_prev)
                elif b_curr < b <= b_next and b_next > b_curr:
                    weights[i, b] = (b_next - b) / (b_next - b_curr)

        return weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)

    def bm(self, x: Tensor) -> Tensor:
        """Forward Band Merging: [B, ..., 257] -> [B, ..., 128]"""
        x_low = x[..., :self.n_linear]
        x_high = x[..., self.n_linear:]
        x_high_erb = torch.matmul(x_high, self.bm_matrix.t())
        return torch.cat([x_low, x_high_erb], dim=-1)

    def bs(self, x_sub: Tensor) -> Tensor:
        """Inverse Band Splitting: [B, ..., 128] -> [B, ..., 257]"""
        x_low = x_sub[..., :self.n_linear]
        x_high = x_sub[..., self.n_linear:]
        x_high_linear = torch.matmul(x_high, self.bs_matrix.t())
        return torch.cat([x_low, x_high_linear], dim=-1)

def wav_to_stft(wav: Tensor, n_fft: int = 512, hop_length: int = 256, win_length: int = 512, window: Optional[Tensor] = None) -> Tensor:
    if window is None:
        window = torch.hann_window(window_length=win_length, device=wav.device).pow(0.5)

    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )
    return spec.transpose(-1, -2)   # [B, T, F]


def stft_to_wav(spec: Tensor, n_fft: int = 512, hop_length: int = 256, win_length: int = 512, window: Optional[Tensor] = None, length: Optional[int] = None) -> Tensor:
    if window is None:
        window = torch.hann_window(window_length=win_length, device=spec.device).pow(0.5)

    # Cast to float precision to prevent ComplexHalf crash under torch.amp.autocast
    if spec.dtype not in (torch.complex64, torch.complex128):
        spec = spec.to(torch.complex64)
    if window.dtype not in (torch.float32, torch.float64):
        window = window.float()

    spec_transposed = spec.transpose(-1, -2).contiguous()  # [B, F, T]
    return torch.istft(
        spec_transposed,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length
    )
