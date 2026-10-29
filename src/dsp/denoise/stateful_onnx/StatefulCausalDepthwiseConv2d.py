import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple


class StatefulCausalDepthwiseConv2d(nn.Module):
    """
    Causal Depthwise 2D Convolution with stateful temporal frame caching.
    
    Supports arbitrary chunk sizes (T >= 1, e.g., T=32 for 0.5s chunks).
    Instead of prepending zero-padding along the time axis, it maintains a history cache
    of the past (kt - 1 = 2) frames across consecutive streaming chunks.
    """

    def __init__(
        self,
        dwconv: nn.Conv2d,
        norm: nn.Module = nn.Identity(),
        act: nn.Module = nn.Identity(),
        pad_freq: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.dwconv = dwconv
        self.norm = norm
        self.act = act
        self.pad_freq = pad_freq  # (pad_left, pad_right)

    @classmethod
    def from_causal_dwconv(cls, causal_dw: nn.Module) -> "StatefulCausalDepthwiseConv2d":
        """
        Factory method to wrap an existing CausalDepthwiseConv2d layer from DuoFilter.
        Extracts weights, norm, activation, and frequency padding.
        """
        pad_freq = (causal_dw.pad[0], causal_dw.pad[1]) if hasattr(causal_dw, "pad") else (1, 1)
        return cls(
            dwconv=causal_dw.dwconv,
            norm=causal_dw.norm if hasattr(causal_dw, "norm") else nn.Identity(),
            act=causal_dw.act if hasattr(causal_dw, "act") else nn.Identity(),
            pad_freq=pad_freq,
        )

    def forward(
        self, x_chunk: Tensor, cache: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Processes an audio chunk of T frames and updates the temporal frame cache.

        Args:
            x_chunk: Current time-frame tensor of shape [B, C, T, F] (e.g. T=32 for 0.5s).
            cache:   History cache of shape [B, C, 2, F] containing the past 2 frames.
            
        Returns:
            out_chunk:  Convolved and normalized output chunk of shape [B, C, T, F_out].
            next_cache: Updated history cache of shape [B, C, 2, F] (the last 2 frames).
        """
        # 1. Concatenate past 2 frames with current chunk along time (dim=2) -> [B, C, 2 + T, F]
        x_time = torch.cat([cache, x_chunk], dim=2)

        # 2. Update cache: keep the 2 most recent frames of the sequence
        next_cache = x_time[:, :, -2:, :]

        # 3. Pad frequency dimension only (time history is already in x_time)
        x_padded = F.pad(x_time, (self.pad_freq[0], self.pad_freq[1], 0, 0))

        # 4. Depthwise Conv -> GroupNorm -> Activation
        out_dw = self.dwconv(x_padded)  # Output time dimension is (2 + T - 3) + 1 = T
        out_chunk = self.act(self.norm(out_dw))

        return out_chunk, next_cache
