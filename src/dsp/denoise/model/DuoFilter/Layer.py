import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional
from src.dsp.denoise.model.DuoFilter.ERBFilterBank import ERBFilterBank
from src.dsp.denoise.model.DuoFilter.EMANorm import CausalEmaNorm


def get_groups(channels: int, max_groups: int = 4) -> int:
    """Finds the largest valid divisor of channels <= max_groups for GroupNorm"""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1

class CausalDepthwiseConv2d(nn.Module):
    """
    Causal Depthwise-Separable 2D Convolution (DeepFilterNet Stage 1 Heritage).
    Decouples spatial time-frequency filtering (groups=in_ch) from channel mixing (1x1 conv).
    Strictly causal in time via asymmetric padding (kt - 1 past, 0 future).
    """
    def __init__(self, channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 1), use_norm: bool = True, act: Optional[nn.Module] = nn.Mish(), bias: Optional[bool] = None):
        super().__init__()
        kt, kf = kernel_size
        self.pad = (kf // 2, kf // 2, kt - 1, 0)  # (pad_left, pad_right, pad_past, pad_future)

        if bias is None:
            bias = not use_norm

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=channels,
            bias=bias
        )

        self.norm = nn.GroupNorm(num_groups=get_groups(channels, 4), num_channels=channels) if use_norm else nn.Identity()
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.dwconv(F.pad(x, self.pad))))

class CausalDepthwiseConvTranspose2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int] = (1, 3),
        stride: Tuple[int, int] = (1, 2),
        use_norm: bool = True,
        act: Optional[nn.Module] = nn.Mish(),
        bias: Optional[bool] = None,
    ):
        super().__init__()
        kt, kf = kernel_size
        self.kt = kt
        if bias is None:
            bias = not use_norm

        self.dwtconv = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, kf // 2),
            output_padding = (0, 1),
            groups=channels,
            bias=bias,
        )

        self.norm = nn.GroupNorm(num_groups=get_groups(channels, 4), num_channels=channels) if use_norm else nn.Identity()
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.dwtconv(x)
        if self.kt > 1:
            out = out[:, :, :-(self.kt - 1), :]
        return self.act(self.norm(out))

class PConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_norm: bool = True,
        act: Optional[nn.Module] = nn.Mish(),
        bias: Optional[bool] = None,
    ):
        super().__init__()
        if bias is None:
            bias = not use_norm

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.GroupNorm(num_groups=get_groups(out_ch, 4), num_channels=out_ch) if use_norm else nn.Identity()
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))
