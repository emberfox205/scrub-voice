import torch.nn as nn
from torch import Tensor, nn
import torch.nn.functional as F
import torch
from typing import Tuple


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, t, d = x.shape
    if d % groups != 0:
        raise ValueError("Feature dimension must be divisible by groups")

    features_per_group = d // groups
    x = x.view(b, t, groups, features_per_group)
    x = x.transpose(2, 3).contiguous()
    x = x.view(b, t, d)

    return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: Tuple[int, int] = (3, 2), stride: Tuple[int, int] = (1, 1), lookahead: int = 0) -> None:
        super().__init__()

        kt, kf = kernel_size

        if lookahead < 0 or lookahead > kt - 1:
            raise ValueError("lookahead must satisfy 0 <= lookahead <= kt - 1")

        time_left = kt - 1 - lookahead
        time_right = lookahead
        self.pad = (
            kf // 2, # left
            kf - 1 - kf // 2, # right
            time_left, # top
            time_right, # bottom
        )

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.pad)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class SeparableTConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, scale_factor: int = 2, lookahead: int = 0) -> None:
        super().__init__()
        self.scale_factor = scale_factor

        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size=(3, 2), lookahead=lookahead)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(
            x, 
            scale_factor=(1, self.scale_factor), # [width, height] ~ [time, frequency]
            mode="nearest"
        )
        return self.conv(x)

class GroupedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int = 8, shuffle: bool = True) -> None:
        super().__init__()

        if in_features % groups:
            raise ValueError("in_features must be divisible by groups")

        if out_features % groups:
            raise ValueError("out_features must be divisible by groups")

        self.groups = groups
        self.shuffle = shuffle

        self.in_per_group = in_features // groups
        self.out_per_group = out_features // groups

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    self.in_per_group,
                    self.out_per_group,
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]

        chunks = x.split(self.in_per_group, dim=-1)

        output_chunks = []

        for layer, chunk in zip(self.layers, chunks):
            y = layer(chunk)
            output_chunks.append(y)

        x = torch.cat(output_chunks, dim=-1)

        if self.shuffle:
            x = channel_shuffle(x, self.groups)

        return x

class GroupedGRU(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 512, groups: int = 8, shuffle: bool = True) -> None:
        super().__init__()
        if input_size % groups:
            raise ValueError("input_size must be divisible by groups")
        if hidden_size % groups:
            raise ValueError("hidden_size must be divisible by groups")

        self.groups = groups
        self.shuffle = shuffle

        self.input_per_group = input_size // groups
        self.hidden_per_groups = hidden_size // groups

        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.input_per_group,
                    hidden_size=self.hidden_per_groups,
                    batch_first=True,
                )
                for _ in range(groups)
            ]
        )


    def forward(self, x: Tensor) -> Tensor:
        # [B, T, D]
        chunks = x.split(self.input_per_group, dim=-1)
        outputs = []

        for gru, chunk in zip(self.grus, chunks):
            # GRU returns [output, hidden]
            y, _ = gru(chunk)
            outputs.append(y)

        x = torch.cat(outputs, dim=-1)

        if self.shuffle:
            x = channel_shuffle(x, self.groups)

        return x

class GroupedGRUStack(nn.Module):
    def __init__(self, size: int = 512, groups: int = 8, num_layers: int = 3) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                GroupedGRU(
                    input_size=size,
                    hidden_size=size,
                    groups=groups,
                    shuffle=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x

class PConv(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()

        self.conv = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)