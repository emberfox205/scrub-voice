import torch.nn as nn
from torch import Tensor
from .Layer import PConv, SeparableTConv2d, SeparableConv2d, GroupedLinear


class ERBDecoder(nn.Module):
    def __init__(self, channels: int = 64, erb_bins: int = 32, hidden_size: int = 512) -> None:
        super().__init__()

        bottleneck_freq = erb_bins // 8
        bottleneck_size = channels * bottleneck_freq

        self.channels = channels
        self.bottelneck_freq = bottleneck_freq

        self.linear = GroupedLinear(hidden_size, bottleneck_size)

        self.p3 = PConv(channels)
        self.p2 = PConv(channels)
        self.p1 = PConv(channels)
        self.p0 = PConv(channels)

        self.up3 = SeparableTConv2d(channels, channels, lookahead=0)
        self.up2 = SeparableTConv2d(channels, channels, lookahead=0)
        self.up1 = SeparableTConv2d(channels, channels, lookahead=0)

        self.final_conv = SeparableConv2d(channels, 1, kernel_size=(3, 2), lookahead=0)
        self.output_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, embedding: Tensor, x0: Tensor, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
        batch, time, _ = embedding.shape

        x = self.linear(embedding)
        x = self.dropout(x)

        x = x.reshape(batch, time, self.channels, self.bottelneck_freq)
        x = x.permute(0, 2, 1, 3).contiguous()

        p3_x3 = self.p3(x3)
        x = x + p3_x3

        x = self.up3(x)
        x = x + self.p2(x2)

        x = self.up2(x)
        x = x + self.p1(x1)

        x = self.up1(x)
        x = x + self.p0(x0)

        gains = self.final_conv(x)
        gains = self.output_activation(gains)

        return gains