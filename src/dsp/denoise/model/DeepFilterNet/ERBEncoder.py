from torch import Tensor
import torch.nn as nn
from typing import Tuple
from .Layer import SeparableConv2d, GroupedLinear, GroupedGRUStack

class ERBEncoder(nn.Module):
    def __init__(self, erb_bins: int = 32, channels: int = 64, hidden_size: int = 512, groups: int = 8, conv_lookahead: int = 2) -> None:
        super().__init__()

        if erb_bins % 8:
            raise ValueError("erb_bins must be divisible by 8")

        self.erb_bins = erb_bins
        self.channels = channels
        self.conv_lookahead = conv_lookahead
        lookahead0 = 1 if conv_lookahead > 0 else 0
        lookahead1 = 1 if conv_lookahead > 1 else 0
        lookahead2 = 1 if conv_lookahead > 2 else 0
        self.conv0 = SeparableConv2d(1, channels, lookahead=lookahead0)
        self.conv1 = SeparableConv2d(channels, channels, stride=(1, 2), lookahead=lookahead1) # B -> B / 2
        self.conv2 = SeparableConv2d(channels, channels, stride=(1, 2), lookahead=lookahead2)
        self.conv3 = SeparableConv2d(channels, channels, stride=(1, 2), lookahead=0)
        self.glinear = GroupedLinear(in_features=channels*erb_bins//8, out_features=hidden_size, groups=groups)
        self.gru = GroupedGRUStack(size=hidden_size, groups=groups, num_layers=3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        batch, channels, time, freq = x3.shape
        x = x3.permute(0, 2, 1, 3)
        x = x.reshape(batch, time, channels*freq)

        x = self.glinear(x)
        embedding = self.gru(x)
        embedding = self.dropout(embedding)

        return x0, x1, x2, x3, embedding
