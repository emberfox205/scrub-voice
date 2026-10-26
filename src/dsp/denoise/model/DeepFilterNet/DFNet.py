import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from .Layer import PConv, SeparableConv2d, GroupedLinear, GroupedGRUStack


class DFNet(nn.Module):
    def __init__(self, df_bins: int, df_order: int, channels: int=64, hidden_size: int = 512, groups: int=8) -> None:
        super().__init__()

        self.df_bins = df_bins
        self.df_order = df_order
        self.channels = channels

        self.conv0 = SeparableConv2d(2, channels, lookahead=1)
        self.conv1 = SeparableConv2d(channels, channels, stride=(1, 2), lookahead=0)
        
        self.projection = GroupedLinear(channels*(df_bins // 2), hidden_size, groups=groups)
        self.grus = GroupedGRUStack(size=hidden_size, groups=groups, num_layers=2)
        self.pconv = PConv(channels=channels, out_channels=df_order*2)
        
        self.output_coefs = nn.Sequential(nn.Linear(hidden_size, df_bins*df_order*2), nn.Tanh())
        self.output_alpha = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        
        self.merge = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, complex_features: Tensor, encoder_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        # x0: [B, C, T, F_DF]
        x0 = self.conv0(complex_features)

        # pconv(x0): [B, df_order * 2, T, F_DF] -> permute to [B, T, F_DF, df_order * 2]
        skip = self.pconv(x0).permute(0, 2, 3, 1)
        x = self.conv1(x0)

        batch, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch, time, channels * freq)

        x = self.projection(x)
        x = self.grus(x)
        x = self.dropout(x)

        x = torch.cat([x, encoder_embedding], dim=-1)
        x = self.merge(x)

        coefficients = self.output_coefs(x)
        coefficients = coefficients.reshape(batch, time, self.df_bins, self.df_order * 2)
        coefficients = coefficients + skip
        coefficients = coefficients.reshape(batch, time, self.df_bins, self.df_order, 2)
        
        alpha = self.output_alpha(x)  # [B, T, 1]

        return coefficients, alpha
