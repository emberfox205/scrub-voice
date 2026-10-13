import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from .ERBEncoder import ERBEncoder
from .ERBDecoder import ERBDecoder
from .DFNet import DFNet

class DeepFilterNetDNN(nn.Module):
    """
    rb_features:
        [B, 1, T, B_erb]

    complex_features:
        [B, 1, T, F_df]

    erb_gains:
        [B, 1, T, B_erb]
    
    df_coefficients:
        [B, T, F_df, N, 2]
    """

    def __init__(self, erb_bins: int = 32, df_bins: int = 96, df_order: int = 5, channels: int = 64, hidden_size: int = 512, groups: int = 8, conv_lookahead: int = 2) -> None:
        super().__init__()
        
        self.encoder = ERBEncoder(
            erb_bins=erb_bins,
            channels=channels,
            hidden_size=hidden_size,
            groups=groups,
            conv_lookahead=conv_lookahead,
        )

        self.decoder = ERBDecoder(
            channels=channels,
            erb_bins=erb_bins,
            hidden_size=hidden_size
        )

        self.df_net = DFNet(
            df_bins=df_bins,
            df_order=df_order,
            channels=channels,
            hidden_size=hidden_size,
            groups=groups,
        )

    def forward(self, erb_features: Tensor, complex_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Stage 1:
        e0, e1, e2, e3, embedding = self.encoder(erb_features)
        gains = self.decoder(embedding, e0, e1, e2, e3)

        # Stage 2:
        df_coefficients, alpha = self.df_net(complex_features, embedding)

        return gains, df_coefficients, alpha