import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple

from src.dsp.denoise.stateful_onnx.StatefulCausalDepthwiseConv2d import StatefulCausalDepthwiseConv2d


class DuoFilterStreaming(nn.Module):
    """
    True Stateful Streaming DuoFilter Engine supporting arbitrary chunk sizes (T >= 1).
    Default target: 0.5s chunks (T=32 frames @ 16 kHz with hop_length=256).
    
    Architecture Highlights:
    1. Zero EMA Norm: Relies entirely on GroupNorm and LayerNorm across the chunk,
       eliminating long-decay variance lag and inter-speech volume squashing.
    2. 0.5s Chunk-Preserved Dynamics: With T=32 frames, GroupNorm has ample temporal
       context to compute stable variance across harmonics and pauses, preserving speech.
    3. Stateful Causal Convolutions: Uses StatefulCausalDepthwiseConv2d for enc0, enc1,
       enc2, enc3, and conv_glu to maintain continuous past 2-frame history caches.
    4. Stateful Recurrent Bottleneck: Passes temporal Uni-GRU hidden states across chunks.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.base.eval()

        # Wrap each causal depthwise conv with StatefulCausalDepthwiseConv2d
        self.stateful_enc0 = StatefulCausalDepthwiseConv2d.from_causal_dwconv(
            self.base.enc0.depthwise
        )
        self.stateful_enc1 = StatefulCausalDepthwiseConv2d.from_causal_dwconv(
            self.base.enc1.depthwise
        )
        self.stateful_enc2 = StatefulCausalDepthwiseConv2d.from_causal_dwconv(
            self.base.enc2.depthwise
        )
        self.stateful_enc3 = StatefulCausalDepthwiseConv2d.from_causal_dwconv(
            self.base.enc3.depthwise
        )

        # Bottleneck CausalConvGLU depthwise layer (Conv2d is inside dwconv[1])
        glu = self.base.bottleneck.conv_glu
        self.stateful_glu = StatefulCausalDepthwiseConv2d(
            dwconv=glu.dwconv[1],
            norm=nn.Identity(),  # GLU applies norm at input to fc1
            act=glu.act,
            pad_freq=(1, 1),
        )

    def forward(
        self,
        noisy_frame: Tensor,  # [B, 2, T, 257] (Yr, Yi) where T=32 for 0.5s chunk
        enc0_cache: Tensor,   # [B, 9, 2, 128]
        enc1_cache: Tensor,   # [B, 64, 2, 128]
        enc2_cache: Tensor,   # [B, 64, 2, 64]
        enc3_cache: Tensor,   # [B, 64, 2, 32]
        glu_cache: Tensor,    # [B, 64, 2, 16]
        gru_state: Tensor,    # [1, B * 16, 64]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Processes an STFT chunk (e.g. T=32 frames) and updates temporal recurrent caches.

        Returns:
            enh_frame:      [B, 2, T, 257] (Enhanced Real, Imag)
            next_enc0:      [B, 9, 2, 128]
            next_enc1:      [B, 64, 2, 128]
            next_enc2:      [B, 64, 2, 64]
            next_enc3:      [B, 64, 2, 32]
            next_glu:       [B, 64, 2, 16]
            next_gru:       [1, B * 16, 64]
        """
        Yr, Yi = noisy_frame[:, 0], noisy_frame[:, 1]
        mag = torch.clamp(Yr**2 + Yi**2, min=1e-12)**0.15

        # 1. Front-end ERB Band Merging (257 linear bins -> 128 subbands)
        feat = torch.stack([Yr, Yi, mag], dim=1)  # [B, 3, T, 257]
        feat_sub = self.base.erb.bm(feat)         # [B, 3, T, 128]

        # 2. Subband Feature Extraction (SFE: frequency sliding context k=3)
        x_pad = F.pad(feat_sub, (1, 1, 0, 0), mode="reflect")
        w0 = x_pad[:, :, :, :-2]   # left neighbor
        w1 = x_pad[:, :, :, 1:-1]  # center bin
        w2 = x_pad[:, :, :, 2:]    # right neighbor
        feat_sfe = torch.stack([w0, w1, w2], dim=2).flatten(1, 2)  # [B, 9, T, 128]

        # 3. Stateful Causal Encoder Convolutions (using StatefulCausalDepthwiseConv2d)
        dw0, next_enc0 = self.stateful_enc0(feat_sfe, enc0_cache)
        x0 = self.base.enc0.pointwise(dw0)

        dw1, next_enc1 = self.stateful_enc1(x0, enc1_cache)
        x1 = self.base.enc1.pointwise(dw1)

        dw2, next_enc2 = self.stateful_enc2(x1, enc2_cache)
        x2 = self.base.enc2.pointwise(dw2)

        dw3, next_enc3 = self.stateful_enc3(x2, enc3_cache)
        x3 = self.base.enc3.pointwise(dw3)

        # 4. Dilated-Recurrent Bottleneck
        # Parallel Frequency Path via ADCB (operates across F=16 independently per frame)
        h_f = x3 + self.base.bottleneck.freq_adcb(x3)

        # Temporal Recurrent Path via Uni-GRU with continuous state cache
        B, C, T_dim, F_dim = h_f.shape  # e.g. T_dim=32, F_dim=16, C=64
        h_t = h_f.permute(0, 3, 2, 1).reshape(B * F_dim, T_dim, C)
        norm_ht = self.base.bottleneck.time_norm(h_t)
        out_t, next_gru = self.base.bottleneck.time_gru(norm_ht, gru_state)
        h_t = h_t + self.base.bottleneck.time_fc(out_t)
        out_gru = h_t.view(B, F_dim, T_dim, C).permute(0, 3, 2, 1).contiguous()

        # Channel Mixer with stateful CausalConvGLU cache
        glu = self.base.bottleneck.conv_glu
        h_glu = glu.norm(out_gru)
        x_branch, v_branch = glu.fc1(h_glu).chunk(2, dim=1)
        gated_act, next_glu = self.stateful_glu(x_branch, glu_cache)
        gated = gated_act * torch.sigmoid(v_branch)
        bot = out_gru + glu.fc2(gated)

        # 5. Decoder (Frequency upsampling only, no temporal conv -> strictly stateless across time)
        d2 = self.base.up3(bot + self.base.p3(x3))
        d1 = self.base.up2(d2 + self.base.p2(x2))
        d0 = self.base.up1(d1 + self.base.p1(x1)) + self.base.p0(x0)

        # 6. Amplified Tanh Complex Subband Masking: [-beta, beta]
        crm_sub = self.base.beta * torch.tanh(self.base.final_conv(d0) / self.base.beta)
        crm_full = self.base.erb.bs(crm_sub)
        Mr, Mi = crm_full[:, 0], crm_full[:, 1]

        # 7. Complex Ratio Masking Synthesis
        Sr = Yr * Mr - Yi * Mi
        Si = Yr * Mi + Yi * Mr
        enh_frame = torch.stack([Sr, Si], dim=1)  # [B, 2, T, 257]

        return (
            enh_frame,
            next_enc0,
            next_enc1,
            next_enc2,
            next_enc3,
            next_glu,
            next_gru,
        )
