import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.dsp.denoise.model.DuoFilter.ERBFilterBank import ERBFilterBank
from src.dsp.denoise.model.DuoFilter.EMANorm import CausalEmaNorm
from src.dsp.denoise.model.DuoFilter.Layer import CausalDepthwiseConv2d, CausalDepthwiseConvTranspose2d, PConv2d, get_groups


class CausalSeparableConv2d(nn.Module):
    """
    Encoder Block:
        x -> CausalDepthwiseConv2d -> PConv2d -> out
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 1), dropout_p: float = 0.0):
        super().__init__()
        self.depthwise = CausalDepthwiseConv2d(in_ch, kernel_size=kernel_size, stride=stride, use_norm=True, act=nn.Mish())
        self.pointwise = PConv2d(in_ch, out_ch, use_norm=True, act=nn.Mish())
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.pointwise(self.depthwise(x)))

class CausalSeparableConvTranspose2d(nn.Module):
    """
    Decoder Block:
        x -> PConv2d -> CausalDepthwiseConvTranspose2d -> out
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple = (1, 3), stride: tuple = (1, 2), dropout_p: float = 0.0):
        super().__init__()
        self.pointwise = PConv2d(in_ch, out_ch, use_norm=True, act=nn.Mish())
        self.depthwise = CausalDepthwiseConvTranspose2d(out_ch, kernel_size=kernel_size, stride=stride, use_norm=True, act=nn.Mish())
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.depthwise(self.pointwise(x)))

class ADCB(nn.Module):
    """
    Asymmetric Dilated Convolutional Block across the Frequency Dimension (F=16).
    Replaces sequential Bi-GRU with parallelized 1D depthwise dilated convolutions (d={1, 2, 4}).
    Receptive field = 1 + 2*(1+2+4) = 15 frequency bins.
    """
    def __init__(self, channels: int = 64, dilations: tuple = (1, 2, 4), dropout_p: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)

        layers = []
        for d in dilations:
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=d, dilation=d, groups=channels),
                nn.GroupNorm(num_groups=4, num_channels=channels),
                nn.Mish(),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
            ))

        self.convs = nn.ModuleList(layers)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Input: (B, C, T, F) -> Output: (B, C, T, F)"""
        B, C, T, F = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * T, C, F)
        res = self.norm(h)
        for conv in self.convs:
            res = res + conv(res)
        out = self.proj(res.permute(0, 2, 1)).permute(0, 2, 1)
        return out.view(B, T, C, F).permute(0, 2, 1, 3).contiguous()

class CausalConvGLU(nn.Module):
    """Channel mixer with depthwise 3x3 causal temporal filtering and GLU gating"""
    def __init__(self, channels: int = 64, expansion_factor: float = 1.0, dropout_p: float = 0.1):
        super().__init__()
        hidden_dim = int(channels * expansion_factor)
        self.norm = nn.GroupNorm(num_groups=get_groups(channels, 4), num_channels=channels)

        # 1x1 Conv pointwise projection 
        self.fc1 = nn.Conv2d(channels, hidden_dim * 2, kernel_size=1)

        # Depthwise 3x3 causal conv with past padding (left=1, right=1, top=2, bottom=0)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, groups=hidden_dim, bias=True)
        )

        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        h = self.norm(x)
        x_branch, v_branch = self.fc1(h).chunk(2, dim=1)
        gated = self.act(self.dwconv(x_branch)) * torch.sigmoid(v_branch)
        out = self.fc2(self.dropout(gated))
        return res + out

class ADCB_Temporal_Bottleneck(nn.Module):
    """Dilated-Recurrent Bottleneck: Parallel Frequency Path + Causal Uni-GRU Temporal Path."""
    def __init__(self, channels: int = 64, dropout_p: float = 0.1):
        super().__init__()
        # 1. Parallel Frequency Path via ADCB 
        self.freq_adcb = ADCB(channels=channels, dilations=(1, 2, 4), dropout_p=dropout_p)

        # 2. Strictly Causal Temporal Path via Uni-GRU
        self.time_norm = nn.LayerNorm(channels)
        self.time_gru = nn.GRU(channels, channels, bidirectional=False, batch_first=True)
        self.time_fc = nn.Linear(channels, channels)
        self.time_dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

        # 3. Channel Mixer
        self.conv_glu = CausalConvGLU(channels=channels, expansion_factor=1.0, dropout_p=dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        h_f = x + self.freq_adcb(x)
        B, C, T, F = h_f.shape
        h_t = h_f.permute(0, 3, 2, 1).reshape(B * F, T, C)
        out_t, _ = self.time_gru(self.time_norm(h_t))
        h_t = h_t + self.time_dropout(self.time_fc(out_t))
        out = h_t.view(B, F, T, C).permute(0, 3, 2, 1).contiguous()
        return self.conv_glu(out)


class DuoFilter(nn.Module):
    """
    Full Pipeline:
        STFT -> BM -> SFE -> Separable Encoder -> ADCB Bottleneck -> Separable Decoder -> CRM -> BS -> iSTFT
    """
    def __init__(self, beta: float = 2.0, use_skip_pconv: bool = False, dropout_p: float = 0.1, use_ema_norm: bool = False):
        super().__init__()
        self.beta = beta
        self.use_skip_pconv = use_skip_pconv
        self.use_ema_norm = use_ema_norm
        self.erb = ERBFilterBank(n_fft=512, fs=16000, n_linear=64, n_erb=64)

        if self.use_ema_norm:
            self.ema_norm = CausalEmaNorm(alpha=0.99, eps=1e-5)
        else:
            self.ema_norm = nn.Identity()
        
        # Causal Depthwise-Separable Encoder
        self.enc0 = CausalSeparableConv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), dropout_p=dropout_p)
        self.enc1 = CausalSeparableConv2d(64, 64, kernel_size=(3, 3), stride=(1, 2), dropout_p=dropout_p)
        self.enc2 = CausalSeparableConv2d(64, 64, kernel_size=(3, 3), stride=(1, 2), dropout_p=dropout_p)
        self.enc3 = CausalSeparableConv2d(64, 64, kernel_size=(3, 3), stride=(1, 2), dropout_p=dropout_p)

        # Skip Connection Projections
        if self.use_skip_pconv:
            self.p3 = PConv2d(64, 64, use_norm=False, act=None, bias=False)
            self.p2 = PConv2d(64, 64, use_norm=False, act=None, bias=False)
            self.p1 = PConv2d(64, 64, use_norm=False, act=None, bias=False)
            self.p0 = PConv2d(64, 64, use_norm=False, act=None, bias=False)
        else:
            self.p3 = nn.Identity()
            self.p2 = nn.Identity()
            self.p1 = nn.Identity()
            self.p0 = nn.Identity()

        # ADCB Dilated-Recurrent Bottleneck
        self.bottleneck = ADCB_Temporal_Bottleneck(channels=64, dropout_p=dropout_p)

        # Causal Depthwise-Separable Trannsposed Conv Decoder
        self.up3 = CausalSeparableConvTranspose2d(64, 64, kernel_size=(1, 3), stride=(1, 2), dropout_p=dropout_p)
        self.up2 = CausalSeparableConvTranspose2d(64, 64, kernel_size=(1, 3), stride=(1, 2), dropout_p=dropout_p)
        self.up1 = CausalSeparableConvTranspose2d(64, 64, kernel_size=(1, 3), stride=(1, 2), dropout_p=dropout_p)

        # Output projection using PConv2d
        self.final_conv = PConv2d(64, 2, use_norm=False, act=None, bias=True)

    def forward(self, noisy_stft: Tensor) -> Tensor:
        """
        Input:
            noisy_stft: [B, 2, T, 257] (Yr, Yi)
        Returns:
            enhanced_stft: [B, 2, T, 257] (Sr, Si)
        """
        Yr, Yi = noisy_stft[:, 0], noisy_stft[:, 1]
        mag = torch.clamp(Yr**2 + Yi**2, min=1e-12)**0.15

        # 1. Front-end ERB Band Mergin
        feat = torch.stack([Yr, Yi, mag], dim=1)  # (B, 3, T, 257)
        feat_sub = self.erb.bm(feat)

        # Apply Causal EMA Norm
        feat_sub = self.ema_norm(feat_sub)

        # 2. Subband Feature Extraction (SFE: context sliding window k = 3)
        x_pad = F.pad(feat_sub, (1, 1, 0, 0), mode='reflect')

        # Slice left, center, and right context bins (equivalent to unfold k=3, step=1)
        w0 = x_pad[:, :, :, :-2]   # left neighbor
        w1 = x_pad[:, :, :, 1:-1]  # center bin
        w2 = x_pad[:, :, :, 2:]    # right neighbor

        # Stack along window dim and flatten into 9 channels (3 channels x 3 bins)
        feat_sfe = torch.stack([w0, w1, w2], dim=2).flatten(1, 2)  # Shape: (B, 9, T, 128)

        # 3. Causal Depthwise-Separable Encoder
        x0 = self.enc0(feat_sfe)  # (B, 64, T, 128)
        x1 = self.enc1(x0)        # (B, 64, T, 64)
        x2 = self.enc2(x1)        # (B, 64, T, 32)
        x3 = self.enc3(x2)        # (B, 64, T, 16)

        # 4. Dilated-Recurrent Bottleneck
        bot = self.bottleneck(x3)

        # 5. Causal Separable Transposed Conv Decoder
        d2 = self.up3(bot + self.p3(x3))
        d1 = self.up2(d2 + self.p2(x2))
        d0 = self.up1(d1 + self.p1(x1)) + self.p0(x0)

        # 6. Amplified Tanh Complex Subband Mask: [-2.0, 2.0]
        crm_sub = self.beta * torch.tanh(self.final_conv(d0) / self.beta)

        # 7. Matrix Band Splitting to 257 linear bins
        crm_full = self.erb.bs(crm_sub)
        Mr, Mi = crm_full[:, 0], crm_full[:, 1]

        # 8. Complex Ration Masking Synthesis
        Sr = Yr * Mr - Yi * Mi
        Si = Yr * Mi + Yi * Mr

        return torch.stack([Sr, Si], dim=1)
