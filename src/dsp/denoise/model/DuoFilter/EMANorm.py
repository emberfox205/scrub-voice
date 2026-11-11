import torch
from torch import nn

class CausalEmaNorm(nn.Module):
    """
    Numerically stable Causal EMA Normalizer (Stage 1 / DeepFilterNet heritage).
    Normalizes time-frequency features strictly causally across time T without FP16 overflow.
    """
    def __init__(self, alpha: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        out = torch.empty_like(x)

        # Initialize moving mean and variance with first frame
        mean = x[:, :, 0, :]
        var = torch.ones_like(mean)

        out[:, :, 0, :] = (x[:, :, 0, :] - mean) / torch.sqrt(var + self.eps)

        for t in range(1, T):
            curr_x = x[:, :, t, :]
            diff = curr_x - mean
            mean = self.alpha * mean + (1.0 - self.alpha) * curr_x
            var = (self.alpha * var + (1.0 - self.alpha) * diff * (curr_x - mean)).clamp_min(0.0)

            out[:, :, t, :] = (curr_x - mean) / torch.sqrt(var + self.eps)

        return out