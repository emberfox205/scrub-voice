"""
DSP Denoise Package: Speech Enhancement & Psychoacoustic Suppression Modules.
"""
from __future__ import annotations

from src.dsp.denoise.continuous_slope_denoiser import (
    ContinuousSlopeDenoiser,
    ContinuousSlopeGainFloor,
    SoftKneeSNRBlender,
)

__all__ = [
    "ContinuousSlopeDenoiser",
    "ContinuousSlopeGainFloor",
    "SoftKneeSNRBlender",
    "DuoFilterStreamingDenoiser",
    "DuoFilterDenoiser",
    "DeepFilterDenoiser",
    "GTCRNDenoiser",
]


def __getattr__(name: str):
    if name == "DuoFilterStreamingDenoiser":
        from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import (
            DuoFilterStreamingDenoiser,
        )
        return DuoFilterStreamingDenoiser
    if name == "DuoFilterDenoiser":
        from src.dsp.denoise.stateless_onnx.stateless_duofilter_denoiser import (
            DuoFilterDenoiser,
        )
        return DuoFilterDenoiser
    if name == "DeepFilterDenoiser":
        from src.dsp.denoise.sota_onnx.deepfilternet_denoiser import (
            DeepFilterDenoiser,
        )
        return DeepFilterDenoiser
    if name == "GTCRNDenoiser":
        from src.dsp.denoise.sota_onnx.gtcrn_denoiser import GTCRNDenoiser
        return GTCRNDenoiser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
