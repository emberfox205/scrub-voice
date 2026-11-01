"""Backward compatibility shim for DuoFilter denoiser."""
from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import DuoFilterStreamingDenoiser as DuoFilterDenoiser

__all__ = ["DuoFilterDenoiser"]
