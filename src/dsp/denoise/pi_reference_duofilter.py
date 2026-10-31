"""Backward compatibility shim for DuoFilter denoiser."""
from src.dsp.denoise.stateless_onnx.stateless_duofilter_denoiser import DuoFilterDenoiser

__all__ = ["DuoFilterDenoiser"]
