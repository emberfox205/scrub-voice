"""Backward compatibility shim for DeepFilterNet denoiser."""
from src.dsp.denoise.sota_onnx.deepfilternet_denoiser import DeepFilterDenoiser

__all__ = ["DeepFilterDenoiser"]
