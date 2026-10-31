"""Component registry and lazy wrapper factory.

Provides unified interfaces across all implementations and baselines:
- Beamformers: ref_channel, ref_srp_das, repo_beamformer
- Denoisers: passthrough, spectral_gate, noisereduce, deepfilternet, repo_denoiser
- ASR engines: vosk_grammar, vosk_open, whisper, repo_asr
- NLU engines: repo_setfit, tfidf_logreg
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

sps = _try_import("scipy.signal")
pc = time.perf_counter

from test.benchmarks.common.acoustics import C_SOUND, unit
from test.benchmarks.common.telemetry import get_rss_mb

# System performance budgets from architecture specification
BUDGETS = {
    "dsp_rtf_max": 0.50,
    "asr_final_ms_max": 150.0,
    "nlu_ms_max": 180.0,
    "ipc_ms_max": 5.0,
    "dispatch_ms_max": 5.0,
    "e2e_ms_max": 500.0,
    "e2e_ms_stretch": 300.0,
    "rss_total_mb_max": 800.0,
}

DEFAULT_NLU_EVAL = [
    ("Please zoom in two point five times on coronal view", "zoom_display", "EXECUTE"),
    ("Zoom out to wide view", "zoom_display", "EXECUTE"),
    ("Advance forward 5 slices please", "navigate_slice", "EXECUTE"),
    ("Go back 10 slices nurse", "navigate_slice", "EXECUTE"),
    ("Jump to slice number 45", "navigate_slice", "EXECUTE"),
    ("Open CT scan on primary monitor", "open_scan", "EXECUTE"),
    ("Show brain MRI", "open_scan", "EXECUTE"),
    ("Reset display view", "reset_camera", "EXECUTE"),
    ("Emergency stop immediately freeze", "halt_emergency", "EXECUTE"),
    ("Clear surgical area back off", "clear_emergency", "EXECUTE"),
    ("Hand me the scalpel please", "pick_tool", "EXECUTE"),
    ("Pass the iris scissors", "pick_tool", "EXECUTE"),
    ("Park robotic arm", "park_arm", "EXECUTE"),
    ("Record milestone trocar placed note clip attached", "log_event", "EXECUTE"),
    ("What is the weather like outside today", "unknown", "OUT-OF-DOMAIN"),
    ("Could you play some soft music in here", "unknown", "OUT-OF-DOMAIN"),
    ("423", "unknown", "OUT-OF-DOMAIN"),
]


class RefChannel:
    name = "ref_channel"

    def process(self, x: np.ndarray, sr: Optional[int] = None):
        return x[0], None


class RefBeamformer:
    """Reference SRP-PHAT DOA + frequency-domain delay-and-sum (numpy baseline and fallback)."""
    name = "ref_srp_das"

    def __init__(self, mic_xy: np.ndarray, fs: int, n_angles: int = 72, fmin: float = 300.0, fmax: float = 4000.0, fstep: int = 4):
        self.mic_xy, self.fs, self.fmin, self.fmax, self.fstep = mic_xy, fs, fmin, fmax, fstep
        self.angles = np.arange(n_angles) * 360.0 / n_angles
        U = np.stack([unit(a) for a in self.angles], axis=1)
        self.pairs = [(i, j) for i in range(len(mic_xy)) for j in range(i + 1, len(mic_xy))]
        diff = np.stack([mic_xy[i] - mic_xy[j] for i, j in self.pairs])
        self.tau = (-(diff @ U) / C_SOUND).astype(np.float32)

    def process(self, x: np.ndarray, sr: Optional[int] = None):
        C, N = x.shape
        F = np.fft.rfftfreq(N, 1 / self.fs)
        band = np.where((F >= self.fmin) & (F <= self.fmax))[0][::self.fstep]
        Xw = np.fft.rfft(x * np.hanning(N)[None, :], axis=1)[:, band]
        R = np.stack([Xw[i] * np.conj(Xw[j]) for i, j in self.pairs])
        R = (R / (np.abs(R) + 1e-9)).astype(np.complex64)
        ph = np.exp(2j * np.pi * F[band][None, None, :].astype(np.float32) * self.tau[:, :, None]).astype(np.complex64)
        srp = np.real(np.einsum("pf,paf->a", R, ph))
        az = float(self.angles[int(np.argmax(srp))])
        d = -(self.mic_xy @ unit(az)) / C_SOUND
        d = d - d[0]
        X = np.fft.rfft(x, axis=1) * np.exp(2j * np.pi * F[None, :] * d[:, None])
        return np.fft.irfft(X.mean(axis=0), n=N).astype(np.float32), az


class Passthrough:
    name = "passthrough"

    def process(self, x: np.ndarray, sr: Optional[int] = None):
        return x


class SpectralGate:
    """Classical stationary-noise spectral gating baseline (Wiener-style gain)."""
    name = "spectral_gate"

    def process(self, x: np.ndarray, sr: int = 16000):
        if sps is None:
            return x
        _, _, Z = sps.stft(x, sr, nperseg=512, noverlap=256)
        mag = np.abs(Z)
        noise = np.percentile(mag, 15, axis=1, keepdims=True)
        gain = np.clip(1.0 - 1.5 * noise / (mag + 1e-9), 0.1, 1.0)
        _, y = sps.istft(Z * gain, sr, nperseg=512, noverlap=256)
        return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)].astype(np.float32)


class NoiseReduceDenoiser:
    name = "noisereduce"

    def __init__(self):
        self.nr = importlib.import_module("noisereduce")

    def process(self, x: np.ndarray, sr: int = 16000):
        return self.nr.reduce_noise(y=x, sr=sr, stationary=False).astype(np.float32)


class DeepFilterNetDenoiser:
    """DeepFilterNet reference implementation (runs at 48 kHz internally)."""
    name = "deepfilternet"

    def __init__(self):
        import torch
        from df.enhance import enhance, init_df
        self.torch, self.enhance = torch, enhance
        self.model, self.state, _ = init_df()
        self.sr = int(self.state.sr())

    def process(self, x: np.ndarray, sr: int = 16000):
        g = math.gcd(self.sr, sr)
        up = sps.resample_poly(x, self.sr // g, sr // g)
        out = self.enhance(self.model, self.state, self.torch.from_numpy(up.astype(np.float32)).unsqueeze(0))
        y = sps.resample_poly(out.squeeze(0).numpy(), sr // g, self.sr // g)
        return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)].astype(np.float32)


class PluginBeamformer:
    def __init__(self, obj: Any):
        self.o, self.name = obj, "repo_beamformer"

    def process(self, x: np.ndarray, sr: int = 16000):
        try:
            out = self.o.process(x, sr)
        except TypeError:
            out = self.o.process(x)
        if isinstance(out, tuple):
            return np.asarray(out[0], np.float32).reshape(-1), (float(out[1]) if out[1] is not None else None)
        return np.asarray(out, np.float32).reshape(-1), None


class PluginDenoiser:
    def __init__(self, obj: Any):
        self.o, self.name = obj, "repo_denoiser"

    def process(self, x: np.ndarray, sr: int = 16000):
        if hasattr(self.o, "enhance"):
            out = self.o.enhance(x, sample_rate=sr)
        else:
            try:
                out = self.o.process(x, sr)
            except TypeError:
                out = self.o.process(x)
        return np.asarray(out, np.float32).reshape(-1)


class VoskASR:
    def __init__(self, model: Any, grammar: Optional[List[str]], sr: int, name: str):
        self.vosk, self.model, self.grammar, self.sr, self.name = _try_import("vosk"), model, grammar, sr, name
        self.rec = None

    def reset(self):
        g = json.dumps(self.grammar) if self.grammar else None
        self.rec = self.vosk.KaldiRecognizer(self.model, self.sr, g) if g else self.vosk.KaldiRecognizer(self.model, self.sr)

    def accept(self, chunk: np.ndarray):
        self.rec.AcceptWaveform((np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes())

    def finalize(self) -> str:
        return json.loads(self.rec.FinalResult()).get("text", "")


class WhisperASR:
    """Non-streaming SOTA baseline: buffers audio, decodes once at end of speech."""

    def __init__(self, size: str, threads: int):
        from faster_whisper import WhisperModel
        self.m, self.name, self.buf = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads), f"whisper_{size}", []

    def reset(self):
        self.buf = []

    def accept(self, chunk: np.ndarray):
        self.buf.append(np.asarray(chunk, np.float32))

    def finalize(self) -> str:
        audio = np.concatenate(self.buf) if self.buf else np.zeros(1600, np.float32)
        segs, _ = self.m.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text for s in segs)


class RepoNLU:
    name = "repo_setfit"

    def __init__(self, threads: int):
        from src.nlu.pipeline import VoiceToIntentPipeline
        self.p = VoiceToIntentPipeline(num_threads=threads)
        self.schema = getattr(self.p, "schema", None)

    def process(self, text: str) -> Dict[str, Any]:
        return self.p.process(text)


class TfidfNLU:
    """Classical baseline: TF-IDF + logistic regression, OOD by max-probability threshold."""
    name = "tfidf_logreg"

    def __init__(self, train: List[Dict[str, Any]], thr: float = 0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000).fit(self.vec.fit_transform([r["text"] for r in train]), [r["intent"] for r in train])
        self.thr = thr

    def process(self, text: str) -> Dict[str, Any]:
        pr = self.clf.predict_proba(self.vec.transform([text]))[0]
        k = int(np.argmax(pr))
        ok = pr[k] >= self.thr
        return {"intent": self.clf.classes_[k] if ok else "unknown", "status": "EXECUTE" if ok else "OUT-OF-DOMAIN",
                "slots": {}, "confidence": float(pr[k])}


def load_plugin(spec: str) -> Any:
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)()


class Registry:
    """Lazy component factory. Missing/unwired components are skipped and reported, never fatal."""

    def __init__(self, args: Any, mic_xy: np.ndarray, grammar: Optional[List[str]]):
        self.a, self.mic_xy, self.grammar = args, mic_xy, grammar
        self.cache: Dict[Tuple[str, str], Any] = {}
        self.skipped: Dict[str, str] = {}
        self.load_info: Dict[str, Dict[str, float]] = {}
        plugin_list = getattr(args, "plugin", []) or []
        self.plugins = dict(p.split("=", 1) for p in plugin_list)
        self._vosk_model = None

    def get(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self.cache:
            r0, t0 = get_rss_mb(), pc()
            try:
                obj = self._build(kind, name)
                self.load_info[f"{kind}:{name}"] = {"load_s": round(pc() - t0, 2), "rss_delta_mb": round(get_rss_mb() - r0, 1)}
            except Exception as e:
                obj = None
                self.skipped[f"{kind}:{name}"] = f"{type(e).__name__}: {e}"
            self.cache[key] = obj
        return self.cache[key]

    def _plugin(self, name: str) -> Any:
        if name not in self.plugins:
            raise RuntimeError(f"not wired; pass --plugin {name}=module:Class")
        return load_plugin(self.plugins[name])

    def _build(self, kind: str, name: str) -> Any:
        a = self.a
        if kind == "bf":
            if name == "ref_channel":
                return RefChannel()
            if name == "ref_srp_das":
                return RefBeamformer(self.mic_xy, a.sr)
            if name == "repo_beamformer":
                if name in self.plugins:
                    return PluginBeamformer(self._plugin(name))
                # Automatic fallback to repo DASBeamformer + SRP-PHAT
                try:
                    from src.dsp.source.beamformer import DASBeamformer
                    from src.dsp.source.localizer import SRPPHATLocalizer

                    class BuiltinDASAdapter:
                        def __init__(self, mic_xy: np.ndarray, sr: int):
                            self.localizer = SRPPHATLocalizer(sample_rate=sr)
                            self.beamformer = DASBeamformer(sample_rate=sr)

                        def process(self, x: np.ndarray, sr: int = 16000):
                            az = self.localizer.estimate_angle(x)
                            y = self.beamformer.process(x, target_angle_deg=az)
                            return y, az

                    return BuiltinDASAdapter(self.mic_xy, a.sr)
                except Exception:
                    return RefBeamformer(self.mic_xy, a.sr)
        elif kind == "dn":
            if name == "passthrough":
                return Passthrough()
            if name == "spectral_gate":
                return SpectralGate()
            if name == "noisereduce":
                return NoiseReduceDenoiser()
            if name == "deepfilternet":
                return DeepFilterNetDenoiser()
            if name == "repo_denoiser":
                if name in self.plugins:
                    return PluginDenoiser(self._plugin(name))
                # Fallback to repo DuoFilter
                try:
                    from src.dsp.denoise.stateful_onnx.duofilter_streaming_denoiser import DuoFilterStreamingDenoiser
                    return PluginDenoiser(DuoFilterStreamingDenoiser())
                except Exception:
                    try:
                        from src.dsp.denoise.stateless_onnx.stateless_duofilter_denoiser import DuoFilterDenoiser
                        return PluginDenoiser(DuoFilterDenoiser())
                    except Exception:
                        return SpectralGate()
        elif kind == "asr":
            if name == "repo_asr":
                if name in self.plugins:
                    return self._plugin(name)
                return self.get("asr", "vosk_grammar")
            if name in ("vosk_grammar", "vosk_open"):
                vosk = importlib.import_module("vosk")
                if self._vosk_model is None:
                    if not getattr(a, "vosk_model", None):
                        raise RuntimeError("pass --vosk-model")
                    vosk.SetLogLevel(-1)
                    self._vosk_model = vosk.Model(a.vosk_model)
                return VoskASR(self._vosk_model, self.grammar if name == "vosk_grammar" else None, a.sr, name)
            if name == "whisper":
                return WhisperASR(a.whisper_model, a.threads)
        raise RuntimeError("unknown component")

