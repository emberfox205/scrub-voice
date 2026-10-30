"""Acoustic modeling, circular microphone array geometry, and noise synthesis.

Provides:
- Far-field plane-wave fractional delay array simulation.
- Synthetic OR-like noise (HVAC, suction, ventilator, clatter, alarms).
- Synthetic harmonic, syllable-modulated speech for smoke tests.
- SI-SDR, lag alignment, and STOI quality measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

sf = _try_import("soundfile")
sps = _try_import("scipy.signal")
spio = _try_import("scipy.io.wavfile")

C_SOUND = 343.0


def load_audio(path: Path, sr_target: int) -> np.ndarray:
    """Loads audio file as float32 in [-1, 1] shape [C, N], resampled to sr_target."""
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    else:
        sr, raw = spio.read(str(path))
        data = raw.astype(np.float32) / (32768.0 if raw.dtype == np.int16 else 1.0)
        data = data[:, None] if data.ndim == 1 else data
    x = data.T.copy()
    if sr != sr_target:
        g = math.gcd(sr, sr_target)
        if sps is not None:
            x = sps.resample_poly(x, sr_target // g, sr // g, axis=1).astype(np.float32)
        else:
            raise RuntimeError("scipy required for audio resampling")
    return x


def save_audio(path: Path, x: np.ndarray, sr: int):
    """Saves audio array to 16-bit PCM WAV."""
    data = np.clip(x.T if x.ndim == 2 else x, -1, 1).astype(np.float32)
    if sf is not None:
        sf.write(str(path), data, sr)
    elif spio is not None:
        spio.write(str(path), sr, (data * 32767).astype(np.int16))


def circular_array(n: int, radius: float) -> np.ndarray:
    """Computes (x, y) coordinates for n circular perimeter microphones."""
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def unit(az_deg: float) -> np.ndarray:
    """Unit vector pointing toward azimuth angle in degrees."""
    a = np.deg2rad(az_deg)
    return np.array([np.cos(a), np.sin(a)])


def ang_err(a: float, b: float) -> float:
    """Absolute circular angular difference between two azimuth angles."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def plane_wave(sig: np.ndarray, az_deg: float, mic_xy: np.ndarray, fs: int) -> np.ndarray:
    """Far-field plane-wave fractional delay filter -> shape [C, N]."""
    n = len(sig)
    F = np.fft.rfftfreq(n, 1 / fs)
    d = -(mic_xy @ unit(az_deg)) / C_SOUND
    X = np.fft.rfft(sig)[None, :] * np.exp(-2j * np.pi * F[None, :] * d[:, None])
    return np.fft.irfft(X, n=n, axis=1)


def _pink(n: int, rng) -> np.ndarray:
    """Generates pink noise (1/f spectral slope)."""
    W = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0)
    f[0] = f[1]
    return np.fft.irfft(W / np.sqrt(f), n=n)


def synth_or_noise(n: int, fs: int, rng) -> np.ndarray:
    """Synthetic OR noise: HVAC rumble, suction hiss, ventilator cycles, clatter, alarms."""
    t = np.arange(n) / fs
    hvac = _pink(n, rng)
    sos_bp = sps.butter(4, [200, 2500], btype="band", fs=fs, output="sos")
    suction = sps.sosfilt(sos_bp, rng.standard_normal(n)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
    vent = sps.sosfilt(sps.butter(2, 400, fs=fs, output="sos"), rng.standard_normal(n)) * (np.sin(2 * np.pi * t / 4.0) > 0.6)
    clatter = np.zeros(n)
    for pos in rng.integers(0, max(1, n - fs // 4), size=max(1, int(3 * n / fs))):
        k = np.arange(min(fs // 8, n - pos))
        clatter[pos:pos + len(k)] += np.exp(-k / (0.004 * fs)) * np.sin(2 * np.pi * rng.uniform(2500, 5500) * k / fs)
    alarm = 0.3 * np.sin(2 * np.pi * 1000 * t) * ((t % 1.5) < 0.2)
    parts = [hvac / hvac.std(), 0.8 * suction / (suction.std() + 1e-9), 0.8 * vent / (vent.std() + 1e-9),
             0.5 * clatter / (clatter.std() + 1e-9), alarm]
    out = np.sum(parts, axis=0)
    return (out / out.std()).astype(np.float32)


def synth_speechlike(dur: float, fs: int, rng) -> np.ndarray:
    """Harmonic, syllable-modulated signal used for harness checks without real data."""
    n = int(dur * fs)
    t = np.arange(n) / fs
    f0 = 110 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    sig = sum(np.sin(k * phase) / k for k in range(1, 25))
    env = np.clip(np.sin(2 * np.pi * 3.5 * t + rng.uniform(0, 6)), 0, None) ** 0.7
    sig = sps.sosfilt(sps.butter(2, [300, 3400], btype="band", fs=fs, output="sos"), sig) * env
    return (sig / (np.abs(sig).max() + 1e-9) * 0.5).astype(np.float32)


class NoiseSource:
    """Noise generator from audio WAV or synthetic generator."""
    def __init__(self, path: Optional[str], fs: int):
        self.fs = fs
        self.wav = load_audio(Path(path), fs)[0] if path else None

    def draw(self, n: int, rng) -> np.ndarray:
        if self.wav is None:
            return synth_or_noise(n, self.fs, rng)
        w = self.wav
        if len(w) < n:
            w = np.tile(w, int(np.ceil(n / len(w))))
        s = int(rng.integers(0, len(w) - n + 1))
        seg = w[s:s + n]
        return (seg / (seg.std() + 1e-9)).astype(np.float32)


def simulate_array(clean: np.ndarray, az: float, snr_db: Optional[float], mic_xy: np.ndarray, fs: int,
                   noise: NoiseSource, rng, diffuse_ratio: float = 0.5) -> np.ndarray:
    """Simulates multi-channel reception at target SNR with diffuse and directional noise."""
    speech = plane_wave(clean, az, mic_xy, fs)
    if snr_db is None:
        return speech.astype(np.float32)
    n, C = len(clean), len(mic_xy)
    diffuse = np.stack([noise.draw(n, rng) for _ in range(C)])
    int_az = (az + rng.uniform(60, 300)) % 360
    directional = plane_wave(noise.draw(n, rng), int_az, mic_xy, fs)
    directional /= directional[0].std() + 1e-9
    nz = math.sqrt(diffuse_ratio) * diffuse + math.sqrt(1 - diffuse_ratio) * directional
    ps, pn = float(np.mean(speech[0] ** 2)), float(np.mean(nz[0] ** 2))
    g = math.sqrt(ps / (pn * 10 ** (snr_db / 10) + 1e-20))
    return (speech + g * nz).astype(np.float32)


def align_to(ref: np.ndarray, est: np.ndarray, max_lag: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Integer-lag cross-correlation alignment."""
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    nfft = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(e, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
    ml = min(max_lag, n - 1)
    cc = np.concatenate([cc[-ml:], cc[:ml + 1]])
    lag = int(np.argmax(cc)) - ml
    return r, np.roll(e, -lag)


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (capped at 60 dB)."""
    ref, est = ref - ref.mean(), est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    tgt = alpha * ref
    return float(min(60.0, 10 * np.log10((np.dot(tgt, tgt) + 1e-12) / (np.dot(est - tgt, est - tgt) + 1e-12))))


def quality(clean: np.ndarray, est: np.ndarray, sr: int, want_stoi: bool = False) -> Dict[str, Optional[float]]:
    """Evaluates SI-SDR and optional STOI."""
    r, e = align_to(clean, est)
    out: Dict[str, Optional[float]] = {"si_sdr": si_sdr(r, e), "stoi": None}
    if want_stoi:
        pystoi = _try_import("pystoi")
        if pystoi:
            try:
                out["stoi"] = float(pystoi.stoi(r, e, sr, extended=False))
            except Exception:
                pass
    return out


@dataclass
class Item:
    """Represents a benchmark test sample."""
    name: str
    audio: np.ndarray                       # [C, N]
    text: str = ""
    intent: str = ""
    status: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    azimuth: Optional[float] = None

    @property
    def real_array(self) -> bool:
        return self.audio.shape[0] > 1


def build_items(args, rng) -> List[Item]:
    """Builds evaluation item list from manifest, WAV file, or synthetic clips."""
    items: List[Item] = []
    if getattr(args, "manifest", None):
        base = Path(args.manifest).parent
        for line in Path(args.manifest).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = Path(r["audio"])
            p = p if p.is_absolute() else base / p
            items.append(Item(p.name, load_audio(p, args.sr), r.get("text", ""), r.get("intent", ""), r.get("status", "EXECUTE"),
                              r.get("slots", {}), r.get("azimuth")))
    if getattr(args, "wav", None):
        items.append(Item(Path(args.wav).name, load_audio(Path(args.wav), args.sr)))
    if getattr(args, "synthetic", 0):
        for i in range(args.synthetic):
            items.append(Item(f"synth_{i}", synth_speechlike(2.0, args.sr, rng)[None, :]))
    if getattr(args, "max_items", 0):
        items = items[:args.max_items]
    return items


def snr_label(s: Optional[float]) -> str:
    return "clean" if s is None else f"{s:g}dB"


def make_sample(item: Item, snr: Optional[float], args, mic_xy, noise: NoiseSource, rng):
    """Returns (multichannel [C,N], clean target at reference mic or None, ground-truth azimuth or None)."""
    if item.real_array:
        idx = [i for i in args.mic_idx if i < item.audio.shape[0]]
        return item.audio[idx], None, item.azimuth
    az = item.azimuth if item.azimuth is not None else float(rng.uniform(0, 360))
    mc = simulate_array(item.audio[0], az, snr, mic_xy, args.sr, noise, rng, getattr(args, "diffuse_ratio", 0.5))
    target = plane_wave(item.audio[0], az, mic_xy, args.sr)[0]
    return mc, target, az


def chunk_iter(x: np.ndarray, n: int):
    """Yields audio chunks of size n, zero-padding the final chunk."""
    for s in range(0, x.shape[-1], n):
        c = x[..., s:s + n]
        if c.shape[-1] < n:
            c = np.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, n - c.shape[-1])])
        yield c


def build_stream(items: List[Item], args, mic_xy, noise, rng, min_s: float = 12.0) -> np.ndarray:
    """Creates a continuous multi-channel test stream for DSP timing reps."""
    pool = [make_sample(it, 10.0, args, mic_xy, noise, rng)[0] for it in (items or [Item("synth", synth_speechlike(2.0, args.sr, rng)[None, :])])]
    reps = int(math.ceil(min_s * args.sr / sum(x.shape[-1] for x in pool)))
    return np.concatenate(pool * max(1, reps), axis=1)

