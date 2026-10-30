"""Hardware telemetry, platform inspection, and background resource sampling.

Provides:
- Per-core CPU utilization (mean, peak, and sum across cores).
- Resident Set Size (RSS memory in MB) for the benchmark process and children.
- Raspberry Pi SoC temperature (/sys/class/thermal).
- Raspberry Pi vcgencmd throttling and under-voltage flags.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

psutil = _try_import("psutil")
pc = time.perf_counter

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip().strip("\x00")
    except Exception:
        return None


def get_rss_mb() -> float:
    """Returns total Resident Set Size (MB) of this process and all child workers."""
    if psutil is None:
        return 0.0
    try:
        p = psutil.Process(os.getpid())
        return (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 1048576.0
    except Exception:
        return 0.0


def hw_info() -> Dict[str, Any]:
    """Inspects OS, CPU governor, memory, and package versions."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device_model": _read("/proc/device-tree/model"),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
    }
    f = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    info["cpu0_freq_mhz"] = int(f) / 1000 if f else None
    if psutil:
        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1048576, 0)
    try:
        info["git_commit"] = (
            subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=3)
            .stdout.strip() or None
        )
    except Exception:
        info["git_commit"] = None
    versions = {}
    for m in ("numpy", "scipy", "torch", "onnxruntime", "vosk", "faster_whisper", "zmq", "sounddevice", "noisereduce", "df"):
        mod = _try_import(m)
        if mod is not None:
            versions[m] = getattr(mod, "__version__", "?")
    info["versions"] = versions
    return info


class ResourceSampler:
    """Background sampling daemon: tracks per-core CPU, RSS, SoC temp, and throttle flags."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.stop_evt = interval, threading.Event()
        self.cores: List[List[float]] = []
        self.rss: List[Tuple[float, float]] = []
        self.temps: List[float] = []
        self.throttle_now = 0
        self.throttle_seen = 0
        self.t0 = 0.0

    def __enter__(self):
        if psutil:
            psutil.cpu_percent(percpu=True)
        self.t0 = pc()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *_):
        self.stop_evt.set()
        self.th.join(timeout=2)

    def _loop(self):
        last_thr = 0.0
        while not self.stop_evt.wait(self.interval):
            if psutil:
                self.cores.append(psutil.cpu_percent(percpu=True))
            self.rss.append((pc() - self.t0, get_rss_mb()))
            t = _read("/sys/class/thermal/thermal_zone0/temp")
            if t:
                self.temps.append(int(t) / 1000.0)
            if pc() - last_thr > 1.0:
                last_thr = pc()
                try:
                    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout
                    v = int(out.strip().split("=")[1], 16)
                    self.throttle_now |= v & 0xF
                    self.throttle_seen |= (v >> 16) & 0xF
                except Exception:
                    pass

    def summary(self) -> Dict[str, Any]:
        """Returns statistical summary of resources during the sampling window."""
        out: Dict[str, Any] = {"duration_s": round(pc() - self.t0, 2)}
        if self.cores:
            c = np.asarray(self.cores)
            out.update(
                cpu_sum_mean_pct=round(float(c.sum(axis=1).mean()), 1),
                cpu_sum_peak_pct=round(float(c.sum(axis=1).max()), 1),
                per_core_mean_pct=[round(float(v), 1) for v in c.mean(axis=0)],
                per_core_peak_pct=[round(float(v), 1) for v in c.max(axis=0)],
            )
        if self.rss:
            out["rss_peak_mb"] = round(max(v for _, v in self.rss), 1)
        if self.temps:
            out["temp_max_c"] = round(max(self.temps), 1)
        out["throttled_now_flags"] = self.throttle_now
        out["throttled_since_boot_flags"] = self.throttle_seen
        return out

