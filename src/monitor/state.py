"""
State representation for the Scrub-Voice Monitor subsystem.
Maintains DICOM viewport state, surgical EMR milestones, emergency interlock status,
and robot telemetry placeholders.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RobotTelemetry:
    """Robot arm and gripper telemetry state (standby placeholders for edge display)."""
    status: str = "STANDBY / OFFLINE"
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0
    gripper_status: str = "STANDBY / OFFLINE"
    gripper_force_n: float = 0.0
    gripper_width_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorState:
    """Canonical intraoperative state cache for the monitor UI."""
    active_scan: Optional[str] = None  # "ct", "mri", "xray" or None
    current_slice: int = 1
    zoom_factor: float = 1.0  # Clamped between 1.0 and 10.0
    held_tool: Optional[str] = None  # "scalpel", "forceps", "scissors", "hemostat" or None
    recent_milestones: List[Dict[str, Any]] = field(default_factory=list)
    emergency_active: bool = False
    last_command: Optional[Dict[str, Any]] = None
    last_updated: float = field(default_factory=time.time)
    telemetry: RobotTelemetry = field(default_factory=RobotTelemetry)
    # Thread-safe reentrant lock to protect concurrent reads/writes across ZMQ and ASGI threads
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    @property
    def lock(self) -> threading.RLock:
        """Exposes reentrant lock for thread-safe external compound transactions."""
        return self._lock

    def to_dict(self) -> Dict[str, Any]:
        """Serializes state to a plain dictionary for JSON transmission in a thread-safe manner."""
        with self._lock:
            zoom = 1.0 if self.zoom_factor is None else float(self.zoom_factor)
            return {
                "active_scan": self.active_scan,
                "current_slice": self.current_slice,
                "zoom_factor": round(zoom, 2),
                "held_tool": self.held_tool,
                "recent_milestones": list(self.recent_milestones) if self.recent_milestones else [],
                "emergency_active": bool(self.emergency_active),
                "last_command": copy.deepcopy(self.last_command) if self.last_command else None,
                "last_updated": float(self.last_updated) if self.last_updated is not None else time.time(),
                "telemetry": self.telemetry.to_dict() if self.telemetry else RobotTelemetry().to_dict(),
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MonitorState:
        """Hydrates a MonitorState instance from a dictionary."""
        if not isinstance(data, dict):
            return cls()

        telemetry_raw = data.get("telemetry") or {}
        if not isinstance(telemetry_raw, dict):
            telemetry_raw = {}

        def _safe_float(val: Any, default: float) -> float:
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def _safe_int(val: Any, default: int) -> int:
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        telemetry = RobotTelemetry(
            status=str(telemetry_raw.get("status") or "STANDBY / OFFLINE"),
            x_mm=_safe_float(telemetry_raw.get("x_mm"), 0.0),
            y_mm=_safe_float(telemetry_raw.get("y_mm"), 0.0),
            z_mm=_safe_float(telemetry_raw.get("z_mm"), 0.0),
            rx_deg=_safe_float(telemetry_raw.get("rx_deg"), 0.0),
            ry_deg=_safe_float(telemetry_raw.get("ry_deg"), 0.0),
            rz_deg=_safe_float(telemetry_raw.get("rz_deg"), 0.0),
            gripper_status=str(telemetry_raw.get("gripper_status") or "STANDBY / OFFLINE"),
            gripper_force_n=_safe_float(telemetry_raw.get("gripper_force_n"), 0.0),
            gripper_width_mm=_safe_float(telemetry_raw.get("gripper_width_mm"), 0.0),
        )

        raw_milestones = data.get("recent_milestones")
        milestones = list(raw_milestones) if isinstance(raw_milestones, list) else []

        return cls(
            active_scan=data.get("active_scan"),
            current_slice=_safe_int(data.get("current_slice"), 1),
            zoom_factor=_safe_float(data.get("zoom_factor"), 1.0),
            held_tool=data.get("held_tool"),
            recent_milestones=milestones,
            emergency_active=bool(data.get("emergency_active", False)),
            last_command=data.get("last_command") if isinstance(data.get("last_command"), dict) else None,
            last_updated=_safe_float(data.get("last_updated"), time.time()),
            telemetry=telemetry,
        )

    def reset_scan(self) -> None:
        """Resets viewport back to empty default state in a thread-safe manner."""
        with self._lock:
            self.active_scan = None
            self.current_slice = 1
            self.zoom_factor = 1.0
