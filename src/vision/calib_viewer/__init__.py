"""Camera calibration viewer package for manual eye-in-hand offset calibration."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "CalibrationViewer":
        from .viewer import CalibrationViewer
        return CalibrationViewer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["CalibrationViewer"]
