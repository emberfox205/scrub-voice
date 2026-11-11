"""Edge Voice-to-Intent Pipeline for Raspberry Pi 4 (ARM Cortex-A72) and Linux workstations.

Compliant with: schemas/command_schema.json (Draft 2020-12)
Components:
  1. Lexical Normalizer (normalizer.py)
  2. Single-Pass SetFit Intent Classifier (setfit_bge-micro-v2_8k)
  3. Deterministic Slot Extractor & Safety Gatekeeper (slot_extractor.py)
  4. Validated 10-Key Production JSON Command Envelope
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import numpy as np
import torch
from setfit import SetFitModel

# Support both absolute and relative module imports
try:
    from nlu.normalizer import LexicalNormalizer
    from nlu.slot_extractor import SlotExtractor
except ImportError:
    try:
        from src.nlu.normalizer import LexicalNormalizer
        from src.nlu.slot_extractor import SlotExtractor
    except ImportError:
        from .normalizer import LexicalNormalizer
        from .slot_extractor import SlotExtractor

logger = logging.getLogger(__name__)

# Enforce strict offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Canonical mapping from SetFit monitor labels to schemas/command_schema.json intents
MODEL_TO_SCHEMA_INTENT: Dict[str, str] = {
    "ZOOM_IN": "zoom_display",
    "ZOOM_OUT": "zoom_display",
    "SLICE_NEXT": "navigate_slice",
    "SLICE_PREV": "navigate_slice",
    "GOTO_SLICE": "navigate_slice",
    "IMAGE_OPEN": "open_scan",
    "IMAGE_CLOSE": "close_scan",
    "VIEW_RESET": "reset_camera",
    "PAN": "move_camera",
    "LOG_EVENT": "log_event",
    "CONTRAST_SET": "open_scan",
    "OUT_OF_SCOPE": "unknown",
    # Pass-through for canonical names
    "pick_tool": "pick_tool",
    "park_arm": "park_arm",
    "clear_emergency": "clear_emergency",
    "halt_emergency": "halt_emergency",
    "open_scan": "open_scan",
    "close_scan": "close_scan",
    "zoom_display": "zoom_display",
    "navigate_slice": "navigate_slice",
    "log_event": "log_event",
    "move_camera": "move_camera",
    "tilt_camera": "tilt_camera",
    "reset_camera": "reset_camera",
    "unknown": "unknown",
}


def apply_safety_and_canonical_rules(raw_text: str, predicted_label: str, confidence: float) -> Tuple[str, float]:
    """Applies medical safety overrides and maps model labels to schema intents."""
    text_lower = raw_text.strip().lower()

    # Deterministic Emergency Overrides (Safety-critical; zero latency tolerance)
    if any(k in text_lower for k in ["emergency stop", "halt robot", "freeze", "hard stop"]):
        return "halt_emergency", 1.0
    if any(k in text_lower for k in ["emergency clear", "clear area", "back off", "evacuate"]):
        return "clear_emergency", 1.0

    # Deterministic Tool Handling & Robotics Overrides
    if any(k in text_lower for k in ["park arm", "standby robot", "retract arm", "dock arm"]):
        return "park_arm", 1.0
    if any(k in text_lower for k in ["hand", "pass", "give", "bring"]) and any(
        t in text_lower for t in ["scalpel", "blade", "scissors", "clamp", "hemostat", "forceps", "tweezers"]
    ):
        return "pick_tool", max(confidence, 0.95)

    # Camera Overrides
    if "tilt" in text_lower or "pitch" in text_lower or "yaw" in text_lower:
        return "tilt_camera", max(confidence, 0.90)
    if ("camera" in text_lower or "wand" in text_lower) and any(
        d in text_lower for d in ["left", "right", "up", "down", "forward", "backward"]
    ):
        return "move_camera", max(confidence, 0.90)

    # Canonical mapping from SetFit model label
    canonical = MODEL_TO_SCHEMA_INTENT.get(predicted_label, predicted_label.lower())
    return canonical, confidence


class VoiceToIntentPipeline:
    """Production Edge Voice-to-Intent Pipeline producing validated JSON commands."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
        dict_dir: Optional[str] = None,
        schema_path: Optional[str] = None,
        num_threads: int = 1,
    ):
        """Initialize the edge pipeline with CPU threading and pre-loaded dictionaries.

        Args:
            model_dir: Path to the fine-tuned SetFit model directory.
            base_dir: Root directory of the nlu module package.
            dict_dir: Directory containing clinical dictionary JSONs.
            schema_path: Path to command_schema.json.
            num_threads: Number of CPU threads for PyTorch inference (default: 1 for Pi 4).
        """
        self.base_dir = base_dir or str(Path(__file__).resolve().parent)
        self.dict_dir = dict_dir or os.path.join(self.base_dir, "dictionaries")

        # 1. Configure CPU core affinity / thread limits
        if hasattr(torch, "set_num_threads") and num_threads > 0:
            torch.set_num_threads(num_threads)

        # 2. Initialize Normalizer and Slot Extractor
        self.normalizer = LexicalNormalizer(dict_dir=self.dict_dir)
        self.slot_extractor = SlotExtractor(dict_dir=self.dict_dir)

        # 3. Locate & Load JSON Schema
        if schema_path is None:
            candidates = [
                os.path.join(self.dict_dir, "command_schema.json"),
                os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), "schemas", "command_schema.json"),
                os.path.join(self.base_dir, "schemas", "command_schema.json"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    schema_path = c
                    break

        if not schema_path or not os.path.exists(schema_path):
            raise FileNotFoundError(f"Command schema file not found in candidates: {schema_path}")

        self.schema_path = schema_path
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        # 4. Locate & Load SetFit Model
        resolved_model_dir = self._resolve_model_path(model_dir)
        self.model_dir = resolved_model_dir

        t0 = time.perf_counter()
        self.model = SetFitModel.from_pretrained(
            self.model_dir,
            local_files_only=True
        )
        if hasattr(self.model, "to"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.load_duration_s = time.perf_counter() - t0

        # 5. Extract Label Mapping
        self.labels: List[str] = self._resolve_labels()

        # 6. Warm up CPU execution cache
        self.command_counter = 0
        self._warmup()

    def _resolve_model_path(self, user_path: Optional[str]) -> str:
        """Find the SetFit model directory across expected locations."""
        if user_path and os.path.exists(user_path):
            return os.path.abspath(user_path)

        search_paths = [
            os.path.join(self.base_dir, "models", "setfit_bge-micro-v2_8k"),
            "/home/admin/Downloads/setfit_bge-micro-v2_8k",
            os.path.join(os.path.dirname(self.base_dir), "models", "setfit_bge-micro-v2_8k"),
            os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), "models", "setfit_bge-micro-v2_8k"),
        ]

        for path in search_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"SetFit model directory not found. Checked:\n"
            + "\n".join(f" - {p}" for p in search_paths)
            + "\nPlease provide the path using --model-dir."
        )

    def _resolve_labels(self) -> List[str]:
        """Resolve ordered list of intent labels from model head or configs."""
        if hasattr(self.model, "model_head") and hasattr(self.model.model_head, "classes_"):
            classes = self.model.model_head.classes_
            if classes is not None and len(classes) > 0:
                return [str(c) for c in classes]

        if hasattr(self.model, "labels") and self.model.labels:
            return [str(c) for c in self.model.labels]

        cfg_path = os.path.join(self.model_dir, "config_setfit.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "labels" in cfg and isinstance(cfg["labels"], list):
                        return [str(c) for c in cfg["labels"]]
            except Exception:
                pass

        raise RuntimeError("Unable to determine label list from SetFit model.")

    def _warmup(self) -> None:
        """Run two warmup inferences so PyTorch JIT kernels and caches are hot."""
        try:
            self.model.predict_proba(["warmup inference"])
            self.model.predict_proba(["system ready"])
        except Exception:
            pass

    def process(self, raw_transcript: str) -> Dict[str, Any]:
        """Execute full end-to-end voice-to-intent edge pipeline in a single pass.

        Args:
            raw_transcript: Spoken utterance text from ASR.

        Returns:
            Validated 10-key dictionary payload conforming to schemas/command_schema.json.
        """
        t0 = time.perf_counter()
        epoch_time = time.time()
        self.command_counter += 1

        # Stage 1: Lexical Normalization
        raw_text, normalized_text = self.normalizer.process(raw_transcript)

        # Stage 2: Single-Pass SetFit Intent & Confidence Estimation
        try:
            probs = self.model.predict_proba([normalized_text])[0]
            if hasattr(probs, "detach"):
                probs = probs.detach().cpu().numpy()
            probs = np.array(probs)

            best_idx = int(np.argmax(probs))
            predicted_intent = self.labels[best_idx]
            confidence = float(probs[best_idx])
        except Exception:
            pred = self.model.predict([normalized_text])[0]
            predicted_intent = str(pred)
            confidence = 0.95

        confidence = max(0.0, min(confidence, 1.0))

        # Map model label to schema intent & apply deterministic safety overrides
        canonical_intent, confidence = apply_safety_and_canonical_rules(
            raw_text=raw_transcript,
            predicted_label=predicted_intent,
            confidence=confidence,
        )

        # Stage 3: Slot Extraction & Safety Gatekeeper
        extracted_slots = self.slot_extractor.extract(canonical_intent, normalized_text)
        final_intent, final_cat, final_prio, sanitized_slots, status = (
            self.slot_extractor.validate_command(
                predicted_intent=canonical_intent,
                slots=extracted_slots,
                raw_text=raw_transcript,
                confidence=confidence,
            )
        )

        # Confidence Threshold Check for EXECUTE status
        if status == "EXECUTE" and confidence < 0.65:
            status = "AMBIGUOUS"

        # Emergency Enforcement
        if final_intent in ("clear_emergency", "halt_emergency"):
            final_prio = "HIGH"
            final_cat = "EMERGENCY"

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Stage 4: Top-Level Production Envelope (10 Mandatory Keys)
        now_dt = datetime.fromtimestamp(epoch_time)
        command_id = f"cmd_{now_dt.strftime('%m%d_%H%M%S')}_{self.command_counter:03d}"

        payload: Dict[str, Any] = {
            "command_id": command_id,
            "timestamp": round(epoch_time, 3),
            "latency_ms": latency_ms,
            "raw_transcript": raw_transcript,
            "intent": final_intent,
            "category": final_cat,
            "slots": sanitized_slots,
            "confidence": round(confidence, 2),
            "priority": final_prio,
            "status": status,
        }

        # Stage 5: Strict JSON Schema Validation (Draft 2020-12)
        jsonschema.validate(instance=payload, schema=self.schema)

        return payload
