"""SetFit-based Intent Classifier for edge inference."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Loads a fine-tuned SetFit intent classification model and runs edge inference."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize the intent classifier.

        Args:
            model_path: Absolute or relative directory path containing the SetFit model.
            device: Device to run inference on ("cpu" for Raspberry Pi).
        """
        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.labels: List[str] = []
        self._load_label_mapping()
        self._load_model()

    def _load_label_mapping(self) -> None:
        """Attempt to read label strings from config_setfit.json or config.json."""
        setfit_cfg = os.path.join(self.model_path, "config_setfit.json")
        if os.path.exists(setfit_cfg):
            try:
                with open(setfit_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "labels" in cfg and isinstance(cfg["labels"], list):
                        self.labels = cfg["labels"]
            except Exception as e:
                logger.warning(f"Could not load labels from config_setfit.json: {e}")

    def _load_model(self) -> None:
        """Load the SetFitModel using setfit library."""
        try:
            from setfit import SetFitModel
        except ImportError as e:
            raise ImportError(
                "The 'setfit' package is required. Install it using:\n"
                "pip install setfit"
            ) from e

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        logger.info(f"Loading SetFit intent model from: {self.model_path}")
        start_time = time.perf_counter()
        self.model = SetFitModel.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        if hasattr(self.model, "to") and self.device:
            try:
                self.model.to(self.device)
            except Exception:
                pass
        load_duration = time.perf_counter() - start_time
        logger.info(f"Model successfully loaded in {load_duration:.2f}s on {self.device}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single input text utterance.

        Args:
            text: Utterance string (e.g. 'zoom in', 'next slice').

        Returns:
            Dictionary containing 'intent', 'confidence', and 'latency_ms'.
        """
        start = time.perf_counter()

        # Run prediction
        raw_pred = self.model.predict([text])
        if hasattr(raw_pred, "tolist"):
            raw_pred = raw_pred.tolist()
        pred_val = raw_pred[0] if isinstance(raw_pred, (list, tuple, np.ndarray)) else raw_pred

        # Resolve intent name
        if isinstance(pred_val, int) and 0 <= pred_val < len(self.labels):
            intent_label = self.labels[pred_val]
        else:
            intent_label = str(pred_val)

        # Compute confidence if probabilities are supported by the classifier head
        confidence = 1.0
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([text])
                if hasattr(probs, "detach"):
                    probs = probs.detach().cpu().numpy()
                probs = np.array(probs)
                if probs.ndim >= 2:
                    confidence = float(np.max(probs[0]))
                elif probs.ndim == 1:
                    confidence = float(np.max(probs))
        except Exception:
            confidence = 1.0

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "text": text,
            "intent": intent_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of input text utterances."""
        return [self.predict(t) for t in texts]
