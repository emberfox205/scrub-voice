"""
Deterministic Intent-Conditioned Slot Extraction & Validation Engine
Compliant with: schemas/command_schema.json (Draft 2020-12)
Target: Sub-1.0ms edge execution on ARM Cortex-A72 (Raspberry Pi 4) / x86.
"""

import os
import json
import re
import time
from typing import Dict, Any, Optional, Tuple, List

# ---------------------------------------------------------------------------
# Default Embedded Vocabularies & Fallbacks
# ---------------------------------------------------------------------------

DEFAULT_TOOLS = {
    "scalpel": ["scalpel", "knife", "ten blade", "blade", "10 blade"],
    "scissors": ["scissors", "iris", "metz", "metzenbaum", "surgical scissors", "iris scissors"],
    "hemostat": ["hemostat", "clamp", "snap", "artery clamp"],
    "forceps": ["forceps", "tweezers", "pickups", "pickup", "tissue forceps"]
}

DEFAULT_SCAN_TYPES = {
    "ct": ["ct", "ct scan", "computed tomography", "cat scan", "abdominal ct", "chest ct", "head ct"],
    "mri": ["mri", "magnetic resonance", "mr", "brain mri", "spine mri"],
    "xray": ["xray", "x-ray", "x ray", "radiograph", "chest x-ray"]
}

DEFAULT_CAMERA_MOVE_DIRS = {
    "LEFT": ["left"],
    "RIGHT": ["right"],
    "UP": ["up"],
    "DOWN": ["down"],
    "FORWARD": ["forward", "in", "closer", "ahead"],
    "BACKWARD": ["backward", "back", "away", "reverse"]
}

DEFAULT_CAMERA_TILT_DIRS = {
    "UP": ["up", "upward", "pitch up"],
    "DOWN": ["down", "downward", "pitch down"],
    "LEFT": ["left", "yaw left"],
    "RIGHT": ["right", "yaw right"]
}

DEFAULT_SLICE_DIRS = {
    "NEXT": ["next", "forward", "ahead", "advance"],
    "PREVIOUS": ["previous", "prev", "back", "backward", "reverse", "rewind"]
}

DEFAULT_CONSTRAINTS = {
    "zoom_display": {
        "min_factor": 1.1,
        "max_factor": 10.0,
        "default_factor": 2.0,
        "default_direction": "IN"
    },
    "navigate_slice": {
        "default_step_count": 1,
        "min_step_count": 1,
        "min_slice_index": 1
    },
    "move_camera": {
        "min_distance_cm": 1.0,
        "max_distance_cm": 25.0,
        "default_distance_cm": 5.0
    },
    "tilt_camera": {
        "min_angle_deg": 1.0,
        "max_angle_deg": 45.0,
        "default_angle_deg": 15.0
    },
    "fuzzy_matching": {
        "similarity_threshold": 0.80
    }
}

# ---------------------------------------------------------------------------
# Zero-Dependency String Similarity
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def string_similarity(s1: str, s2: str) -> float:
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(s1.lower(), s2.lower()) / max_len)

def find_canonical_in_dict(text: str, category_dict: Dict[str, List[str]], threshold: float = 0.82) -> Optional[str]:
    text_lower = text.lower()
    
    # 1. Exact substring check across aliases
    for canonical, aliases in category_dict.items():
        for alias in aliases:
            # Word boundary check
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                return canonical

    # 2. Token & bigram fuzzy matching
    words = re.findall(r"\w+", text_lower)
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")

    for canonical, aliases in category_dict.items():
        for token in tokens:
            for alias in aliases:
                if string_similarity(token, alias) >= threshold:
                    return canonical
    return None

# ---------------------------------------------------------------------------
# SlotExtractor Engine
# ---------------------------------------------------------------------------

class SlotExtractor:
    """
    Deterministic Slot Extractor & Strict Schema Format Validator.
    Conforms strictly to schemas/command_schema.json (Draft 2020-12).
    """

    def __init__(self, dict_dir: Optional[str] = None):
        if dict_dir is None:
            dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionaries")
        self.dict_dir = dict_dir
        self._load_dictionaries()

        # Regex patterns
        self.re_float = re.compile(r"(\d+(?:\.\d+)?)")
        self.re_zoom_factor = re.compile(r"(?:zoom|magnify|by|to|scale)\s*(?:in|out)?\s*(\d+(?:\.\d+)?)\s*(?:x|times)?", re.IGNORECASE)
        self.re_bare_factor = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|times)", re.IGNORECASE)
        self.re_distance_cm = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters?|centimetres?)", re.IGNORECASE)
        self.re_angle_deg = re.compile(r"(\d+(?:\.\d+)?)\s*(?:deg|degrees?)", re.IGNORECASE)
        self.re_slice_step = re.compile(r"(\d+)\s*(?:slices?|frames?|images?)", re.IGNORECASE)
        self.re_goto_slice = re.compile(r"(?:slice|frame|to\s+slice|number)\s*(\d+)", re.IGNORECASE)

    def _load_dictionaries(self):
        slots_path = os.path.join(self.dict_dir, "medical_slots.json")
        constraints_path = os.path.join(self.dict_dir, "slot_constraints.json")

        if os.path.exists(slots_path):
            try:
                with open(slots_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tools = data.get("tools", DEFAULT_TOOLS)
                    self.scan_types = data.get("scan_types", DEFAULT_SCAN_TYPES)
                    self.camera_move_dirs = data.get("camera_move_directions", DEFAULT_CAMERA_MOVE_DIRS)
                    self.camera_tilt_dirs = data.get("camera_tilt_directions", DEFAULT_CAMERA_TILT_DIRS)
                    self.slice_dirs = data.get("slice_directions", DEFAULT_SLICE_DIRS)
                    self.view_planes = data.get("view_planes", {})
            except Exception:
                self._use_default_slots()
        else:
            self._use_default_slots()

        if os.path.exists(constraints_path):
            try:
                with open(constraints_path, "r", encoding="utf-8") as f:
                    self.constraints = json.load(f)
            except Exception:
                self.constraints = DEFAULT_CONSTRAINTS
        else:
            self.constraints = DEFAULT_CONSTRAINTS

        self.threshold = self.constraints.get("fuzzy_matching", {}).get("similarity_threshold", 0.82)

    def _use_default_slots(self):
        self.tools = DEFAULT_TOOLS
        self.scan_types = DEFAULT_SCAN_TYPES
        self.camera_move_dirs = DEFAULT_CAMERA_MOVE_DIRS
        self.camera_tilt_dirs = DEFAULT_CAMERA_TILT_DIRS
        self.slice_dirs = DEFAULT_SLICE_DIRS
        self.view_planes = {}

    def extract(self, intent: str, text: str) -> Dict[str, Any]:
        """
        Extracts slot parameters conditioned on the intent.
        """
        intent = intent.lower()
        if intent == "pick_tool":
            return self._extract_pick_tool(text)
        elif intent == "open_scan":
            return self._extract_open_scan(text)
        elif intent == "close_scan":
            return self._extract_close_scan(text)
        elif intent in ("zoom_display", "zoom_in", "zoom_out"):
            return self._extract_zoom_display(text)
        elif intent == "navigate_slice":
            return self._extract_navigate_slice(text)
        elif intent == "log_event":
            return self._extract_log_event(text)
        elif intent == "move_camera":
            return self._extract_move_camera(text)
        elif intent == "tilt_camera":
            return self._extract_tilt_camera(text)
        elif intent in ("park_arm", "clear_emergency", "halt_emergency", "reset_camera", "unknown"):
            return {}
        else:
            return {}

    # -----------------------------------------------------------------------
    # Slot Extractors Per Intent
    # -----------------------------------------------------------------------

    def _extract_pick_tool(self, text: str) -> Dict[str, Any]:
        tool = find_canonical_in_dict(text, self.tools, threshold=self.threshold)
        if tool:
            return {"tool_identifier": tool}
        return {}

    def _extract_open_scan(self, text: str) -> Dict[str, Any]:
        st = find_canonical_in_dict(text, self.scan_types, threshold=self.threshold)
        if st:
            return {"scan_type": st}
        return {}

    def _extract_close_scan(self, text: str) -> Dict[str, Any]:
        st = find_canonical_in_dict(text, self.scan_types, threshold=self.threshold)
        if st:
            return {"scan_type": st}
        return {}

    def _extract_zoom_display(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        text_lower = text.lower()
        if re.search(r"\b(?:in|magnify|closer)\b", text_lower):
            slots["direction"] = "IN"
        elif re.search(r"\b(?:out|wider|reduce)\b", text_lower):
            slots["direction"] = "OUT"

        cfg = self.constraints.get("zoom_display") or self.constraints.get("zoom", DEFAULT_CONSTRAINTS["zoom_display"])
        min_f = cfg.get("min_factor", 1.1)
        max_f = cfg.get("max_factor", 10.0)

        m = self.re_zoom_factor.search(text) or self.re_bare_factor.search(text)
        if m:
            val = float(m.group(1))
            val = max(min_f, min(val, max_f))
            slots["factor"] = val

        if hasattr(self, "view_planes") and self.view_planes:
            vp = find_canonical_in_dict(text, self.view_planes, threshold=self.threshold)
            if vp:
                slots["view_plane"] = vp

        return slots

    def _extract_navigate_slice(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        # Check for absolute slice jump first: e.g. "go to slice 45", "jump to slice 12", "slice 45"
        m_goto = self.re_goto_slice.search(text)
        if m_goto:
            idx = int(m_goto.group(1))
            if idx >= 1:
                return {"slice_index": idx}

        # Check for relative direction: e.g. "next slice", "previous slice", "back 5 slices"
        d = find_canonical_in_dict(text, self.slice_dirs, threshold=self.threshold)
        if d:
            slots: Dict[str, Any] = {"direction": d}
            m_step = self.re_slice_step.search(text)
            if m_step:
                steps = int(m_step.group(1))
                if steps >= 1:
                    slots["step_count"] = steps
            return slots

        return {}

    def _extract_log_event(self, text: str) -> Dict[str, Any]:
        # Formats:
        # "log event: first incision, note patient vitals stable"
        # "record milestone: trocar insertion complete"
        m = re.search(
            r"(?:log\s+event|record\s+milestone|record\s+event|log\s+milestone|document\s+event)[:\s]+(.+?)(?:[,\s]+(?:note|comment)[:\s]+(.+?))?(?:\s+(?:now|immediately|please)|$)",
            text,
            re.IGNORECASE
        )
        if m:
            event_name = m.group(1).strip().rstrip(",")
            comment = m.group(2).strip() if m.group(2) else None
            slots: Dict[str, Any] = {"event_name": event_name}
            if comment:
                slots["comment"] = comment
            return slots

        # Fallback if no colon
        m_simple = re.search(r"(?:log|record|document)\s+(?:event|milestone)?\s*(.+)", text, re.IGNORECASE)
        if m_simple:
            event_name = m_simple.group(1).strip()
            if event_name:
                return {"event_name": event_name}

        return {}

    def _extract_move_camera(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        d = find_canonical_in_dict(text, self.camera_move_dirs, threshold=self.threshold)
        if d:
            slots["direction"] = d

        m_dist = self.re_distance_cm.search(text)
        if m_dist:
            dist = float(m_dist.group(1))
            cfg = self.constraints.get("move_camera", DEFAULT_CONSTRAINTS["move_camera"])
            if cfg["min_distance_cm"] <= dist <= cfg["max_distance_cm"]:
                slots["distance_cm"] = dist
        return slots

    def _extract_tilt_camera(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        d = find_canonical_in_dict(text, self.camera_tilt_dirs, threshold=self.threshold)
        if d:
            slots["direction"] = d

        m_ang = self.re_angle_deg.search(text)
        if m_ang:
            ang = float(m_ang.group(1))
            cfg = self.constraints.get("tilt_camera", DEFAULT_CONSTRAINTS["tilt_camera"])
            if cfg["min_angle_deg"] <= ang <= cfg["max_angle_deg"]:
                slots["angle_deg"] = ang
        return slots

    # -----------------------------------------------------------------------
    # Strict Format Gatekeeper: Intent & Slot Validation
    # -----------------------------------------------------------------------

    def validate_command(
        self,
        predicted_intent: str,
        slots: Dict[str, Any],
        raw_text: str,
        confidence: float
    ) -> Tuple[str, str, str, Dict[str, Any], str]:
        """
        Validates whether the predicted intent and extracted slots satisfy command grammar,
        domain triggers, and schema constraints.

        Returns: (final_intent, category, priority, sanitized_slots, status)
        """
        text_lower = raw_text.strip().lower()

        # RULE 1: If input is pure numbers or non-command gibberish (e.g. "423", "99")
        # and has no command trigger keywords: strictly OUT-OF-DOMAIN unknown!
        has_letters = bool(re.search(r"[a-zA-Z]", text_lower))
        if not has_letters:
            return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"

        # Check intent-specific triggers & slot requirements
        if predicted_intent == "pick_tool":
            # Must contain a pick verb OR a valid tool name
            has_verb = bool(re.search(r"\b(hand|pass|give|pick|bring)\b", text_lower))
            tool = slots.get("tool_identifier")
            if not tool:
                # If it had pick verb, it's ambiguous; if it had neither, it's OOD
                if has_verb:
                    return "pick_tool", "TOOL_HANDLING", "NORMAL", {}, "AMBIGUOUS"
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            if tool not in ["scalpel", "scissors", "hemostat", "forceps"]:
                return "pick_tool", "TOOL_HANDLING", "NORMAL", {}, "AMBIGUOUS"
            return "pick_tool", "TOOL_HANDLING", "NORMAL", {"tool_identifier": tool}, "EXECUTE"

        elif predicted_intent == "park_arm":
            has_trigger = bool(re.search(r"\b(park|standby|retract|dock|stow)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            return "park_arm", "TOOL_HANDLING", "NORMAL", {}, "EXECUTE"

        elif predicted_intent == "clear_emergency":
            has_trigger = bool(re.search(r"\b(clear|back off|evacuate|ascent|safe height)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            return "clear_emergency", "EMERGENCY", "HIGH", {}, "EXECUTE"

        elif predicted_intent == "halt_emergency":
            has_trigger = bool(re.search(r"\b(stop|halt|freeze|abort|e-stop|brake)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            return "halt_emergency", "EMERGENCY", "HIGH", {}, "EXECUTE"

        elif predicted_intent == "open_scan":
            scan = slots.get("scan_type")
            has_verb = bool(re.search(r"\b(open|show|display|load|bring up|view)\b", text_lower))
            if not scan:
                if has_verb:
                    return "open_scan", "MONITOR_DISPLAY", "NORMAL", {}, "AMBIGUOUS"
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            if scan not in ["ct", "mri", "xray"]:
                return "open_scan", "MONITOR_DISPLAY", "NORMAL", {}, "AMBIGUOUS"
            return "open_scan", "MONITOR_DISPLAY", "NORMAL", {"scan_type": scan}, "EXECUTE"

        elif predicted_intent == "close_scan":
            has_trigger = bool(re.search(r"\b(close|dismiss|exit|shut down|hide)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            scan = slots.get("scan_type")
            clean_slots = {}
            if scan and scan in ["ct", "mri", "xray"]:
                clean_slots["scan_type"] = scan
            return "close_scan", "MONITOR_DISPLAY", "NORMAL", clean_slots, "EXECUTE"

        elif predicted_intent == "zoom_display":
            has_trigger = bool(re.search(r"\b(zoom|magnify|magnification)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            clean_slots = {}
            if "direction" in slots and slots["direction"] in ["IN", "OUT"]:
                clean_slots["direction"] = slots["direction"]
            if "factor" in slots and isinstance(slots["factor"], (int, float)):
                if 1.1 <= slots["factor"] <= 10.0:
                    clean_slots["factor"] = float(slots["factor"])
            return "zoom_display", "MONITOR_DISPLAY", "NORMAL", clean_slots, "EXECUTE"

        elif predicted_intent == "navigate_slice":
            # Must contain slice navigation terms
            has_trigger = bool(re.search(r"\b(slice|slices|frame|frames|advance|jump|step|next|previous|prev|back|rewind)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            
            clean_slots = {}
            if "slice_index" in slots and isinstance(slots["slice_index"], int) and slots["slice_index"] >= 1:
                clean_slots["slice_index"] = slots["slice_index"]
                return "navigate_slice", "MONITOR_DISPLAY", "NORMAL", clean_slots, "EXECUTE"

            if "direction" in slots and slots["direction"] in ["NEXT", "PREVIOUS"]:
                clean_slots["direction"] = slots["direction"]
                if "step_count" in slots and isinstance(slots["step_count"], int) and slots["step_count"] >= 1:
                    clean_slots["step_count"] = slots["step_count"]
                return "navigate_slice", "MONITOR_DISPLAY", "NORMAL", clean_slots, "EXECUTE"

            # Neither direction nor slice_index present
            return "navigate_slice", "MONITOR_DISPLAY", "NORMAL", {}, "AMBIGUOUS"

        elif predicted_intent == "log_event":
            has_trigger = bool(re.search(r"\b(log|record|milestone|event|document)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            event_name = slots.get("event_name")
            if not event_name or len(event_name) < 1:
                return "log_event", "MONITOR_DISPLAY", "NORMAL", {}, "AMBIGUOUS"
            clean_slots = {"event_name": event_name}
            if "comment" in slots and slots["comment"]:
                clean_slots["comment"] = slots["comment"]
            return "log_event", "MONITOR_DISPLAY", "NORMAL", clean_slots, "EXECUTE"

        elif predicted_intent == "move_camera":
            has_trigger = bool(re.search(r"\b(camera|endoscope|wand|move|jog|shift)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            direction = slots.get("direction")
            if not direction or direction not in ["LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACKWARD"]:
                return "move_camera", "CAMERA_LIGHT", "NORMAL", {}, "AMBIGUOUS"
            clean_slots = {"direction": direction}
            if "distance_cm" in slots and isinstance(slots["distance_cm"], (int, float)):
                if 1.0 <= slots["distance_cm"] <= 25.0:
                    clean_slots["distance_cm"] = float(slots["distance_cm"])
            return "move_camera", "CAMERA_LIGHT", "NORMAL", clean_slots, "EXECUTE"

        elif predicted_intent == "tilt_camera":
            has_trigger = bool(re.search(r"\b(tilt|pitch|yaw|angle)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            direction = slots.get("direction")
            if not direction or direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
                return "tilt_camera", "CAMERA_LIGHT", "NORMAL", {}, "AMBIGUOUS"
            clean_slots = {"direction": direction}
            if "angle_deg" in slots and isinstance(slots["angle_deg"], (int, float)):
                if 1.0 <= slots["angle_deg"] <= 45.0:
                    clean_slots["angle_deg"] = float(slots["angle_deg"])
            return "tilt_camera", "CAMERA_LIGHT", "NORMAL", clean_slots, "EXECUTE"

        elif predicted_intent == "reset_camera":
            has_trigger = bool(re.search(r"\b(reset|center|recenter|home)\b", text_lower))
            if not has_trigger:
                return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"
            return "reset_camera", "CAMERA_LIGHT", "NORMAL", {}, "EXECUTE"

        else:
            return "unknown", "UNKNOWN", "LOW", {}, "OUT-OF-DOMAIN"


# ---------------------------------------------------------------------------
# Self Verification
# ---------------------------------------------------------------------------

def run_tests():
    extractor = SlotExtractor()
    print("Testing SlotExtractor...")

    cases = [
        ("pick_tool", "hand me the scalpel", {"tool_identifier": "scalpel"}),
        ("pick_tool", "pass the iris scissors", {"tool_identifier": "scissors"}),
        ("pick_tool", "give me the clamp", {"tool_identifier": "hemostat"}),
        ("open_scan", "open CT scan", {"scan_type": "ct"}),
        ("open_scan", "show mri", {"scan_type": "mri"}),
        ("zoom_display", "zoom in 2x", {"direction": "IN", "factor": 2.0}),
        ("zoom_display", "zoom out", {"direction": "OUT"}),
        ("navigate_slice", "next slice", {"direction": "NEXT"}),
        ("navigate_slice", "advance 5 slices", {"direction": "NEXT", "step_count": 5}),
        ("navigate_slice", "go to slice 45", {"slice_index": 45}),
        ("move_camera", "camera left 5 cm", {"direction": "LEFT", "distance_cm": 5.0}),
        ("tilt_camera", "tilt down 15 degrees", {"direction": "DOWN", "angle_deg": 15.0}),
        ("reset_camera", "reset camera", {}),
        ("park_arm", "park arm", {}),
        ("halt_emergency", "emergency stop", {}),
        ("clear_emergency", "clear area", {})
    ]

    for intent, text, expected in cases:
        slots = extractor.extract(intent, text)
        assert slots == expected, f"Failed for {intent} '{text}': got {slots}, expected {expected}"
        intent_res, cat, prio, clean_slots, status = extractor.validate_command(intent, slots, text, 0.95)
        assert status == "EXECUTE", f"Failed validation for {text}: status {status}"

    # Test "423" bug
    print("Testing '423' OOD rejection...")
    slots_423 = extractor.extract("navigate_slice", "423")
    intent_423, cat_423, prio_423, clean_423, status_423 = extractor.validate_command("navigate_slice", slots_423, "423", 0.95)
    assert status_423 == "OUT-OF-DOMAIN", f"Expected OUT-OF-DOMAIN for '423', got {status_423}"
    assert intent_423 == "unknown", f"Expected intent 'unknown', got {intent_423}"

    # Test ambiguous command: "pass me that thing"
    slots_amb = extractor.extract("pick_tool", "pass me that thing")
    intent_amb, cat_amb, prio_amb, clean_amb, status_amb = extractor.validate_command("pick_tool", slots_amb, "pass me that thing", 0.90)
    assert status_amb == "AMBIGUOUS", f"Expected AMBIGUOUS, got {status_amb}"

    print("ALL SLOT EXTRACTOR TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
