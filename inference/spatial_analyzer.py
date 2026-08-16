"""
BlindAid - Spatial Reasoning Engine
=====================================
Takes detected objects and produces human-readable
navigation instructions with direction and urgency.

Logic:
  - Frame split into LEFT / CENTER / RIGHT zones
  - BBox area fraction determines proximity (CRITICAL / NEAR / FAR)
  - Priority queue ensures most urgent obstacle is spoken first
"""

import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from pathlib import Path

# Import Detection from sibling module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from detector import Detection


# ── Enums ──────────────────────────────────────────────────────────────────────

class Zone(Enum):
    LEFT   = "left"
    CENTER = "ahead"
    RIGHT  = "right"

class Urgency(Enum):
    SAFE     = 0   # Object too small / far — skip
    FAR      = 1   # Visible but distant
    NEAR     = 2   # Getting close — warn
    CRITICAL = 3   # Immediate — stop / redirect


# ── Instruction Data Class ─────────────────────────────────────────────────────

@dataclass
class NavInstruction:
    """A single navigation instruction to be spoken."""
    message: str
    urgency: Urgency
    object_label: str
    zone: Zone
    priority_score: float        # Higher = announce sooner

    def __repr__(self):
        return f"[{self.urgency.name}] {self.message}"


# ── Spatial Analyzer ───────────────────────────────────────────────────────────

class SpatialAnalyzer:
    """
    Converts a list of Detection objects into prioritized NavInstructions.

    Usage:
        analyzer = SpatialAnalyzer()
        instructions = analyzer.analyze(detections, frame_width, frame_height)
        for inst in instructions:
            print(inst.message)
    """

    # Objects that must ALWAYS be announced to the visually impaired user
    ALWAYS_ANNOUNCE = {
        "obstacle", "person", "chair", "couch", "table", "dining table", "bed",
        "bottle", "laptop", "tv", "bench", "backpack", "suitcase", "door", "stairs",
        "potted plant", "cell phone", "car", "truck", "bus", "motorcycle", "stop sign"
    }

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        sp = cfg["spatial"]
        self.left_end      = sp["left_zone_end"]        # default 0.38
        self.right_start   = sp["right_zone_start"]     # default 0.62
        self.critical_size = sp["critical_size"]        # default 0.18
        self.near_size     = sp["near_size"]            # default 0.07
        self.far_size      = sp["far_size"]             # default 0.01

        # Priority maps
        prio_cfg = cfg.get("priority", {})
        self._priority_map = {}
        for obj in prio_cfg.get("critical_objects", []):
            self._priority_map[obj] = 100
        for obj in prio_cfg.get("high_objects", []):
            self._priority_map[obj] = 70
        for obj in prio_cfg.get("medium_objects", []):
            self._priority_map[obj] = 40

    # ── Public API ──────────────────────────────────────────────────────────────

    def analyze(self, detections: List[Detection], path_result=None) -> List[NavInstruction]:
        """
        Produce a prioritized list of NavInstructions from detections.
        Already sorted: CRITICAL first, then by object priority score.
        """
        instructions: List[NavInstruction] = []
        seen_labels = set()

        for det in detections:
            label = det.label
            zone = self._get_zone(det.center_x)
            urgency = self._get_urgency(det.area_fraction, label)

            if urgency == Urgency.SAFE:
                continue

            dedup_key = f"{label}_{zone.value}"
            if dedup_key in seen_labels:
                continue
            seen_labels.add(dedup_key)

            message = self._build_message(label, zone, urgency, path_result)
            priority = self._compute_priority(label, urgency)

            instructions.append(NavInstruction(
                message=message,
                urgency=urgency,
                object_label=label,
                zone=zone,
                priority_score=priority,
            ))

        instructions.sort(key=lambda i: i.priority_score, reverse=True)

        if not instructions:
            instructions.append(NavInstruction(
                message="Path is clear. You may proceed.",
                urgency=Urgency.FAR,
                object_label="none",
                zone=Zone.CENTER,
                priority_score=0,
            ))

        return instructions

    def get_top_instruction(self, detections: List[Detection], path_result=None) -> Optional[NavInstruction]:
        """Returns only the single most important instruction."""
        results = self.analyze(detections, path_result)
        return results[0] if results else None

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _get_zone(self, center_x: float) -> Zone:
        if center_x <= self.left_end:
            return Zone.LEFT
        elif center_x >= self.right_start:
            return Zone.RIGHT
        else:
            return Zone.CENTER

    def _get_urgency(self, area_fraction: float, label: str) -> Urgency:
        if area_fraction >= self.critical_size:
            return Urgency.CRITICAL
        elif area_fraction >= self.near_size:
            return Urgency.NEAR
        elif area_fraction >= self.far_size:
            return Urgency.FAR
        else:
            if label in self.ALWAYS_ANNOUNCE:
                return Urgency.FAR
            return Urgency.SAFE

    def _build_message(self, label: str, zone: Zone, urgency: Urgency, path_result=None) -> str:
        """Craft a natural-language instruction for the user guiding toward open paths."""
        direction = zone.value   # "left", "ahead", "right"

        # Check if an open corridor exists from path_result
        best_dir = None
        all_blocked = False
        if path_result is not None:
            best_dir = getattr(path_result, "recommended", None)
            all_blocked = getattr(path_result, "all_blocked", False)

        if urgency == Urgency.CRITICAL:
            if zone == Zone.CENTER:
                if best_dir and str(best_dir.value) == "left":
                    return f"{label} directly ahead. Please move to the left."
                elif best_dir and str(best_dir.value) == "right":
                    return f"{label} directly ahead. Please move to the right."
                elif all_blocked:
                    return f"Stop! {label} ahead and all paths are blocked. Stop immediately."
                else:
                    return f"{label} ahead. Step carefully to the left or right."
            elif zone == Zone.LEFT:
                return f"{label} close on your left. Please move to the right."
            else:
                return f"{label} close on your right. Please move to the left."

        elif urgency == Urgency.NEAR:
            if zone == Zone.CENTER:
                if best_dir and str(best_dir.value) == "left":
                    return f"{label} ahead. Please bear left."
                elif best_dir and str(best_dir.value) == "right":
                    return f"{label} ahead. Please bear right."
                else:
                    return f"Warning. {label} ahead. Slow down."
            elif zone == Zone.LEFT:
                return f"{label} on your left. Bear right."
            else:
                return f"{label} on your right. Bear left."

        elif urgency == Urgency.FAR:
            if zone == Zone.CENTER:
                return f"{label} detected ahead."
            else:
                return f"{label} detected on your {direction}."

        return f"{label} detected."

    def _compute_priority(self, label: str, urgency: Urgency) -> float:
        base = self._priority_map.get(label, 20)
        urgency_boost = {
            Urgency.CRITICAL: 1000,
            Urgency.NEAR:     100,
            Urgency.FAR:      0,
        }.get(urgency, 0)
        return base + urgency_boost
