"""
BlindAid - Path Analyzer Module
=================================
Identifies the clearest walkable corridor in the camera frame
and provides both a visual overlay and navigation guidance.

How it works:
  1. Frame is divided into 3 vertical corridors: LEFT / CENTER / RIGHT
  2. Each corridor is scored by how much of its WALKABLE ZONE
     (bottom 65% of frame) is blocked by detected objects
  3. Edge detection finds floor/wall boundaries for better depth cues
  4. The clearest corridor is recommended with a colored overlay:
       GREEN  = recommended clear path
       YELLOW = caution, some obstacles
       RED    = blocked, do not go this way
  5. A navigation instruction is generated every N seconds
"""

import cv2
import numpy as np
import yaml
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from detector import Detection


# ── Enums ──────────────────────────────────────────────────────────────────────

class CorridorStatus(Enum):
    CLEAR    = "clear"
    CAUTION  = "caution"
    BLOCKED  = "blocked"


class RecommendedDir(Enum):
    LEFT   = "left"
    CENTER = "straight ahead"
    RIGHT  = "right"
    STOP   = "stop"


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class CorridorInfo:
    name: str                     # "LEFT", "CENTER", "RIGHT"
    status: CorridorStatus
    blockage_score: float         # 0.0 = clear, 1.0 = fully blocked
    x_start: int                  # pixel column start
    x_end: int                    # pixel column end


@dataclass
class PathResult:
    corridors: List[CorridorInfo]
    recommended: RecommendedDir
    best_corridor: CorridorInfo
    all_blocked: bool
    instruction: str              # Voice instruction text


# ── Path Analyzer ──────────────────────────────────────────────────────────────

class PathAnalyzer:
    """
    Analyzes detected objects to find the clearest walking path
    and renders a colored overlay on the frame.

    Usage:
        analyzer = PathAnalyzer()
        result = analyzer.analyze(frame, detections)
        frame_with_overlay = analyzer.draw_overlay(frame, result)
        print(result.instruction)
    """

    COLORS = {
        CorridorStatus.CLEAR:   (40, 200, 40),    # Green
        CorridorStatus.CAUTION: (0, 165, 255),    # Orange
        CorridorStatus.BLOCKED: (30, 30, 220),    # Red
    }

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        pc = cfg.get("path", {})
        self.walkable_zone_top  = pc.get("walkable_zone_top", 0.35)
        self.blocked_thresh     = pc.get("corridor_blocked_threshold", 0.25)
        self.caution_thresh     = pc.get("corridor_caution_threshold", 0.10)
        self.overlay_alpha      = pc.get("overlay_alpha", 0.30)
        self.show_overlay       = pc.get("show_path_overlay", True)
        self.path_announce_ivl  = pc.get("path_announce_interval", 4.0)

        self._last_path_announce = 0.0
        self._last_direction     = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, detections: List[Detection]) -> PathResult:
        """
        Analyze the frame and detections to find the best path.
        Returns a PathResult with corridor statuses and voice instruction.
        """
        h, w = frame.shape[:2]

        # Define 3 corridor pixel boundaries
        corridor_bounds = [
            (0,          w // 3,      "LEFT"),
            (w // 3,     2 * w // 3,  "CENTER"),
            (2 * w // 3, w,           "RIGHT"),
        ]

        walkable_top = int(h * self.walkable_zone_top)  # y pixel where walkable zone starts

        corridors: List[CorridorInfo] = []

        for x_start, x_end, name in corridor_bounds:
            score = self._compute_blockage_score(
                detections, x_start, x_end, w, h, walkable_top
            )

            if score >= self.blocked_thresh:
                status = CorridorStatus.BLOCKED
            elif score >= self.caution_thresh:
                status = CorridorStatus.CAUTION
            else:
                status = CorridorStatus.CLEAR

            corridors.append(CorridorInfo(
                name=name,
                status=status,
                blockage_score=score,
                x_start=x_start,
                x_end=x_end,
            ))

        # Find the best corridor (lowest blockage score)
        best = min(corridors, key=lambda c: c.blockage_score)
        all_blocked = all(c.status == CorridorStatus.BLOCKED for c in corridors)

        if all_blocked:
            recommended = RecommendedDir.STOP
        elif best.name == "LEFT":
            recommended = RecommendedDir.LEFT
        elif best.name == "RIGHT":
            recommended = RecommendedDir.RIGHT
        else:
            recommended = RecommendedDir.CENTER

        instruction = self._build_instruction(corridors, recommended, all_blocked)

        return PathResult(
            corridors=corridors,
            recommended=recommended,
            best_corridor=best,
            all_blocked=all_blocked,
            instruction=instruction,
        )

    def should_announce_path(self) -> bool:
        """Rate-limits path announcements to avoid spam."""
        now = time.time()
        if now - self._last_path_announce >= self.path_announce_ivl:
            self._last_path_announce = now
            return True
        return False

    def draw_overlay(self, frame: np.ndarray, result: PathResult) -> np.ndarray:
        """
        Draw colored corridor overlays and a direction arrow on the frame.
        Returns annotated frame.
        """
        if not self.show_overlay:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()
        walkable_top = int(h * self.walkable_zone_top)

        # Draw corridor overlays
        for corridor in result.corridors:
            color = self.COLORS[corridor.status]
            cv2.rectangle(
                overlay,
                (corridor.x_start, walkable_top),
                (corridor.x_end,   h),
                color,
                -1,   # filled
            )

        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        # Draw corridor labels + scores
        font       = cv2.FONT_HERSHEY_SIMPLEX
        for corridor in result.corridors:
            cx = (corridor.x_start + corridor.x_end) // 2
            color = self.COLORS[corridor.status]

            # Status text
            status_text = corridor.status.value.upper()
            tw, _ = cv2.getTextSize(status_text, font, 0.55, 2)[0], None
            cv2.putText(frame, status_text,
                        (cx - 35, walkable_top + 30),
                        font, 0.55, color, 2)

            # Blockage % text
            pct_text = f"{corridor.blockage_score * 100:.0f}% blocked"
            cv2.putText(frame, pct_text,
                        (cx - 45, walkable_top + 55),
                        font, 0.42, (220, 220, 220), 1)

        # Draw dividing lines between corridors
        cv2.line(frame, (w // 3,     walkable_top), (w // 3,     h), (200, 200, 200), 1)
        cv2.line(frame, (2 * w // 3, walkable_top), (2 * w // 3, h), (200, 200, 200), 1)

        # Draw horizontal line marking walkable zone boundary
        cv2.line(frame, (0, walkable_top), (w, walkable_top), (180, 180, 180), 1)

        # Draw direction arrow on the BEST corridor
        if not result.all_blocked:
            self._draw_arrow(frame, result.best_corridor, h, walkable_top)

        # Draw "BEST PATH" label
        bc = result.best_corridor
        best_cx = (bc.x_start + bc.x_end) // 2
        best_color = self.COLORS[bc.status]
        cv2.putText(frame, "BEST PATH",
                    (best_cx - 42, h - 15),
                    font, 0.52, best_color, 2)

        return frame

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _compute_blockage_score(
        self,
        detections: List[Detection],
        x_start: int, x_end: int,
        frame_w: int, frame_h: int,
        walkable_top: int,
    ) -> float:
        """
        Score how blocked a corridor is (0=clear, 1=fully blocked).

        Method:
        - For each detection whose bounding box overlaps this corridor
          AND is in the walkable zone (y > walkable_top),
          compute the overlapping area as a fraction of the corridor's walkable area.
        - Sum of all overlapping fractions = blockage score (capped at 1.0).
        """
        corridor_w = max(1, x_end - x_start)
        walkable_h = max(1, frame_h - walkable_top)
        corridor_area = corridor_w * walkable_h

        total_overlap = 0.0

        for det in detections:
            bx1, by1, bx2, by2 = det.bbox

            # Clip to walkable zone
            by1 = max(by1, walkable_top)
            if by1 >= by2:
                continue  # object is entirely above walkable zone

            # Clip to corridor horizontal bounds
            ox1 = max(bx1, x_start)
            ox2 = min(bx2, x_end)
            if ox1 >= ox2:
                continue  # no horizontal overlap

            overlap_area = (ox2 - ox1) * (by2 - by1)
            total_overlap += overlap_area / corridor_area

        return min(1.0, total_overlap)

    def _build_instruction(
        self,
        corridors: List[CorridorInfo],
        recommended: RecommendedDir,
        all_blocked: bool,
    ) -> str:
        if all_blocked:
            return "All paths are blocked. Please stop and wait."

        c_left, c_center, c_right = corridors[0], corridors[1], corridors[2]
        dir_name = recommended.value  # "left", "straight ahead", "right"

        if recommended == RecommendedDir.CENTER:
            if c_center.status == CorridorStatus.CLEAR:
                return "Path ahead is clear. Continue forward."
            else:
                return "Path ahead is mostly clear. Proceed carefully."

        elif recommended == RecommendedDir.LEFT:
            if c_left.status == CorridorStatus.CLEAR:
                return "Path on your left is clear. Move left."
            else:
                return "Best path is on your left. Move left carefully."

        elif recommended == RecommendedDir.RIGHT:
            if c_right.status == CorridorStatus.CLEAR:
                return "Path on your right is clear. Move right."
            else:
                return "Best path is on your right. Move right carefully."

        return f"Move {dir_name}."

    def _draw_arrow(
        self,
        frame: np.ndarray,
        corridor: CorridorInfo,
        frame_h: int,
        walkable_top: int,
    ):
        """Draw a direction arrow in the center of the best corridor."""
        cx = (corridor.x_start + corridor.x_end) // 2
        arrow_y_start = frame_h - 50
        arrow_y_end   = walkable_top + int((frame_h - walkable_top) * 0.4)

        color = self.COLORS[corridor.status]
        cv2.arrowedLine(
            frame,
            (cx, arrow_y_start),
            (cx, arrow_y_end),
            color,
            thickness=4,
            tipLength=0.3,
        )
