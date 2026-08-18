"""
BlindAid — Display Manager (replaces Voice)
=============================================
Renders all inference results as on-screen text overlays using
PIL/Pillow (for Unicode + anti-aliased fonts) composited onto the
OpenCV frame.

Panels:
  ┌──────────────────────────────────────────────────┐
  │ TOP BAR: BlindAid v2 | Cam:30FPS | Inf:12FPS | DB│
  ├─────────────────────────┬────────────────────────┤
  │  VIDEO FRAME            │  OCR TEXT PANEL        │
  │  + YOLO bounding boxes  │  (detected words)      │
  ├─────────────────────────┴────────────────────────┤
  │ NAVIGATION PANEL: last 3 instructions (scrolling) │
  └──────────────────────────────────────────────────┘
"""

import logging
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("[Display] Pillow not available. Falling back to OpenCV text.")

# ── Color Palette ──────────────────────────────────────────────────────────────
# BGR format for OpenCV, (R,G,B) for PIL

C_BLACK       = (0,   0,   0)
C_WHITE       = (255, 255, 255)
C_YELLOW      = (0,   220, 255)   # BGR
C_RED         = (30,  30,  220)   # BGR
C_ORANGE      = (0,   165, 255)   # BGR
C_GREEN       = (50,  200, 50)    # BGR
C_CYAN        = (200, 200, 0)     # BGR
C_DARK_BG     = (15,  15,  15)    # BGR

PIL_YELLOW    = (255, 220,   0)
PIL_RED       = (220,  30,  30)
PIL_GREEN     = ( 50, 200,  50)
PIL_ORANGE    = (255, 165,   0)
PIL_WHITE     = (255, 255, 255)
PIL_DARK_BG   = ( 15,  15,  15)
PIL_CYAN      = (  0, 200, 200)


class DisplayManager:
    """
    Composites all overlays onto the camera frame:
      - Bounding boxes (color-coded urgency)
      - Navigation instruction panel (bottom strip, scrolling)
      - OCR text panel (right sidebar)
      - Status bar (top)
      - RL action indicator (top-right corner)
    """

    TOP_BAR_H    = 40
    NAV_PANEL_H  = 100
    OCR_PANEL_W  = 240

    def __init__(self, config: dict = None):
        self.config = config or {}
        disp_cfg = self.config.get("display", {})

        self._nav_history: deque = deque(maxlen=5)   # last 5 instructions
        self._ocr_history: deque = deque(maxlen=8)   # last 8 OCR results
        self._fade_entries: list = []                 # [(text, alpha, born_at)]
        self._font_loaded   = False
        self._font_lg       = None
        self._font_md       = None
        self._font_sm       = None

        self._db_status = "—"
        self._rl_action = ""

        self._load_fonts()

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray, result) -> np.ndarray:
        """
        Main render method. Composites all overlays and returns final frame.

        Args:
            frame   : BGR frame from camera
            result  : FrameResult from pipeline

        Returns:
            Annotated BGR frame ready for cv2.imshow()
        """
        vis = frame.copy()

        # Draw YOLO bounding boxes
        vis = self._draw_detections(vis, result.detections)

        # Draw OCR text bounding boxes (dashed yellow)
        vis = self._draw_ocr_boxes(vis, result.ocr_results)

        # Update history buffers
        if result.instructions:
            top_inst = result.instructions[0]
            self._nav_history.appendleft(top_inst.message)
        for ev in result.text_events:
            self._ocr_history.appendleft(ev.message)

        # Draw panels using PIL (better font quality)
        if PIL_AVAILABLE:
            vis = self._draw_panels_pil(vis, result)
        else:
            vis = self._draw_panels_cv(vis, result)

        return vis

    def set_db_status(self, connected: bool):
        self._db_status = "✓ DB" if connected else "✗ DB"

    def set_rl_action(self, action_name: str):
        self._rl_action = action_name

    # ── Bounding Boxes ────────────────────────────────────────────────────────

    def _draw_detections(self, vis: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Urgency color
            if det.area_fraction > 0.18:
                color = C_RED
                border = 3
            elif det.area_fraction > 0.07:
                color = C_ORANGE
                border = 2
            else:
                color = C_GREEN
                border = 1

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, border)

            label = f"{det.label} {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.50, 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4), font, scale, C_WHITE, thick)

        return vis

    def _draw_ocr_boxes(self, vis: np.ndarray, ocr_results: list) -> np.ndarray:
        for r in ocr_results:
            x1, y1, x2, y2 = r.bbox
            # Dashed yellow rectangle effect via line segments
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i+1], C_YELLOW, 2)

            tag_label = r.semantic_tag or r.text[:20]
            cv2.putText(vis, tag_label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_YELLOW, 1)
        return vis

    # ── PIL Panels ────────────────────────────────────────────────────────────

    def _draw_panels_pil(self, vis: np.ndarray, result) -> np.ndarray:
        """Draw all text panels using Pillow (Unicode, anti-aliased)."""
        h, w = vis.shape[:2]

        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        self._draw_top_bar(draw, img, w, result)
        self._draw_nav_panel(draw, img, w, h)
        self._draw_ocr_sidebar(draw, img, w, h)

        # Convert back to BGR numpy
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _draw_top_bar(self, draw: "ImageDraw.ImageDraw",
                       img: "Image.Image", w: int, result) -> None:
        """Top status bar with FPS, DB, RL action."""
        bar_h = self.TOP_BAR_H
        # Semi-transparent background overlay
        overlay = Image.new("RGBA", (w, bar_h), (*PIL_DARK_BG, 180))
        img.paste(overlay, (0, 0), overlay)

        cam_fps = getattr(result, "cam_fps", 0.0)
        inf_fps = getattr(result, "inf_fps", 0.0)

        text = (f"BlindAid v2  │  Cam: {cam_fps:.1f} FPS  "
                f"│  Inf: {inf_fps:.1f} FPS  │  {self._db_status}")
        if self._rl_action:
            text += f"  │  RL: {self._rl_action}"

        draw.text((10, 10), text, font=self._font_sm, fill=PIL_WHITE)

    def _draw_nav_panel(self, draw: "ImageDraw.ImageDraw",
                         img: "Image.Image", w: int, h: int) -> None:
        """Bottom navigation instruction strip."""
        panel_h = self.NAV_PANEL_H
        panel_w = w - self.OCR_PANEL_W
        y_start = h - panel_h

        overlay = Image.new("RGBA", (panel_w, panel_h), (10, 10, 30, 190))
        img.paste(overlay, (0, y_start), overlay)

        draw.text((10, y_start + 6), "NAVIGATION", font=self._font_sm,
                  fill=(100, 160, 255))

        nav_items = list(self._nav_history)[:3]
        y = y_start + 24
        for i, msg in enumerate(nav_items):
            alpha = 255 if i == 0 else max(100, 255 - i * 60)
            color = (*PIL_WHITE, alpha) if i == 0 else (180, 180, 200)
            prefix = "►" if i == 0 else "  "
            draw.text((14, y), f"{prefix} {msg}", font=self._font_sm, fill=color)
            y += 22

    def _draw_ocr_sidebar(self, draw: "ImageDraw.ImageDraw",
                           img: "Image.Image", w: int, h: int) -> None:
        """Right sidebar showing OCR-detected text."""
        sidebar_w = self.OCR_PANEL_W
        x_start   = w - sidebar_w
        sidebar_h = h - self.TOP_BAR_H

        overlay = Image.new("RGBA", (sidebar_w, sidebar_h), (20, 20, 10, 185))
        img.paste(overlay, (x_start, self.TOP_BAR_H), overlay)

        draw.text((x_start + 8, self.TOP_BAR_H + 8), "OCR TEXT",
                  font=self._font_sm, fill=PIL_YELLOW)

        y = self.TOP_BAR_H + 28
        ocr_items = list(self._ocr_history)[:7]
        for i, msg in enumerate(ocr_items):
            color = PIL_YELLOW if i == 0 else (180, 180, 100)
            # Word wrap at ~26 chars
            words = msg.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > 24:
                    draw.text((x_start + 8, y), line, font=self._font_sm, fill=color)
                    y += 18
                    line = word
                else:
                    line = (line + " " + word).strip()
            if line:
                draw.text((x_start + 8, y), line, font=self._font_sm, fill=color)
                y += 18

            y += 4   # gap between entries
            if y > h - 30:
                break

    # ── OpenCV fallback panels ─────────────────────────────────────────────────

    def _draw_panels_cv(self, vis: np.ndarray, result) -> np.ndarray:
        """Simpler OpenCV-only fallback when Pillow unavailable."""
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top bar
        cv2.rectangle(vis, (0, 0), (w, 36), C_DARK_BG, -1)
        cam_fps = getattr(result, "cam_fps", 0.0)
        inf_fps = getattr(result, "inf_fps", 0.0)
        bar_text = f"BlindAid v2 | Cam:{cam_fps:.1f} | Inf:{inf_fps:.1f} | {self._db_status}"
        cv2.putText(vis, bar_text, (10, 24), font, 0.5, C_WHITE, 1)

        # Nav panel (bottom)
        nav_h = 90
        cv2.rectangle(vis, (0, h - nav_h), (w - 200, h), (10, 10, 30), -1)
        cv2.putText(vis, "NAVIGATION", (10, h - nav_h + 18), font, 0.42, C_CYAN, 1)
        nav_items = list(self._nav_history)[:3]
        for i, msg in enumerate(nav_items):
            color = C_WHITE if i == 0 else (160, 160, 180)
            cv2.putText(vis, f"> {msg[:70]}", (14, h - nav_h + 36 + i * 20),
                        font, 0.42, color, 1)

        # OCR sidebar (right)
        cv2.rectangle(vis, (w - 200, 36), (w, h), (20, 20, 10), -1)
        cv2.putText(vis, "OCR TEXT", (w - 195, 58), font, 0.42, C_YELLOW, 1)
        ocr_items = list(self._ocr_history)[:6]
        for i, msg in enumerate(ocr_items):
            color = C_YELLOW if i == 0 else (140, 140, 80)
            cv2.putText(vis, msg[:22], (w - 195, 78 + i * 20), font, 0.40, color, 1)

        return vis

    # ── Font Loading ──────────────────────────────────────────────────────────

    def _load_fonts(self):
        """Try to load a nice system font; fall back to PIL default."""
        if not PIL_AVAILABLE:
            return
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        loaded = False
        for fpath in font_candidates:
            try:
                self._font_lg = ImageFont.truetype(fpath, 20)
                self._font_md = ImageFont.truetype(fpath, 16)
                self._font_sm = ImageFont.truetype(fpath, 13)
                loaded = True
                logger.debug(f"[Display] Font loaded: {fpath}")
                break
            except Exception:
                continue

        if not loaded:
            self._font_lg = ImageFont.load_default()
            self._font_md = ImageFont.load_default()
            self._font_sm = ImageFont.load_default()
            logger.debug("[Display] Using PIL default font.")
