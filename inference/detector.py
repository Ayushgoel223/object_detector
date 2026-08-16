"""
BlindAid - Object Detector Module
==================================
Wraps YOLOv8 inference with COCO-class filtering.
Returns structured detection results per frame.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from ultralytics import YOLO


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object in a frame."""
    label: str                    # Human-readable class name
    class_id: int                 # COCO class index
    confidence: float             # Detection confidence [0, 1]
    bbox: tuple                   # (x1, y1, x2, y2) in pixels
    center_x: float               # Normalized center x [0, 1]
    center_y: float               # Normalized center y [0, 1]
    area_fraction: float          # BBox area / frame area [0, 1]


# ── COCO Class Map ─────────────────────────────────────────────────────────────

COCO_NAMES = {
    0: "obstacle", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}


# ── Detector Class ─────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Real-time object detector using YOLOv8.
    
    Auto-downloads 'yolov8n.pt' on first run (~6MB).
    Filters detections to navigation-relevant COCO classes only.
    Applies per-class confidence thresholds and temporal filtering
    to reduce false positives (e.g., cat on walls/clothes/photos).
    """

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg["model"]
        self.weights       = model_cfg["weights"]
        self.conf_thresh   = model_cfg["confidence_threshold"]
        self.iou_thresh    = model_cfg["iou_threshold"]
        self.img_size      = model_cfg["image_size"]
        self.device        = model_cfg["device"]
        self.active_ids    = set(cfg.get("active_classes", list(COCO_NAMES.keys())))

        # Per-class confidence overrides (e.g., cat needs 0.75+)
        raw_overrides = model_cfg.get("class_confidence_overrides", {})
        self.class_conf_overrides = {int(k): float(v) for k, v in raw_overrides.items()}

        # Temporal filter: track how many consecutive frames each class appears
        self.temporal_frames = model_cfg.get("temporal_filter_frames", 3)
        self._temporal_counter: dict = {}   # class_id -> consecutive frame count
        self._temporal_confirmed: set = set()  # class_ids that passed the filter

        print(f"[Detector] Loading model: {self.weights} on {self.device}")
        self.model = YOLO(self.weights)
        self.model.to(self.device)
        print(f"[Detector] Ready. Watching {len(self.active_ids)} classes. "
              f"Temporal filter: {self.temporal_frames} frames. "
              f"Per-class overrides: {len(self.class_conf_overrides)} classes.")

    # ── Public API ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (OpenCV format).
        Returns list of Detection objects, filtered to active classes,
        applying per-class confidence thresholds and temporal filtering.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        frame_area = h * w

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
            stream=False,
        )

        raw_detections: List[Detection] = []
        seen_class_ids: set = set()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.active_ids:
                    continue

                conf = float(box.conf[0].item())

                # Apply per-class confidence override
                required_conf = self.class_conf_overrides.get(cls_id, self.conf_thresh)
                if conf < required_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bbox_area = max(1, (x2 - x1) * (y2 - y1))
                area_frac = bbox_area / frame_area
                label = COCO_NAMES.get(cls_id, f"object_{cls_id}")

                raw_detections.append(Detection(
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center_x=cx,
                    center_y=cy,
                    area_fraction=area_frac,
                ))
                seen_class_ids.add(cls_id)

        # ── Temporal Filter ──────────────────────────────────────────────────
        # Increment counter for seen classes, reset for unseen ones
        for cls_id in list(self._temporal_counter.keys()):
            if cls_id in seen_class_ids:
                self._temporal_counter[cls_id] = min(
                    self._temporal_counter[cls_id] + 1, self.temporal_frames + 2
                )
            else:
                self._temporal_counter[cls_id] = max(
                    0, self._temporal_counter[cls_id] - 1
                )
                if self._temporal_counter[cls_id] == 0:
                    self._temporal_confirmed.discard(cls_id)

        for cls_id in seen_class_ids:
            if cls_id not in self._temporal_counter:
                self._temporal_counter[cls_id] = 1
            if self._temporal_counter[cls_id] >= self.temporal_frames:
                self._temporal_confirmed.add(cls_id)

        # Only return detections whose class has passed the temporal filter
        detections = [
            d for d in raw_detections
            if d.class_id in self._temporal_confirmed
        ]

        detections.sort(key=lambda d: d.area_fraction, reverse=True)
        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for visual debugging.
        Returns annotated frame copy.
        """
        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # Color by urgency
            if d.area_fraction > 0.18:
                color = (0, 0, 220)    # Red = CRITICAL
            elif d.area_fraction > 0.07:
                color = (0, 165, 255)  # Orange = WARNING
            else:
                color = (0, 200, 80)   # Green = OK

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label_text = f"{d.label} {d.confidence:.0%}"
            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 0.55
            thickness   = 2
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Label background
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                        font, font_scale, (255, 255, 255), thickness)

        return vis
