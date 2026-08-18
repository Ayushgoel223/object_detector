"""
BlindAid — CNN-OCR Reader
===========================
High-level OCR API used by the inference pipeline.

Priority order:
  1. Custom CRNN model (models/cnn_ocr_supervised.pt) — if trained
  2. EasyOCR fallback                                  — always available

Outputs OCRResult objects with:
  - text, confidence
  - bbox (x1,y1,x2,y2) in pixel coordinates
  - zone (left/center/right)
  - char_dynamics: per-character confidence list (from CRNN)
  - semantic_tag: EXIT / STAIRS / DANGER / etc.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Try PyTorch imports ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from .model import (
        CRNN, TextRegionCNN, ctc_greedy_decode,
        BLANK_IDX, NUM_CHARS, get_device
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[OCR] PyTorch not available. Will use EasyOCR fallback.")

# ── Try EasyOCR fallback ──────────────────────────────────────────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("[OCR] EasyOCR not installed. OCR will be disabled.")

ROOT = Path(__file__).parent.parent.parent


# ── Data Class ────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """A single OCR detection result from one frame."""
    text:           str
    confidence:     float
    bbox:           Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    zone:           str                          # 'left' | 'center' | 'right'
    semantic_tag:   Optional[str] = None         # 'EXIT', 'STAIRS', etc.
    char_dynamics:  List[float]   = field(default_factory=list)  # per-char conf
    engine:         str           = "unknown"    # 'crnn' or 'easyocr'

    def __repr__(self):
        return (f"OCRResult('{self.text}' [{self.confidence:.0%}] "
                f"zone={self.zone} tag={self.semantic_tag})")


# ── Text Region Extractor (for custom CRNN path) ──────────────────────────────

class TextRegionExtractor:
    """
    Uses TextRegionCNN to produce candidate text-region bounding boxes
    from a heatmap, then extracts and normalizes crops for CRNN.
    """

    def __init__(self, model: "TextRegionCNN", device: "torch.device",
                 heatmap_thresh: float = 0.35):
        self.model  = model
        self.device = device
        self.thresh = heatmap_thresh

    def extract_regions(self, frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Returns list of (crop_gray_32h, bbox) where bbox=(x1,y1,x2,y2).
        crop_gray_32h is a (1,1,32,W) tensor ready for CRNN.
        """
        h, w = frame_bgr.shape[:2]

        # Preprocess for ResNet
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb  = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            heatmap = self.model(tensor)       # (1, 2, H/4, W/4)
        text_hmap = heatmap[0, 0].cpu().numpy()   # text probability map

        # Resize heatmap to original frame size
        hmap_resized = cv2.resize(text_hmap, (w, h))

        # Threshold + find contours
        binary = (hmap_resized > self.thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:     # too small — skip
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Expand bbox slightly
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            # Crop + convert to grayscale, resize height to 32
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            target_h = 32
            scale = target_h / gray.shape[0]
            target_w = max(1, int(gray.shape[1] * scale))
            gray_resized = cv2.resize(gray, (target_w, target_h))
            # Normalize
            gray_norm = (gray_resized.astype(np.float32) / 127.5) - 1.0
            # (1, 1, 32, W) tensor
            t = torch.from_numpy(gray_norm).unsqueeze(0).unsqueeze(0).float()
            regions.append((t, (x1, y1, x2, y2)))

        return regions


# ── CNN-OCR Reader ────────────────────────────────────────────────────────────

class CNNOCRReader:
    """
    Main OCR interface for the BlindAid pipeline.

    Run every N frames (configurable) to avoid CPU overload.
    Falls back from custom CRNN → EasyOCR → disabled gracefully.
    """

    # Navigation-critical keywords → semantic tags
    KEYWORD_TAGS = {
        "exit":      "EXIT",
        "stairs":    "STAIRS",
        "stair":     "STAIRS",
        "step":      "STAIRS",
        "elevator":  "ELEVATOR",
        "lift":      "ELEVATOR",
        "danger":    "DANGER",
        "warning":   "WARNING",
        "stop":      "STOP",
        "caution":   "CAUTION",
        "emergency": "EMERGENCY",
        "fire":      "FIRE",
        "restroom":  "RESTROOM",
        "toilet":    "RESTROOM",
        "bathroom":  "RESTROOM",
        "entrance":  "ENTRANCE",
        "enter":     "ENTRANCE",
        "push":      "DOOR_PUSH",
        "pull":      "DOOR_PULL",
        "no entry":  "NO_ENTRY",
    }

    def __init__(self, config: dict = None, model_dir: Path = None):
        self.config    = config or {}
        self.model_dir = model_dir or (ROOT / "models")
        ocr_cfg        = self.config.get("ocr", {})

        self.enabled        = ocr_cfg.get("enabled", True)
        self.min_confidence = ocr_cfg.get("min_confidence", 0.45)
        self.scan_interval  = ocr_cfg.get("scan_interval_frames", 10)
        self._frame_counter = 0
        self._last_results: List[OCRResult] = []

        # Zone boundaries (match spatial_analyzer defaults)
        sp_cfg = self.config.get("spatial", {})
        self._left_end     = sp_cfg.get("left_zone_end", 0.38)
        self._right_start  = sp_cfg.get("right_zone_start", 0.62)

        self._engine = "none"
        self._crnn: Optional["CRNN"] = None
        self._text_region_cnn: Optional["TextRegionCNN"] = None
        self._region_extractor: Optional[TextRegionExtractor] = None
        self._easyocr_reader = None
        self._device = None

        # Async background worker thread for zero camera lag
        import queue
        import threading
        self._ocr_input_queue = queue.Queue(maxsize=1)
        self._worker_thread = None
        self._stop_event = threading.Event()

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Try loading custom CRNN, then EasyOCR."""
        # ── Try custom CRNN ──
        crnn_path = self.model_dir / "cnn_ocr_supervised.pt"
        trcnn_path = self.model_dir / "text_region_cnn.pt"

        if TORCH_AVAILABLE and crnn_path.exists():
            try:
                self._device = get_device()
                self._crnn = CRNN(num_chars=NUM_CHARS).to(self._device)
                self._crnn.load_state_dict(
                    torch.load(crnn_path, map_location=self._device)
                )
                self._crnn.eval()

                tr_cnn = TextRegionCNN(pretrained=False).to(self._device)
                if trcnn_path.exists():
                    tr_cnn.load_state_dict(
                        torch.load(trcnn_path, map_location=self._device)
                    )
                tr_cnn.eval()
                self._text_region_cnn = tr_cnn
                self._region_extractor = TextRegionExtractor(tr_cnn, self._device)
                self._engine = "crnn"
                logger.info(f"[OCR] Custom CRNN loaded on {self._device}.")
                self._start_worker()
                return
            except Exception as e:
                logger.warning(f"[OCR] CRNN load failed ({e}). Falling back to EasyOCR.")

        # ── Try EasyOCR ──
        if EASYOCR_AVAILABLE:
            try:
                use_gpu = TORCH_AVAILABLE and (
                    torch.cuda.is_available() if TORCH_AVAILABLE else False
                )
                self._easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self._engine = "easyocr"
                logger.info(f"[OCR] EasyOCR initialized (gpu={use_gpu}).")
                self._start_worker()
            except Exception as e:
                logger.error(f"[OCR] EasyOCR init failed: {e}")
                self._engine = "none"
        else:
            logger.warning("[OCR] No OCR engine available. OCR disabled.")
            self._engine = "none"

    def _start_worker(self):
        """Start async OCR worker thread for zero lag."""
        import threading
        self._worker_thread = threading.Thread(
            target=self._ocr_worker_loop, name="OCR-Worker-Thread", daemon=True
        )
        self._worker_thread.start()

    def _ocr_worker_loop(self):
        """Asynchronous background loop processing OCR without blocking main pipeline."""
        import queue
        while not self._stop_event.is_set():
            try:
                frame_bgr = self._ocr_input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame_bgr is None:
                break

            try:
                # Enhance frame image clarity (CLAHE + unsharp sharpening)
                enhanced = self._enhance_frame_for_ocr(frame_bgr)

                if self._engine == "crnn":
                    raw_res = self._read_crnn(enhanced)
                elif self._engine == "easyocr":
                    raw_res = self._read_easyocr(enhanced)
                else:
                    raw_res = []

                results = [r for r in raw_res if r.confidence >= self.min_confidence]
                if results:
                    self._last_results = results
            except Exception as e:
                logger.debug(f"[OCR Worker] Read error: {e}")

    def _enhance_frame_for_ocr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Enhancement (CLAHE) + Unsharp Mask Sharpening
        to sharpen blurry textbook fonts and small medicine bottle labels.
        """
        try:
            h, w = frame_bgr.shape[:2]

            # 1. CLAHE Contrast Enhancement
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            # 2. Unsharp Masking for Sharp Text Edges
            blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 3)
            sharpened = cv2.addWeighted(enhanced_bgr, 1.4, blurred, -0.4, 0)

            return sharpened
        except Exception:
            return frame_bgr

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, frame_bgr: np.ndarray) -> List[OCRResult]:
        """
        Non-blocking async call. Enqueues frame for background OCR worker
        and returns latest cached results instantly (zero pipeline lag).
        """
        if not self.enabled or self._engine == "none":
            return []

        # Non-blocking put frame into worker queue
        if self._ocr_input_queue.full():
            try:
                self._ocr_input_queue.get_nowait()
            except Exception:
                pass
        try:
            self._ocr_input_queue.put_nowait(frame_bgr.copy())
        except Exception:
            pass

        return self._last_results

    def is_ready(self) -> bool:
        return self.enabled and self._engine != "none"

    # ── CRNN Path ─────────────────────────────────────────────────────────────

    def _read_crnn(self, frame_bgr: np.ndarray) -> List[OCRResult]:
        """Full CNN pipeline: TextRegionCNN → crops → CRNN → decoded text."""
        h, w = frame_bgr.shape[:2]
        regions = self._region_extractor.extract_regions(frame_bgr)

        results = []
        for crop_tensor, bbox in regions:
            try:
                crop_tensor = crop_tensor.to(self._device)
                with torch.no_grad():
                    log_probs = self._crnn(crop_tensor)   # (T, 1, C)
                texts = ctc_greedy_decode(log_probs)
                text = texts[0] if texts else ""
                if not text.strip():
                    continue

                # Estimate confidence as mean of max probabilities along sequence
                probs = log_probs.exp()[:, 0, :]          # (T, C)
                max_probs = probs.max(dim=1).values        # (T,)
                conf = float(max_probs.mean().item())

                # Per-char confidence (take top-probability at each timestep)
                char_dynamics = max_probs.tolist()

                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                zone = self._get_zone(cx)
                tag  = self._get_tag(text)

                results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    zone=zone,
                    semantic_tag=tag,
                    char_dynamics=char_dynamics,
                    engine="crnn",
                ))
            except Exception as e:
                logger.debug(f"[OCR] CRNN region decode error: {e}")

        return results

    # ── EasyOCR Path ──────────────────────────────────────────────────────────

    def _read_easyocr(self, frame_bgr: np.ndarray) -> List[OCRResult]:
        """EasyOCR path with enhanced text reading."""
        h, w = frame_bgr.shape[:2]
        try:
            raw = self._easyocr_reader.readtext(frame_bgr, detail=1, paragraph=False)
        except Exception as e:
            logger.warning(f"[OCR] EasyOCR read error: {e}")
            return []

        results = []
        for (pts, text, conf) in raw:
            if not text.strip():
                continue
            # pts is list of 4 corner points [[x,y], ...]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            cx = ((x1 + x2) / 2) / w
            zone = self._get_zone(cx)
            tag  = self._get_tag(text)

            results.append(OCRResult(
                text=text,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                zone=zone,
                semantic_tag=tag,
                char_dynamics=[],
                engine="easyocr",
            ))
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_zone(self, center_x_norm: float) -> str:
        if center_x_norm <= self._left_end:
            return "left"
        elif center_x_norm >= self._right_start:
            return "right"
        return "center"

    def _get_tag(self, text: str) -> Optional[str]:
        lower = text.lower().strip()
        for keyword, tag in self.KEYWORD_TAGS.items():
            if keyword in lower:
                return tag
        return None

    @property
    def engine_name(self) -> str:
        return self._engine


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read {img_path}")
            sys.exit(1)
    else:
        # Create a synthetic test image with text
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
        cv2.putText(frame, "EXIT", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(frame, "STAIRS ->", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,200,50), 4)

    reader = CNNOCRReader()
    # Force scan
    reader._frame_counter = reader.scan_interval - 1
    results = reader.read(frame)

    print(f"Engine: {reader.engine_name}")
    print(f"Results ({len(results)}):")
    for r in results:
        print(f"  {r}")
