"""
BlindAid — 3-Thread Frame Pipeline
=====================================
Producer/Consumer architecture for real-time inference at maximum FPS.

Thread layout:
  Thread 1 (Camera)   : cv2.VideoCapture → frame_queue
  Thread 2 (Inference): frame_queue → YOLO + OCR + Spatial → result_queue
  Thread 3 (Main)     : result_queue → Display + DB write + FPS tracking

Queues are bounded (size 3-5). If inference is slow, the camera thread
drops the oldest frame rather than blocking — maintaining real-time feel.

Usage:
    pipeline = FramePipeline(config, detector, ocr_reader, spatial_analyzer,
                              text_interpreter, db_manager)
    pipeline.start()
    try:
        while True:
            result = pipeline.get_result(timeout=0.1)
            if result:
                display_manager.render(result)
    except KeyboardInterrupt:
        pipeline.stop()
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """All inference outputs for a single frame, ready for display + DB write."""
    frame_id:       int
    frame:          np.ndarray                  # Original BGR frame
    detections:     list = field(default_factory=list)   # List[Detection]
    ocr_results:    list = field(default_factory=list)   # List[OCRResult]
    text_events:    list = field(default_factory=list)   # List[TextEvent]
    instructions:   list = field(default_factory=list)   # List[NavInstruction]
    path_result:    object = None
    cam_fps:        float = 0.0
    inf_fps:        float = 0.0
    timestamp:      float = field(default_factory=time.time)


# ── Frame Pipeline ────────────────────────────────────────────────────────────

class FramePipeline:
    """
    Manages the 3-stage camera→inference→display pipeline.
    All inter-thread communication via thread-safe queues.
    """

    def __init__(self, config: dict, detector, ocr_reader,
                 spatial_analyzer, text_interpreter, path_analyzer,
                 db_manager=None):
        self.config           = config
        self.detector         = detector
        self.ocr_reader       = ocr_reader
        self.spatial_analyzer = spatial_analyzer
        self.text_interpreter = text_interpreter
        self.path_analyzer    = path_analyzer
        self.db               = db_manager

        pl_cfg = config.get("pipeline", {})
        cam_cfg = config.get("camera", {})

        self._cam_queue_size = pl_cfg.get("camera_queue_size", 3)
        self._res_queue_size = pl_cfg.get("result_queue_size", 5)
        self._cam_source     = cam_cfg.get("source", 0)
        self._overlay_w      = cam_cfg.get("overlay_width", 960)
        self._overlay_h      = cam_cfg.get("overlay_height", 540)

        self._frame_queue  = queue.Queue(maxsize=self._cam_queue_size)
        self._result_queue = queue.Queue(maxsize=self._res_queue_size)
        self._stop_event   = threading.Event()

        self._cam_thread = threading.Thread(
            target=self._camera_loop, name="Camera-Thread", daemon=True
        )
        self._inf_thread = threading.Thread(
            target=self._inference_loop, name="Inference-Thread", daemon=True
        )

        # FPS tracking
        self._cam_fps    = 0.0
        self._inf_fps    = 0.0
        self._cam_count  = 0
        self._inf_count  = 0
        self._fps_timer  = time.time()
        self._fps_lock   = threading.Lock()

        self._frame_id   = 0
        self._cap        = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open camera and launch threads. Returns False if camera fails."""
        logger.info(f"[Pipeline] Opening camera source: {self._cam_source}")
        self._cap = cv2.VideoCapture(self._cam_source)
        if not self._cap.isOpened():
            logger.error("[Pipeline] Cannot open camera!")
            return False

        # Request max FPS from camera driver
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._overlay_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._overlay_h)
        self._cap.set(cv2.CAP_PROP_FPS, 60)          # driver will cap at hardware max
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # minimize latency

        self._cam_thread.start()
        self._inf_thread.start()
        logger.info("[Pipeline] All threads started.")
        return True

    def stop(self):
        """Signal all threads to stop and release resources."""
        logger.info("[Pipeline] Stopping...")
        self._stop_event.set()

        # Unblock threads waiting on queues
        for _ in range(10):
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                break
        for _ in range(10):
            try:
                self._result_queue.put_nowait(None)
            except queue.Full:
                break

        self._cam_thread.join(timeout=3.0)
        self._inf_thread.join(timeout=5.0)

        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("[Pipeline] Stopped.")

    def get_result(self, timeout: float = 0.05) -> Optional[FrameResult]:
        """
        Called by main thread to retrieve the next processed result.
        Returns None on timeout (no new result yet).
        """
        try:
            result = self._result_queue.get(timeout=timeout)
            if result is None:
                return None
            return result
        except queue.Empty:
            return None

    @property
    def camera_fps(self) -> float:
        return self._cam_fps

    @property
    def inference_fps(self) -> float:
        return self._inf_fps

    # ── Stage 1: Camera Thread ────────────────────────────────────────────────

    def _camera_loop(self):
        """
        Reads frames from camera as fast as possible.
        Drops oldest queued frame if inference can't keep up.
        """
        logger.info("[Camera] Thread started.")
        cam_count = 0
        fps_timer = time.time()

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("[Camera] Frame read failed. Retrying...")
                time.sleep(0.1)
                continue

            cam_count += 1
            self._frame_id += 1

            # Update cam FPS every second
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                with self._fps_lock:
                    self._cam_fps = cam_count / elapsed
                cam_count = 0
                fps_timer = time.time()

            # Non-blocking put: if queue full, drop oldest to stay real-time
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()   # discard oldest
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait((self._frame_id, frame))
            except queue.Full:
                pass   # This shouldn't happen after the drop above, but guard anyway

        logger.info("[Camera] Thread stopped.")

    # ── Stage 2: Inference Thread ─────────────────────────────────────────────

    def _inference_loop(self):
        """
        Pulls frames from camera queue, runs YOLO + OCR + Spatial.
        Pushes FrameResult to result_queue.
        """
        logger.info("[Inference] Thread started.")
        inf_count = 0
        fps_timer = time.time()

        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            frame_id, frame = item

            # ── YOLO Detection ──────────────────────────────────────────────
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                logger.warning(f"[Inference] YOLO error: {e}")
                detections = []

            # ── Path Analysis ───────────────────────────────────────────────
            try:
                path_result = self.path_analyzer.analyze(frame, detections)
            except Exception as e:
                logger.debug(f"[Inference] Path analyzer error: {e}")
                path_result = None

            # ── Spatial Analysis ────────────────────────────────────────────
            try:
                instructions = self.spatial_analyzer.analyze(detections, path_result)
            except Exception as e:
                logger.debug(f"[Inference] Spatial error: {e}")
                instructions = []

            # ── OCR ─────────────────────────────────────────────────────────
            try:
                ocr_results = self.ocr_reader.read(frame)
            except Exception as e:
                logger.debug(f"[Inference] OCR error: {e}")
                ocr_results = []

            # ── Text Interpretation ─────────────────────────────────────────
            try:
                text_events = self.text_interpreter.interpret(ocr_results)
            except Exception as e:
                logger.debug(f"[Inference] Text interp error: {e}")
                text_events = []

            # ── DB Writes (non-blocking) ─────────────────────────────────────
            if self.db and self.db._started:
                self._write_to_db(frame_id, detections, ocr_results, instructions)

            # ── FPS update ──────────────────────────────────────────────────
            inf_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                with self._fps_lock:
                    self._inf_fps = inf_count / elapsed
                    self._cam_fps_snapshot = self._cam_fps
                inf_count = 0
                fps_timer = time.time()

            # ── Push result ──────────────────────────────────────────────────
            result = FrameResult(
                frame_id=frame_id,
                frame=frame,
                detections=detections,
                ocr_results=ocr_results,
                text_events=text_events,
                instructions=instructions,
                path_result=path_result,
                cam_fps=self._cam_fps,
                inf_fps=self._inf_fps,
            )

            # Non-blocking put to result queue
            if self._result_queue.full():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass

        logger.info("[Inference] Thread stopped.")

    # ── DB Writer ─────────────────────────────────────────────────────────────

    def _write_to_db(self, frame_id: int, detections: list,
                      ocr_results: list, instructions: list):
        """Write significant events to DB asynchronously (non-blocking via DB queue)."""
        from inference.spatial_analyzer import Urgency

        # Only store NEAR or CRITICAL detections to avoid DB flooding
        for det in detections:
            # Find corresponding instruction urgency
            urgency = "FAR"
            for inst in instructions:
                if inst.object_label == det.label:
                    urgency = inst.urgency.name
                    zone    = inst.zone.value
                    break
            else:
                zone = "ahead"

            if urgency in ("NEAR", "CRITICAL"):
                try:
                    self.db.insert_detection(
                        frame_id=frame_id,
                        label=det.label,
                        confidence=det.confidence,
                        zone=zone,
                        urgency=urgency,
                        bbox=det.bbox,
                        center_x=det.center_x,
                        center_y=det.center_y,
                        area_fraction=det.area_fraction,
                    )
                except Exception:
                    pass

        for ocr_r in ocr_results:
            try:
                self.db.insert_ocr(
                    frame_id=frame_id,
                    raw_text=ocr_r.text,
                    cleaned_text=ocr_r.text.strip(),
                    confidence=ocr_r.confidence,
                    zone=ocr_r.zone,
                    semantic_tag=ocr_r.semantic_tag,
                    word_dynamics=ocr_r.char_dynamics[:20] if ocr_r.char_dynamics else [],
                )
            except Exception:
                pass
