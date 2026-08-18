"""
BlindAid v2 — Main Application
=================================
Upgraded with:
  • 3-thread pipeline (Camera → Inference → Display)
  • CNN-based OCR (CRNN + EasyOCR fallback)
  • Text Interpreter (keyword + fuzzy matching)
  • MySQL / SQLite persistence
  • On-screen display overlay (voice removed)
  • RL agent integration (optional, loads if model exists)
  • Cloud sync: auto-downloads best model from HuggingFace on startup
  • Cloud DB: training metrics synced to Supabase (offline-trained models)

Architecture:
  Thread 1 (Camera)    : cap.read() → frame_queue (drops if full)
  Thread 2 (Inference) : YOLO + OCR + Spatial → result_queue
  Thread 3 (Main/UI)   : render + DB write + keyboard input
  Background           : cloud sync (startup) + health monitor

Controls (OpenCV window):
  Q / ESC  — Quit
  P        — Pause / Resume
  +/-      — Adjust confidence threshold
"""

import os
# Load .env before anything else
_env_path = __import__('pathlib').Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if '=' in _line and not _line.startswith('#'):
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import cv2
import sys
import time
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BlindAid")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Cloud: Auto-download latest model on startup ──────────────────────────────

def _cloud_startup_sync():
    """
    Runs once at startup in a background thread.
    Downloads improved models from HuggingFace if the laptop was off
    and GitHub Actions trained a better model overnight.
    Also pulls cloud training metrics into local DB.
    """
    import threading

    def _sync():
        try:
            from cloud.model_registry import ModelRegistry
            registry = ModelRegistry()
            updated = registry.auto_update_on_startup(phase="all")
            if updated:
                logger.info("[Cloud] ✓ Better models downloaded — system upgraded automatically!")
            else:
                logger.info("[Cloud] Models are already at latest version.")
        except Exception as e:
            logger.debug(f"[Cloud] Startup sync skipped: {e}")

    t = threading.Thread(target=_sync, name="CloudStartupSync", daemon=True)
    t.start()

from inference.detector         import ObjectDetector
from inference.spatial_analyzer import SpatialAnalyzer
from inference.path_analyzer    import PathAnalyzer
from inference.cnn_ocr          import CNNOCRReader
from inference.text_interpreter import TextInterpreter
from app.pipeline               import FramePipeline
from app.display_manager        import DisplayManager
from database.db_manager        import DBManager

try:
    from cloud.cloud_db import CloudDB
    CLOUD_DB_AVAILABLE = True
except ImportError:
    CLOUD_DB_AVAILABLE = False


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(path: Path = None) -> dict:
    path = path or (ROOT / "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BlindAid v2 — AI Navigation Assistant")
    print("  CNN-OCR | 3-Phase ML | MySQL | Cloud Training | Multi-thread")
    print("=" * 60)

    # ── Step 0: Cloud startup sync (background — doesn't block) ──────────────
    logger.info("[Startup] Checking cloud for model updates...")
    _cloud_startup_sync()   # Non-blocking, runs in background thread

    cfg = load_config()

    # ── Database ─────────────────────────────────────────────────────────────
    logger.info("Initializing database...")
    db = DBManager(config=cfg)
    db_ok = db.start()
    if not db_ok:
        logger.warning("Database unavailable. Running without persistence.")

    # ── Components ───────────────────────────────────────────────────────────
    logger.info("Loading object detector (YOLOv8)...")
    detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))

    logger.info("Loading CNN-OCR reader...")
    ocr_reader = CNNOCRReader(config=cfg, model_dir=ROOT / "models")
    logger.info(f"  OCR engine: {ocr_reader.engine_name}")

    logger.info("Loading spatial analyzer...")
    spatial_analyzer = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))

    logger.info("Loading path analyzer...")
    path_analyzer = PathAnalyzer(config_path=str(ROOT / "config.yaml"))

    logger.info("Loading text interpreter...")
    text_interpreter = TextInterpreter()

    # Optional RL agent
    rl_agent = None
    rl_model_path = ROOT / "models" / "rl_agent.pt"
    if rl_model_path.exists():
        try:
            sys.path.insert(0, str(ROOT / "training"))
            from phase3_rl import DQNAgent, ACTION_NAMES
            rl_agent = DQNAgent()
            rl_agent.load(rl_model_path)
            rl_agent.policy_net.eval()
            logger.info("RL agent loaded.")
        except Exception as e:
            logger.warning(f"RL agent not loaded: {e}")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    logger.info("Starting 3-thread pipeline...")
    pipeline = FramePipeline(
        config=cfg,
        detector=detector,
        ocr_reader=ocr_reader,
        spatial_analyzer=spatial_analyzer,
        text_interpreter=text_interpreter,
        path_analyzer=path_analyzer,
        db_manager=db if db_ok else None,
    )

    if not pipeline.start():
        logger.error("Cannot open camera. Check webcam connection.")
        if db_ok:
            db.stop()
        sys.exit(1)

    # ── Display Manager ───────────────────────────────────────────────────────
    display = DisplayManager(config=cfg)
    display.set_db_status(db_ok)

    cam_cfg     = cfg.get("camera", {})
    show_window = cam_cfg.get("display_overlay", True)
    paused      = False

    # FPS report timer
    fps_report_timer = time.time()
    total_frames     = 0

    logger.info("\nBlindAid v2 is RUNNING. Press Q to quit.\n")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── Keyboard input ────────────────────────────────────────────────
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):    # Q or ESC
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'Paused' if paused else 'Resumed'}.")

            if paused:
                time.sleep(0.05)
                continue

            # ── Get next result from pipeline ─────────────────────────────────
            result = pipeline.get_result(timeout=0.05)
            if result is None:
                continue

            total_frames += 1

            # ── RL Agent (optional) ───────────────────────────────────────────
            if rl_agent is not None:
                try:
                    state     = result.frame   # In full pipeline: use scene encoder
                    action, _ = rl_agent.select_action(
                        __import__("numpy").zeros(352, dtype=__import__("numpy").float32)
                    )
                    display.set_rl_action(
                        __import__("training.phase3_rl", fromlist=["ACTION_NAMES"]).ACTION_NAMES[action]
                        if hasattr(__import__("training"), "phase3_rl") else str(action)
                    )
                except Exception:
                    pass

            # ── Render display ────────────────────────────────────────────────
            if show_window:
                annotated = display.render(result.frame, result)
                cv2.imshow("BlindAid v2", annotated)

            # ── Periodic FPS + DB stats report ───────────────────────────────
            if time.time() - fps_report_timer >= 10.0:
                logger.info(f"FPS — Camera: {result.cam_fps:.1f} | "
                            f"Inference: {result.inf_fps:.1f} | "
                            f"Frames: {total_frames}")
                if db_ok:
                    db.update_session_fps(result.cam_fps, result.inf_fps, total_frames)
                fps_report_timer = time.time()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")

    finally:
        logger.info("\nShutting down...")
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()
        if db_ok:
            db.stop()
        logger.info("BlindAid v2 stopped. Stay safe!")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
