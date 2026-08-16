"""
BlindAid - Main Application
==============================
Real-time blind navigation assistant.

Architecture:
  Thread 1 (main): Camera capture → YOLO inference → Spatial analysis
  Thread 2 (daemon): Voice TTS output

Controls (when display window is open):
  Q  — Quit
  P  — Pause / Resume
  +  — Increase confidence threshold
  -  — Decrease confidence threshold
  M  — Mute / Unmute voice
"""

import cv2
import time
import yaml
import sys
import threading
from pathlib import Path

# ── Path setup so sibling imports work ────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # project root (object_detetor/)
INFERENCE_DIR = ROOT / "inference"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INFERENCE_DIR))

from inference.detector         import ObjectDetector, Detection
from inference.spatial_analyzer import SpatialAnalyzer, Urgency
from inference.voice_assistant  import VoiceAssistant
from inference.path_analyzer    import PathAnalyzer, CorridorStatus


# ── Config Loader ──────────────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    if path is None:
        path = ROOT / "config.yaml"
    config_path = Path(path)
    if not config_path.exists():
        config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Stats Overlay ─────────────────────────────────────────────────────────────

def draw_stats(frame, fps: float, num_detections: int, paused: bool, muted: bool,
               last_msg: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black bar at bottom
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    small      = 0.50
    medium     = 0.62
    white      = (255, 255, 255)
    yellow     = (0, 220, 255)
    red        = (50, 50, 255)
    green      = (50, 220, 50)

    # FPS + Objects count
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 56), font, small, white, 1)
    cv2.putText(frame, f"Objects: {num_detections}", (10, h - 36), font, small, white, 1)

    # Status
    status_color = red if paused else green
    status_text  = "PAUSED" if paused else "RUNNING"
    cv2.putText(frame, status_text, (w - 110, h - 56), font, small, status_color, 2)

    mute_color = red if muted else green
    cv2.putText(frame, "MUTED" if muted else "AUDIO ON", (w - 110, h - 36),
                font, small, mute_color, 1)

    # Last spoken message
    if last_msg:
        msg = last_msg[:70] + "..." if len(last_msg) > 70 else last_msg
        cv2.putText(frame, f">> {msg}", (10, h - 10), font, small, yellow, 1)

    # Top-left watermark
    cv2.putText(frame, "BlindAid v1.0", (10, 28), font, medium, white, 2)
    cv2.putText(frame, "[Q]Quit  [P]Pause  [M]Mute  [+/-]Confidence", (10, 52),
                font, 0.40, (180, 180, 180), 1)

    return frame


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BlindAid — AI Navigation Assistant")
    print("  For Visually Impaired Users")
    print("=" * 60)

    cfg = load_config()
    cam_cfg = cfg["camera"]

    # ── Initialize Components ─────────────────────────────────────────────────
    print("\n[App] Initializing detector...")
    detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing spatial analyzer...")
    analyzer = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing path analyzer...")
    path_analyzer = PathAnalyzer(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing voice assistant...")
    voice = VoiceAssistant(config_path=str(ROOT / "config.yaml"))
    voice.start()
    voice.speak_now("BlindAid system starting. Camera is now active.")

    # ── Open Camera ───────────────────────────────────────────────────────────
    cam_source = cam_cfg["source"]
    print(f"\n[App] Opening camera: source={cam_source}")
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print("[App] ERROR: Cannot open camera. Check your webcam connection.")
        voice.speak_now("Error. Cannot open camera. Please check your webcam.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["overlay_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["overlay_height"])

    target_fps     = cam_cfg["target_fps"]
    show_display   = cam_cfg["display_overlay"]
    frame_interval = 1.0 / target_fps

    # ── State Variables ───────────────────────────────────────────────────────
    paused       = False
    muted        = False
    last_msg     = ""
    fps          = 0.0
    frame_count  = 0
    fps_timer    = time.time()
    conf_adjust  = 0.0    # Dynamic threshold adjustment

    print("\n[App] BlindAid is now RUNNING. Press Q to quit.\n")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while True:
        loop_start = time.time()

        # ── Read Frame ────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[App] Camera read failed. Retrying...")
            time.sleep(0.5)
            continue

        frame_count += 1

        # ── FPS Calculation ────────────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps       = frame_count / elapsed
            frame_count = 0
            fps_timer  = time.time()

        # ── Keyboard Input (non-blocking) ─────────────────────────────────────
        if show_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:    # Q or ESC to quit
                break
            elif key == ord('p'):
                paused = not paused
                state = "Paused" if paused else "Resumed"
                print(f"[App] {state}.")
                if not muted:
                    voice.speak(state)
            elif key == ord('m'):
                muted = not muted
                state = "Voice muted" if muted else "Voice enabled"
                print(f"[App] {state}.")
                if not muted:
                    voice.speak_now(state)
            elif key == ord('+') or key == ord('='):
                conf_adjust = min(conf_adjust + 0.05, 0.3)
                print(f"[App] Confidence threshold: {detector.conf_thresh + conf_adjust:.2f}")
            elif key == ord('-'):
                conf_adjust = max(conf_adjust - 0.05, -0.2)
                print(f"[App] Confidence threshold: {detector.conf_thresh + conf_adjust:.2f}")

        # ── Skip if paused ────────────────────────────────────────────────────
        if paused:
            if show_display:
                paused_frame = frame.copy()
                cv2.putText(paused_frame, "PAUSED — Press P to resume",
                            (50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 220), 3)
                cv2.imshow("BlindAid", paused_frame)
            time.sleep(0.05)
            continue

        # ── Inference ──────────────────────────────────────────────────
        orig_conf = detector.conf_thresh
        detector.conf_thresh = max(0.15, min(0.9, orig_conf + conf_adjust))
        detections = detector.detect(frame)
        detector.conf_thresh = orig_conf

        # ── Path Analysis ───────────────────────────────────────────────
        path_result = path_analyzer.analyze(frame, detections)

        # ── Obstacle Spatial Analysis ──────────────────────────────────────
        instructions = analyzer.analyze(detections)

        # ── Voice Output ───────────────────────────────────────────────
        if not muted:
            top = instructions[0] if instructions else None

            if top and top.urgency == Urgency.CRITICAL:
                # Critical obstacle: announce immediately, skip path guidance
                voice.speak(top.message, critical=True)
                last_msg = top.message

            elif top and top.urgency == Urgency.NEAR:
                # Near obstacle: warn
                voice.speak(top.message, critical=False)
                last_msg = top.message

            elif path_analyzer.should_announce_path():
                # No immediate danger: give path/direction guidance
                voice.speak(path_result.instruction, critical=False)
                last_msg = path_result.instruction

        # ── Display Overlay ──────────────────────────────────────────────
        if show_display:
            vis_frame = path_analyzer.draw_overlay(frame, path_result)  # path first
            vis_frame = detector.draw_detections(vis_frame, detections)  # objects on top
            vis_frame = draw_stats(vis_frame, fps, len(detections),
                                   paused, muted, last_msg)
            cv2.imshow("BlindAid", vis_frame)

        # ── Frame Rate Limiter ──────────────────────────────────────────────
        proc_time = time.time() - loop_start
        sleep_time = frame_interval - proc_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[App] Shutting down...")
    voice.speak_now("BlindAid shutting down. Goodbye.")
    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    voice.stop()
    print("[App] Done. Stay safe!")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
