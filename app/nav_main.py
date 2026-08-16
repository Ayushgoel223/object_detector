"""
BlindAid v2 - Navigation App
==============================
Full indoor navigation assistant for blind users.

Modes:
  MAP MODE   — Floor plan loaded → turn-by-turn route + live camera obstacles
  CAMERA MODE — No map → pure camera path finding only

Controls:
  Type commands in the terminal (runs alongside camera window)
  Camera window: Q=quit, P=pause, M=mute, N=next step

Usage:
    python app/nav_main.py --map data/floorplan.jpg
    python app/nav_main.py                            # camera-only mode
"""

import cv2
import time
import yaml
import sys
import os
import threading
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector         import ObjectDetector
from inference.spatial_analyzer import SpatialAnalyzer, Urgency
from inference.voice_assistant  import VoiceAssistant
from inference.path_analyzer    import PathAnalyzer, CorridorStatus
from navigation.map_manager     import MapManager
from navigation.route_planner   import RoutePlanner
from navigation.command_parser  import CommandParser, CommandType


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def draw_hud(frame, fps, nav_state: dict, last_msg: str, paused: bool, muted: bool):
    """Draw the full HUD overlay on the frame."""
    h, w = frame.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    white  = (255, 255, 255)
    yellow = (0, 220, 255)
    green  = (50, 220, 50)
    red    = (50, 50, 220)
    cyan   = (255, 220, 50)

    # ── Bottom bar ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 68), font, 0.45, white, 1)
    cv2.putText(frame, "PAUSED" if paused else "LIVE",
                (w - 90, h - 68), font, 0.45,
                red if paused else green, 1)
    cv2.putText(frame, "MUTED" if muted else "AUDIO ON",
                (w - 90, h - 48), font, 0.40,
                red if muted else green, 1)

    if last_msg:
        msg = last_msg[:72] + "..." if len(last_msg) > 72 else last_msg
        cv2.putText(frame, f"> {msg}", (10, h - 10), font, 0.42, yellow, 1)

    # ── Top navigation bar ────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.putText(frame, "BlindAid v2", (10, 22), font, 0.65, white, 2)

    mode = nav_state.get("mode", "CAMERA")
    mode_color = cyan if mode == "MAP" else yellow
    cv2.putText(frame, f"Mode: {mode}", (10, 48), font, 0.50, mode_color, 1)

    if nav_state.get("navigating"):
        progress = nav_state.get("progress", "")
        cv2.putText(frame, progress, (w // 3, 22), font, 0.48, green, 1)
        dest = nav_state.get("destination", "")
        cv2.putText(frame, f"→ {dest}", (w // 3, 48), font, 0.48, yellow, 1)

    cv2.putText(frame, "[Q]Quit [P]Pause [M]Mute [N]NextStep",
                (w - 320, 60), font, 0.36, (160, 160, 160), 1)

    return frame


# ── Command Input Thread ───────────────────────────────────────────────────────

class CommandThread(threading.Thread):
    """Background thread that reads typed commands from stdin."""

    def __init__(self):
        super().__init__(daemon=True, name="CommandInput")
        self.command_queue = []
        self._lock = threading.Lock()

    def run(self):
        print("\n" + "─" * 55)
        print("  BlindAid Navigation — Type commands below")
        print("─" * 55)
        print("  Type 'help' for all commands")
        print("─" * 55 + "\n")
        while True:
            try:
                cmd = input("  ▶ ").strip()
                if cmd:
                    with self._lock:
                        self.command_queue.append(cmd)
            except EOFError:
                break

    def get_commands(self):
        with self._lock:
            cmds = list(self.command_queue)
            self.command_queue.clear()
        return cmds


# ── Main Navigation App ────────────────────────────────────────────────────────

def main(map_path: str = None):
    print("=" * 60)
    print("  BlindAid v2 — Indoor Navigation Assistant")
    print("=" * 60)

    cfg = load_config()
    cam_cfg = cfg["camera"]

    # ── Init all components ───────────────────────────────────────────────────
    print("\n[App] Initializing detector...")
    detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing path analyzer...")
    path_analyzer = PathAnalyzer(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing spatial analyzer...")
    obstacle_analyzer = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))

    print("[App] Initializing voice assistant...")
    voice = VoiceAssistant(config_path=str(ROOT / "config.yaml"))
    voice.start()

    print("[App] Initializing navigation engine...")
    map_manager  = MapManager()
    route_planner = None
    cmd_parser   = CommandParser()

    # ── Load Map ──────────────────────────────────────────────────────────────
    map_mode = False
    if map_path:
        mp = Path(map_path)
        if not mp.exists():
            # Try relative to data/ folder
            mp = ROOT / "data" / map_path
        if mp.exists():
            print(f"\n[App] Loading floor plan: {mp}")
            if map_manager.load_map(str(mp)):
                route_planner = RoutePlanner(map_manager)
                map_mode = True
                locs = map_manager.get_location_names()
                print(f"[App] Map loaded! Locations: {', '.join(locs)}")
                voice.speak_now(
                    f"Map loaded. {len(locs)} locations available. "
                    f"Type your current location to begin."
                )
            else:
                print("[App] Map load failed. Switching to camera-only mode.")
        else:
            print(f"[App] Map file not found: {mp}. Camera-only mode.")

    if not map_mode:
        voice.speak_now("BlindAid ready. Camera mode active. I will guide you using the live camera feed.")

    # ── Open Camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_cfg["source"])
    if not cap.isOpened():
        print("[App] ERROR: Cannot open camera.")
        voice.speak_now("Cannot open camera. Please check your webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["overlay_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["overlay_height"])

    show_display = cam_cfg["display_overlay"]
    target_fps   = cam_cfg["target_fps"]
    frame_ivl    = 1.0 / target_fps

    # ── Start command thread ──────────────────────────────────────────────────
    cmd_thread = CommandThread()
    cmd_thread.start()

    # ── Navigation state ──────────────────────────────────────────────────────
    current_location_id = None
    paused  = False
    muted   = False
    last_msg = ""
    fps     = 0.0
    fc      = 0
    fps_t   = time.time()

    nav_state = {
        "mode":        "MAP" if map_mode else "CAMERA",
        "navigating":  False,
        "progress":    "",
        "destination": "",
    }

    print(f"\n[App] Running in {'MAP' if map_mode else 'CAMERA'} mode. Type commands in terminal.\n")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        fc += 1
        el = time.time() - fps_t
        if el >= 1.0:
            fps = fc / el
            fc  = 0
            fps_t = time.time()

        # ── Keyboard input ─────────────────────────────────────────────────────
        if show_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                paused = not paused
                state = "Paused" if paused else "Resumed"
                if not muted:
                    voice.speak(state)
            elif key == ord('m'):
                muted = not muted
                if not muted:
                    voice.speak_now("Voice enabled")
            elif key == ord('n') and route_planner and route_planner.is_navigating:
                # Advance to next step
                next_s = route_planner.advance_step()
                if next_s:
                    voice.speak(next_s.instruction)
                    last_msg = next_s.instruction
                else:
                    dest_name = map_manager.locations.get(
                        route_planner.destination_id, {}
                    ).get("name", "destination")
                    msg = f"You have arrived at {dest_name}. Navigation complete."
                    voice.speak(msg)
                    last_msg = msg
                    nav_state["navigating"] = False

        # ── Process typed commands ─────────────────────────────────────────────
        for raw_cmd in cmd_thread.get_commands():
            response = _handle_command(
                raw_cmd, cmd_parser, map_manager, route_planner,
                voice, nav_state, current_location_id, map_mode
            )
            if response.get("location_update"):
                current_location_id = response["location_update"]
            if response.get("last_msg"):
                last_msg = response["last_msg"]

        if paused:
            if show_display:
                cv2.putText(frame, "PAUSED — Press P", (50, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 3)
                cv2.imshow("BlindAid v2", frame)
            time.sleep(0.05)
            continue

        # ── YOLO Detection ────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Path Analysis ────────────────────────────────────────────────────
        path_result  = path_analyzer.analyze(frame, detections)
        instructions = obstacle_analyzer.analyze(detections, path_result)

        # ── Voice: obstacle priority, then nav step, then path ────────────────
        if not muted:
            top = instructions[0] if instructions else None

            if top and top.urgency == Urgency.CRITICAL:
                voice.speak(top.message, critical=True)
                last_msg = top.message

            elif top and top.urgency == Urgency.NEAR:
                voice.speak(top.message)
                last_msg = top.message

            elif route_planner and route_planner.is_navigating:
                # Remind current navigation step every ~6 seconds
                step = route_planner.get_current_step()
                if step and path_analyzer.should_announce_path():
                    voice.speak(step.instruction)
                    last_msg = step.instruction

            elif path_analyzer.should_announce_path():
                voice.speak(path_result.instruction)
                last_msg = path_result.instruction

        # ── Navigation state update ───────────────────────────────────────────
        if route_planner:
            nav_state["navigating"] = route_planner.is_navigating
            if route_planner.is_navigating:
                nav_state["progress"] = route_planner.get_progress_summary()
                dest_loc = map_manager.locations.get(route_planner.destination_id, {})
                nav_state["destination"] = dest_loc.get("name", "")
            else:
                nav_state["progress"] = ""
                nav_state["destination"] = ""

        # ── Display ───────────────────────────────────────────────────────────
        if show_display:
            vis = path_analyzer.draw_overlay(frame, path_result)
            vis = detector.draw_detections(vis, detections)

            # Draw current nav step on frame
            if route_planner and route_planner.is_navigating:
                step = route_planner.get_current_step()
                if step:
                    _draw_nav_step(vis, step)

            vis = draw_hud(vis, fps, nav_state, last_msg, paused, muted)
            cv2.imshow("BlindAid v2", vis)

        # ── Frame rate cap ────────────────────────────────────────────────────
        proc = time.time() - loop_start
        sleep = frame_ivl - proc
        if sleep > 0:
            time.sleep(sleep)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[App] Shutting down...")
    voice.speak_now("BlindAid shutting down. Stay safe.")
    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    voice.stop()


# ── Command Handler ────────────────────────────────────────────────────────────

def _handle_command(
    raw_cmd, cmd_parser, map_manager, route_planner,
    voice, nav_state, current_location_id, map_mode
) -> dict:
    """Process a typed command. Returns dict with side-effects."""
    result = {}
    parsed = cmd_parser.parse(raw_cmd)
    print(f"  [Cmd] {parsed.type.value}: {parsed.argument or ''}")

    if parsed.type == CommandType.HELP:
        txt = cmd_parser.get_help_text()
        print(txt)
        voice.speak("Help information printed in the terminal.")
        result["last_msg"] = "Help shown in terminal"

    elif parsed.type == CommandType.LIST_PLACES:
        if map_mode and map_manager.is_loaded():
            places = map_manager.get_location_names()
            print(f"\n  Available locations ({len(places)}):")
            for p in places:
                print(f"    • {p}")
            print()
            voice.speak(f"There are {len(places)} locations. " + ", ".join(places[:5]))
            result["last_msg"] = f"Listed {len(places)} locations"
        else:
            msg = "No map loaded. Running in camera-only mode."
            print(f"  {msg}")
            voice.speak(msg)

    elif parsed.type == CommandType.SET_LOCATION:
        if not map_mode:
            voice.speak("No map loaded. I cannot track your location without a floor plan.")
            return result
        loc_id = map_manager.find_location(parsed.argument or "")
        if loc_id:
            loc_name = map_manager.locations[loc_id]["name"]
            desc = map_manager.get_location_description(loc_id)
            msg = f"Location set to {loc_name}. {desc}"
            print(f"  ✓ {msg}")
            voice.speak(msg)
            result["location_update"] = loc_id
            result["last_msg"] = msg
        else:
            msg = f"I don't recognise '{parsed.argument}'. Type 'list places' to see available locations."
            print(f"  ✗ {msg}")
            voice.speak(msg)

    elif parsed.type == CommandType.NAVIGATE_TO:
        if not map_mode:
            voice.speak("No map loaded. I can only give camera-based directions without a floor plan.")
            return result
        if not current_location_id:
            msg = "Please tell me where you are first. Type: I am at [location name]"
            print(f"  ! {msg}")
            voice.speak(msg)
            return result

        dest_id = map_manager.find_location(parsed.argument or "")
        if not dest_id:
            msg = f"I don't recognise '{parsed.argument}'. Type 'list places' to see all locations."
            print(f"  ✗ {msg}")
            voice.speak(msg)
            return result

        if dest_id == current_location_id:
            loc_name = map_manager.locations[dest_id]["name"]
            voice.speak(f"You are already at {loc_name}.")
            return result

        print(f"  Planning route: {current_location_id} → {dest_id}")
        steps = route_planner.plan_route(current_location_id, dest_id)

        if steps is None:
            msg = "I couldn't find a route to that location."
            voice.speak(msg)
            return result

        dest_name = map_manager.locations[dest_id]["name"]
        print(f"\n  Route to {dest_name} ({len(steps)} steps):")
        print(route_planner.format_full_route())
        print()

        msg = f"Route to {dest_name} found. {len(steps)} steps. Starting navigation now."
        voice.speak(msg)
        result["last_msg"] = msg

        # Announce first step
        if steps:
            time.sleep(1.5)
            first = route_planner.get_current_step()
            voice.speak(f"Step 1: {first.instruction}")
            result["last_msg"] = first.instruction

    elif parsed.type == CommandType.NEXT_STEP:
        if route_planner and route_planner.is_navigating:
            next_s = route_planner.advance_step()
            if next_s:
                voice.speak(next_s.instruction)
                result["last_msg"] = next_s.instruction
            else:
                dest_name = map_manager.locations.get(
                    route_planner.destination_id, {}
                ).get("name", "destination")
                msg = f"You have arrived at {dest_name}. Navigation complete!"
                print(f"  🏁 {msg}")
                voice.speak(msg)
                nav_state["navigating"] = False
                result["last_msg"] = msg
        else:
            voice.speak("No active navigation. Type 'go to [place]' to start.")

    elif parsed.type == CommandType.REPEAT:
        if route_planner and route_planner.is_navigating:
            step = route_planner.get_current_step()
            if step:
                voice.speak(step.instruction)
                result["last_msg"] = step.instruction

    elif parsed.type == CommandType.CANCEL:
        if route_planner and route_planner.is_navigating:
            route_planner.cancel_navigation()
            msg = "Navigation cancelled."
            voice.speak(msg)
            nav_state["navigating"] = False
            result["last_msg"] = msg
        else:
            voice.speak("No active navigation to cancel.")

    elif parsed.type == CommandType.WHERE_AM_I:
        if current_location_id and map_mode:
            name = map_manager.locations.get(current_location_id, {}).get("name", "unknown")
            voice.speak(f"You are at {name}.")
        else:
            voice.speak("Your current location is not set. Type: I am at [location name]")

    else:
        msg = f"I didn't understand '{raw_cmd}'. Type 'help' for commands."
        print(f"  ? {msg}")
        voice.speak(msg)

    return result


def _draw_nav_step(frame, step):
    """Draw the current navigation step prominently on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 72), (w, 130), (0, 60, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Wrap long instructions
    words = step.instruction.split()
    line1 = " ".join(words[:8])
    line2 = " ".join(words[8:16]) if len(words) > 8 else ""

    cv2.putText(frame, f"NAV Step {step.step_number}: {line1}",
                (10, 95), font, 0.48, (100, 255, 100), 1)
    if line2:
        cv2.putText(frame, line2, (10, 120), font, 0.48, (100, 255, 100), 1)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlindAid v2 Navigation App")
    parser.add_argument("--map", type=str, default=None,
                        help="Path to floor plan image (JPG/PNG)")
    args = parser.parse_args()

    main(map_path=args.map)
