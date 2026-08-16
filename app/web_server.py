"""
BlindAid Mobile Web Server
============================
Flask backend that serves a mobile-first Web Application for smartphones.
Receives camera frames from mobile browser, runs YOLOv8 + Path Analyzer,
and returns directional voice instructions & overlays.

Access from phone: http://<YOUR_PC_IP>:5000
"""

import cv2
import numpy as np
import base64
import json
import socket
import sys
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector         import ObjectDetector
from inference.spatial_analyzer import SpatialAnalyzer, Urgency
from inference.path_analyzer    import PathAnalyzer
from navigation.map_manager     import MapManager
from navigation.route_planner   import RoutePlanner
from navigation.command_parser  import CommandParser

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Global AI Systems ─────────────────────────────────────────────────────────
print("[Web] Loading Object Detector...")
detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))

print("[Web] Loading Spatial Analyzer...")
spatial_analyzer = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))

print("[Web] Loading Path Analyzer...")
path_analyzer = PathAnalyzer(config_path=str(ROOT / "config.yaml"))

print("[Web] Loading Map Engine...")
map_manager = MapManager()
map_loaded = map_manager.load_map(str(ROOT / "data" / "floorplan.jpg"))

route_planner = RoutePlanner(map_manager) if map_loaded else None
cmd_parser = CommandParser()

# Rate limiter for path speech on mobile
last_path_speech_time = 0.0


def get_local_ip():
    """Get PC local IP address on Wi-Fi network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def get_info():
    local_ip = get_local_ip()
    locations = map_manager.get_location_names() if map_manager.is_loaded() else []
    return jsonify({
        "status": "online",
        "local_ip": local_ip,
        "port": 5000,
        "mobile_url": f"http://{local_ip}:5000",
        "map_loaded": map_manager.is_loaded(),
        "building_name": map_manager.building_name if map_manager.is_loaded() else "Camera Only",
        "locations": locations,
    })


@app.route("/api/locations")
def get_locations():
    if not map_manager.is_loaded():
        return jsonify({"locations": []})
    locs = []
    for loc_id, loc in map_manager.locations.items():
        locs.append({
            "id": loc_id,
            "name": loc["name"],
            "description": loc.get("description", ""),
            "type": loc.get("type", "room"),
        })
    return jsonify({"locations": locs})


@app.route("/api/route", methods=["POST"])
def calculate_route():
    data = request.json or {}
    from_loc = data.get("from_location", "")
    to_loc = data.get("to_location", "")

    if not map_manager.is_loaded() or not route_planner:
        return jsonify({"error": "No map loaded"}), 400

    from_id = map_manager.find_location(from_loc)
    to_id = map_manager.find_location(to_loc)

    if not from_id:
        return jsonify({"error": f"Unknown start location: '{from_loc}'"}), 400
    if not to_id:
        return jsonify({"error": f"Unknown destination: '{to_loc}'"}), 400

    steps = route_planner.plan_route(from_id, to_id)
    if steps is None:
        return jsonify({"error": "No route found between these locations"}), 400

    formatted_steps = []
    for s in steps:
        formatted_steps.append({
            "step_number": s.step_number,
            "instruction": s.instruction,
            "from_name": s.from_name,
            "to_name": s.to_name,
            "distance_steps": s.distance_steps,
            "turn": s.turn_direction,
            "is_final": s.is_final,
        })

    return jsonify({
        "from_id": from_id,
        "from_name": map_manager.locations[from_id]["name"],
        "to_id": to_id,
        "to_name": map_manager.locations[to_id]["name"],
        "total_steps_count": len(steps),
        "steps": formatted_steps,
    })


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """
    Receives base64 encoded frame from mobile camera.
    Runs YOLO object detection & corridor path analysis.
    Returns JSON with detections, corridors, and text to speak.
    """
    global last_path_speech_time
    data = request.json or {}
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    # Strip header if data URL
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    if frame is None:
        return jsonify({"error": "Decoded frame is empty"}), 400

    h, w = frame.shape[:2]

    # 1. Detect objects
    detections = detector.detect(frame)

    # 2. Path Corridor Analysis
    path_res = path_analyzer.analyze(frame, detections)

    # 3. Obstacle Warnings
    obstacle_insts = spatial_analyzer.analyze(detections, path_res)

    # Prepare return structure
    det_list = []
    for d in detections:
        det_list.append({
            "label": d.label,
            "confidence": round(d.confidence, 2),
            "bbox": d.bbox,   # [x1, y1, x2, y2]
            "center_x": round(d.center_x, 2),
            "center_y": round(d.center_y, 2),
            "area_fraction": round(d.area_fraction, 3),
        })

    corridors_list = []
    for c in path_res.corridors:
        corridors_list.append({
            "name": c.name,
            "status": c.status.value,
            "blockage": round(c.blockage_score, 2),
        })

    # Decide voice instruction to return
    voice_msg = path_res.instruction
    is_critical = False

    top_obs = obstacle_insts[0] if obstacle_insts else None
    if top_obs and top_obs.urgency == Urgency.CRITICAL:
        voice_msg = top_obs.message
        is_critical = True
    elif top_obs and top_obs.urgency in (Urgency.NEAR, Urgency.FAR) and top_obs.object_label != "none":
        voice_msg = top_obs.message

    return jsonify({
        "width": w,
        "height": h,
        "detections": det_list,
        "recommended_direction": path_res.recommended.value,
        "best_corridor": path_res.best_corridor.name,
        "all_blocked": path_res.all_blocked,
        "corridors": corridors_list,
        "path_instruction": path_res.instruction,
        "voice_msg": voice_msg,
        "is_critical": is_critical,
    })


# ── Server Start ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("=" * 60)
    print("  BlindAid v2 — Mobile Web Application Server")
    print("=" * 60)
    print(f"  PC Local URL : http://localhost:5000")
    print(f"  Mobile URL   : http://{local_ip}:5000")
    print("=" * 60)
    print("  Open http://{}:5000 on your phone browser!".format(local_ip))
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
