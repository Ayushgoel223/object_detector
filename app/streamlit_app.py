"""
BlindAid — Streamlit Mobile Web Application
=============================================
Streamlit Web App deployable to Streamlit Community Cloud (streamlit.io).
Supports camera frame capture, YOLO obstacle detection, 3-corridor path analysis,
Dijkstra indoor navigation, and audio voice instructions.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector         import ObjectDetector
from inference.spatial_analyzer import SpatialAnalyzer, Urgency
from inference.path_analyzer    import PathAnalyzer
from navigation.map_manager     import MapManager
from navigation.route_planner   import RoutePlanner


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BlindAid Mobile Navigation",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #ffffff; }
    h1, h2, h3 { color: #00e5ff !important; font-weight: 800; }
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        border: none !important;
        height: 50px !important;
        width: 100% !important;
    }
    .status-card {
        background: rgba(18, 26, 42, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    .voice-box {
        background: #004e92;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 5px solid #00ff87;
    }
    .voice-box-critical {
        background: #dc2626;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 5px solid #ff3366;
        animation: pulse 1s infinite alternate;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached AI Models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_ai_engine():
    detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))
    spatial  = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))
    path     = PathAnalyzer(config_path=str(ROOT / "config.yaml"))
    map_mgr  = MapManager()
    map_mgr.load_map(str(ROOT / "data" / "floorplan.jpg"))
    planner  = RoutePlanner(map_mgr) if map_mgr.is_loaded() else None
    return detector, spatial, path, map_mgr, planner


detector, spatial_analyzer, path_analyzer, map_manager, route_planner = load_ai_engine()


# ── App Header ─────────────────────────────────────────────────────────────────
st.title("👁️ BlindAid — AI Mobile Navigation")
st.caption("Real-Time Obstacle Detection • Floor Plan Route Guidance • Voice Alerts")

# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Navigation Controls")
    
    if map_manager.is_loaded():
        st.success(f"📍 Map Loaded: {map_manager.building_name}")
        locs = map_manager.get_location_names()
        
        from_loc = st.selectbox("Current Location", locs, index=0)
        to_loc   = st.selectbox("Destination", locs, index=min(2, len(locs)-1))
        
        if st.button("🚀 Plan & Start Route"):
            from_id = map_manager.find_location(from_loc)
            to_id   = map_manager.find_location(to_loc)
            if from_id and to_id:
                steps = route_planner.plan_route(from_id, to_id)
                st.session_state["nav_steps"] = steps
                st.session_state["nav_idx"] = 0
                st.session_state["nav_active"] = True
                st.rerun()
    else:
        st.warning("⚠️ Running in Camera-Only Mode")

    st.markdown("---")
    st.markdown("### 🎙️ Audio Settings")
    auto_speak = st.checkbox("Enable Auto Web Speech (TTS)", value=True)


# ── Active Route Display ───────────────────────────────────────────────────────
if st.session_state.get("nav_active") and route_planner and route_planner.is_navigating:
    steps = st.session_state.get("nav_steps", [])
    idx   = st.session_state.get("nav_idx", 0)

    if idx < len(steps):
        step = steps[idx]
        st.info(f"🚩 **Step {step.step_number} of {len(steps)}**: {step.instruction}")
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("➡️ Next Step"):
                st.session_state["nav_idx"] = idx + 1
                route_planner.advance_step()
                st.rerun()
        with cols[1]:
            if st.button("❌ Cancel Route"):
                st.session_state["nav_active"] = False
                route_planner.cancel_navigation()
                st.rerun()
    else:
        st.balloons()
        st.success("🎉 You have arrived at your destination!")
        st.session_state["nav_active"] = False


# ── Camera Input ───────────────────────────────────────────────────────────────
st.subheader("📷 Mobile Camera Feed")
camera_file = st.camera_input("Take a photo or point your camera ahead")

if camera_file is not None:
    # Convert image to OpenCV format
    img_pil = Image.open(camera_file)
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 1. Detect objects
    detections = detector.detect(frame_bgr)

    # 2. Path analysis
    path_res = path_analyzer.analyze(frame_bgr, detections)

    # 3. Obstacle warnings
    obstacle_insts = spatial_analyzer.analyze(detections, path_res)

    # Render Visual Overlay
    vis_frame = path_analyzer.draw_overlay(frame_bgr, path_res)
    vis_frame = detector.draw_detections(vis_frame, detections)
    vis_rgb   = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

    st.image(vis_rgb, use_column_width=True, caption="Live Path Corridors (Green = Clear, Red = Blocked)")

    # Voice Instruction Selection
    voice_msg = path_res.instruction
    is_critical = False

    if obstacle_insts:
        top_obs = obstacle_insts[0]
        if top_obs.urgency == Urgency.CRITICAL:
            voice_msg = top_obs.message
            is_critical = True
        elif top_obs.urgency == Urgency.NEAR:
            voice_msg = top_obs.message

    # Display Spoken Box
    box_class = "voice-box-critical" if is_critical else "voice-box"
    st.markdown(f'<div class="{box_class}">📢 {voice_msg}</div>', unsafe_allow_html=True)

    # HTML5 Web Speech Synthesis (Auto Speaks on Phone!)
    if auto_speak and voice_msg:
        escaped_msg = voice_msg.replace("'", "\\'").replace('"', '\\"')
        st.components.v1.html(f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{escaped_msg}");
                msg.rate = 1.0;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            }}
        </script>
        """, height=0)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("BlindAid v2 • Built for Visually Impaired Independent Mobility • Deployable to Streamlit Cloud")
