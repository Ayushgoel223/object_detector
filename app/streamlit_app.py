"""
BlindAid — Streamlit Mobile Web Application
=============================================
High-Speed Live WebRTC Streamer + Mobile Web Speech Synthesis (TTS).
Optimized for 30+ FPS fast video processing on mobile & cloud servers.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import time
import threading
from pathlib import Path
import av

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector         import ObjectDetector
from inference.spatial_analyzer import SpatialAnalyzer, Urgency
from inference.path_analyzer    import PathAnalyzer
from navigation.map_manager     import MapManager
from navigation.route_planner   import RoutePlanner

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BlindAid Mobile Navigation",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# High-Contrast Mobile CSS
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
    .voice-box {
        background: #004e92;
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        border-left: 6px solid #00ff87;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 78, 146, 0.4);
    }
    .voice-box-critical {
        background: #dc2626;
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        border-left: 6px solid #ff3366;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
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
st.title("👁️ BlindAid — Real-Time Navigation")
st.caption("High-Speed 30FPS Camera Feed • Obstacle Guidance • Mobile Speech Output")


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
    auto_speak = st.checkbox("🔊 Enable Phone Voice Output", value=True)


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


# ── Fast High-FPS WebRTC Video Processor ───────────────────────────────────────
st.subheader("📹 Live Camera & Obstacle Tracking")

class FastVideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_path_res = None
        self.last_detections = []
        self.latest_speech_msg = ""
        self.is_critical_msg = False
        self.last_spoken_time = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        self.frame_count += 1

        # Run AI detection on downscaled image (320px) for 35+ FPS high speed
        if self.frame_count % 2 == 1 or self.last_path_res is None:
            small_img = cv2.resize(img, (320, 240))
            scale_x = w / 320.0
            scale_y = h / 240.0

            raw_dets = detector.detect(small_img)

            # Scale bboxes back to full size
            scaled_dets = []
            for d in raw_dets:
                x1, y1, x2, y2 = d.bbox
                scaled_bbox = (
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y)
                )
                d.bbox = scaled_bbox
                scaled_dets.append(d)

            self.last_detections = scaled_dets
            self.last_path_res = path_analyzer.analyze(img, scaled_dets)

            # Determine Voice Message
            obstacle_insts = spatial_analyzer.analyze(scaled_dets, self.last_path_res)
            top_obs = obstacle_insts[0] if obstacle_insts else None

            now = time.time()
            if top_obs and top_obs.urgency == Urgency.CRITICAL:
                self.latest_speech_msg = top_obs.message
                self.is_critical_msg = True
                self.last_spoken_time = now
            elif top_obs and top_obs.urgency == Urgency.NEAR:
                self.latest_speech_msg = top_obs.message
                self.is_critical_msg = False
                self.last_spoken_time = now
            elif now - self.last_spoken_time >= 4.0:
                self.latest_speech_msg = self.last_path_res.instruction
                self.is_critical_msg = False
                self.last_spoken_time = now

        # Draw overlays on high-res image
        vis_frame = path_analyzer.draw_overlay(img, self.last_path_res)
        vis_frame = detector.draw_detections(vis_frame, self.last_detections)

        return av.VideoFrame.from_ndarray(vis_frame, format="bgr24")


if WEBRTC_AVAILABLE:
    webrtc_ctx = webrtc_streamer(
        key="blindaid-fast-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FastVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": {"ideal": "environment"}
            },
            "audio": False
        },
        async_processing=True,
    )

    # Audio Banner & Web Speech Output for Mobile Browser
    if webrtc_ctx.video_processor:
        speech_text = webrtc_ctx.video_processor.latest_speech_msg
        is_crit     = webrtc_ctx.video_processor.is_critical_msg

        if speech_text:
            box_style = "voice-box-critical" if is_crit else "voice-box"
            st.markdown(f'<div class="{box_style}">📢 {speech_text}</div>', unsafe_allow_html=True)

            if auto_speak:
                escaped_text = speech_text.replace("'", "\\'").replace('"', '\\"')
                st.components.v1.html(f"""
                <script>
                    if ('speechSynthesis' in window) {{
                        var msg = new SpeechSynthesisUtterance("{escaped_text}");
                        msg.rate = 1.05;
                        msg.volume = 1.0;
                        if ({'true' if is_crit else 'false'}) {{
                            window.speechSynthesis.cancel();
                        }}
                        window.speechSynthesis.speak(msg);
                    }}
                </script>
                """, height=0)


# ── Snapshot Fallback Mode ─────────────────────────────────────────────────────
with st.expander("📷 Snapshot Photo Mode"):
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        img_pil = Image.open(camera_file)
        frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        detections = detector.detect(frame_bgr)
        path_res   = path_analyzer.analyze(frame_bgr, detections)
        obstacle_insts = spatial_analyzer.analyze(detections, path_res)

        vis_frame = path_analyzer.draw_overlay(frame_bgr, path_res)
        vis_frame = detector.draw_detections(vis_frame, detections)
        vis_rgb   = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

        st.image(vis_rgb, use_column_width=True)

        voice_msg = path_res.instruction
        is_critical = False

        if obstacle_insts:
            top_obs = obstacle_insts[0]
            if top_obs.urgency == Urgency.CRITICAL:
                voice_msg = top_obs.message
                is_critical = True
            elif top_obs.urgency == Urgency.NEAR:
                voice_msg = top_obs.message

        box_class = "voice-box-critical" if is_critical else "voice-box"
        st.markdown(f'<div class="{box_class}">📢 {voice_msg}</div>', unsafe_allow_html=True)

        if auto_speak and voice_msg:
            escaped_msg = voice_msg.replace("'", "\\'").replace('"', '\\"')
            st.components.v1.html(f"""
            <script>
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{escaped_msg}");
                    msg.rate = 1.0;
                    window.speechSynthesis.speak(msg);
                }}
            </script>
            """, height=0)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("BlindAid v2 • Optimized High-Speed Camera Processing & Mobile Speech Synthesis")
