# BlindAid - README
# ==================
# AI Navigation Assistant for the Visually Impaired

# ─────────────────────────────────────────────────────────
# HOW TO RUN (Windows PC - No GPU needed)
# ─────────────────────────────────────────────────────────

# STEP 1: Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate

# STEP 2: Install dependencies
pip install -r requirements.txt

# STEP 3: Run system tests (no camera needed)
python test_system.py

# STEP 4: Launch the real-time assistant
python app/main.py

# ─────────────────────────────────────────────────────────
# KEYBOARD CONTROLS (while app is running)
# ─────────────────────────────────────────────────────────
# Q or ESC  : Quit
# P         : Pause / Resume
# M         : Mute / Unmute voice
# +         : Increase confidence (fewer detections, more accurate)
# -         : Decrease confidence (more detections, may have false positives)

# ─────────────────────────────────────────────────────────
# EVALUATE THE MODEL
# ─────────────────────────────────────────────────────────
python models/evaluate.py

# ─────────────────────────────────────────────────────────
# EXPORT FOR MOBILE (future Android version)
# ─────────────────────────────────────────────────────────
python models/export.py --format onnx
python models/export.py --format tflite
