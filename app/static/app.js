/**
 * BlindAid Mobile Web Application Client
 * Real-time Mobile Camera Processing + Web Speech API TTS Output
 */

// ── DOM Elements ─────────────────────────────────────────────────────────────
const webcamVideo     = document.getElementById('webcam');
const outputCanvas    = document.getElementById('outputCanvas');
const ctx             = outputCanvas.getContext('2d');

const modeBadge       = document.getElementById('modeBadge');
const muteBtn         = document.getElementById('muteBtn');
const dirPill         = document.getElementById('dirPill');
const fpsPill         = document.getElementById('fpsPill');
const speechBanner    = document.getElementById('speechBanner');
const speechText      = document.getElementById('speechText');

const startLocSelect  = document.getElementById('startLocSelect');
const destLocSelect   = document.getElementById('destLocSelect');
const startNavBtn     = document.getElementById('startNavBtn');
const cancelNavBtn    = document.getElementById('cancelNavBtn');

const navBanner       = document.getElementById('navBanner');
const navStepCount    = document.getElementById('navStepCount');
const navStepText     = document.getElementById('navStepText');
const nextStepBtn     = document.getElementById('nextStepBtn');

const connDot         = document.getElementById('connDot');
const connText        = document.getElementById('connText');

// ── Application State ────────────────────────────────────────────────────────
let isMuted           = false;
let isProcessingFrame = false;
let routeSteps        = [];
let currentStepIdx    = 0;
let isNavigating      = false;

let frameCount        = 0;
let fpsTimer          = Date.now();
let lastSpokenText    = "";
let lastSpokenTime    = 0;

// ── Web Speech API (TTS Output on Phone / Bluetooth) ──────────────────────────
const synth = window.speechSynthesis;

function speak(text, isCritical = false) {
  if (isMuted || !text || !text.trim()) return;

  const now = Date.now();
  // Avoid re-repeating the exact same non-critical message within 3s
  if (text === lastSpokenText && !isCritical && (now - lastSpokenTime) < 3000) {
    return;
  }

  // Update UI text banner
  speechText.textContent = text;
  if (isCritical) {
    speechBanner.classList.add('critical');
  } else {
    speechBanner.classList.remove('critical');
  }

  // Speak via browser TTS
  if (synth) {
    if (isCritical) {
      synth.cancel(); // Interrupt current speech for critical alerts!
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    synth.speak(utterance);

    lastSpokenText = text;
    lastSpokenTime = now;
  }
}

// ── Init Camera ──────────────────────────────────────────────────────────────
async function initCamera() {
  try {
    const constraints = {
      video: {
        facingMode: { ideal: 'environment' }, // Prefer rear smartphone camera
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamVideo.srcObject = stream;
    await webcamVideo.play();

    outputCanvas.width  = webcamVideo.videoWidth || 640;
    outputCanvas.height = webcamVideo.videoHeight || 480;

    console.log(`[Camera] Started: ${outputCanvas.width}x${outputCanvas.height}`);
    speak("Camera active. Point your phone forward.");
    
    // Start processing loop
    setInterval(captureAndProcessFrame, 150); // ~7 FPS frame processing
  } catch (err) {
    console.error("[Camera] Error:", err);
    speechText.textContent = "Camera access error. Please grant permissions.";
    alert("Camera permission denied. Please allow camera access in browser settings.");
  }
}

// ── Frame Processing Loop ────────────────────────────────────────────────────
async function captureAndProcessFrame() {
  if (isProcessingFrame || !webcamVideo.videoWidth) return;
  isProcessingFrame = true;

  // Render raw video to canvas first
  ctx.drawImage(webcamVideo, 0, 0, outputCanvas.width, outputCanvas.height);

  // Convert canvas frame to JPEG blob / base64
  const imageB64 = outputCanvas.toDataURL('image/jpeg', 0.65);

  try {
    const response = await fetch('/api/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageB64 })
    });

    if (!response.ok) throw new Error("Server error");
    const data = await response.json();

    // 1. Draw Path Corridors & Bounding Boxes
    drawOverlay(data);

    // 2. Update HUD Status
    dirPill.textContent = (data.recommended_direction || "FORWARD").toUpperCase();
    
    // Update FPS
    frameCount++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
      fpsPill.textContent = `${frameCount} FPS`;
      frameCount = 0;
      fpsTimer = now;
    }

    // 3. Handle Voice Instruction
    if (data.voice_msg) {
      speak(data.voice_msg, data.is_critical);
    }

    connDot.className = "dot online";
    connText.textContent = "Connected to BlindAid Server";

  } catch (err) {
    connDot.className = "dot";
    connText.textContent = "Server offline / Reconnecting...";
  } finally {
    isProcessingFrame = false;
  }
}

// ── Draw Overlay on Mobile Screen ─────────────────────────────────────────────
function drawOverlay(data) {
  const w = outputCanvas.width;
  const h = outputCanvas.height;

  // 1. Draw 3 Path Corridors (Bottom 65% of screen)
  const walkableTop = h * 0.35;
  const corridorW   = w / 3;

  const statusColors = {
    'clear':   'rgba(40, 200, 40, 0.28)',
    'caution': 'rgba(255, 165, 0, 0.32)',
    'blocked': 'rgba(220, 30, 30, 0.40)'
  };

  const borderColors = {
    'clear':   '#28c828',
    'caution': '#ffa500',
    'blocked': '#dc1e1e'
  };

  if (data.corridors) {
    data.corridors.forEach((c, idx) => {
      const x1 = idx * corridorW;
      
      ctx.fillStyle = statusColors[c.status] || statusColors['clear'];
      ctx.fillRect(x1, walkableTop, corridorW, h - walkableTop);

      ctx.strokeStyle = borderColors[c.status] || borderColors['clear'];
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, walkableTop, corridorW, h - walkableTop);

      // Label status
      ctx.fillStyle = borderColors[c.status];
      ctx.font = 'bold 14px Outfit, sans-serif';
      ctx.fillText(c.status.toUpperCase(), x1 + 10, walkableTop + 24);
    });
  }

  // 2. Draw Recommended Direction Arrow
  if (data.best_corridor && !data.all_blocked) {
    let arrowX = w / 2;
    if (data.best_corridor === 'LEFT')   arrowX = corridorW / 2;
    if (data.best_corridor === 'RIGHT')  arrowX = w - (corridorW / 2);

    ctx.strokeStyle = '#00ff87';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(arrowX, h - 30);
    ctx.lineTo(arrowX, walkableTop + 40);
    ctx.lineTo(arrowX - 12, walkableTop + 60);
    ctx.moveTo(arrowX, walkableTop + 40);
    ctx.lineTo(arrowX + 12, walkableTop + 60);
    ctx.stroke();
  }

  // 3. Draw Bounding Boxes
  if (data.detections) {
    data.detections.forEach(d => {
      const [bx1, by1, bx2, by2] = d.bbox;
      const bw = bx2 - bx1;
      const bh = by2 - by1;

      // Color by size
      const isLarge = d.area_fraction > 0.18;
      const color = isLarge ? '#ff3366' : '#00e5ff';

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(bx1, by1, bw, bh);

      // Label background & text
      ctx.fillStyle = color;
      const labelText = `${d.label} ${Math.round(d.confidence * 100)}%`;
      ctx.font = 'bold 13px Outfit, sans-serif';
      const textW = ctx.measureText(labelText).width;

      ctx.fillRect(bx1, by1 - 22, textW + 10, 22);
      ctx.fillStyle = '#000000';
      ctx.fillText(labelText, bx1 + 5, by1 - 6);
    });
  }
}

// ── Load Server Data & Setup UI ───────────────────────────────────────────────
async function loadServerInfo() {
  try {
    const res = await fetch('/api/info');
    const data = await res.json();

    if (data.map_loaded) {
      modeBadge.textContent = "MAP MODE";
      populateLocations(data.locations);
    } else {
      modeBadge.textContent = "CAMERA ONLY";
    }
  } catch (err) {
    console.error("[Init] Failed to load server info:", err);
  }
}

function populateLocations(locations) {
  startLocSelect.innerHTML = '<option value="">-- Start Location --</option>';
  destLocSelect.innerHTML  = '<option value="">-- Destination --</option>';

  locations.forEach(loc => {
    const opt1 = new Option(loc, loc);
    const opt2 = new Option(loc, loc);
    startLocSelect.add(opt1);
    destLocSelect.add(opt2);
  });
}

// ── Route Navigation Logic ────────────────────────────────────────────────────
async function startNavigation() {
  const startLoc = startLocSelect.value;
  const destLoc  = destLocSelect.value;

  if (!startLoc || !destLoc) {
    alert("Please select both a Start location and Destination.");
    return;
  }

  try {
    const res = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from_location: startLoc, to_location: destLoc })
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Route planning failed");
      return;
    }

    routeSteps = data.steps;
    currentStepIdx = 0;
    isNavigating = true;

    // Show UI Banner
    navBanner.style.display = 'flex';
    cancelNavBtn.style.display = 'inline-block';
    startNavBtn.style.display = 'none';

    updateNavStepUI();

    speak(`Route to ${data.to_name} calculated. ${routeSteps.length} steps. Starting navigation.`, true);

  } catch (err) {
    alert("Failed to start route navigation.");
  }
}

function updateNavStepUI() {
  if (currentStepIdx < routeSteps.length) {
    const step = routeSteps[currentStepIdx];
    navStepCount.textContent = `Step ${step.step_number} of ${routeSteps.length}`;
    navStepText.textContent  = step.instruction;

    speak(step.instruction);
  } else {
    // Arrival
    navStepCount.textContent = "ARRIVED";
    navStepText.textContent  = "You have arrived at your destination!";
    speak("You have arrived at your destination! Navigation complete.", true);
    cancelNavigation();
  }
}

function advanceNextStep() {
  if (!isNavigating) return;
  currentStepIdx++;
  updateNavStepUI();
}

function cancelNavigation() {
  isNavigating = false;
  routeSteps = [];
  currentStepIdx = 0;

  navBanner.style.display = 'none';
  cancelNavBtn.style.display = 'none';
  startNavBtn.style.display = 'inline-block';

  speak("Navigation cancelled.");
}

// ── Event Listeners ───────────────────────────────────────────────────────────
startNavBtn.addEventListener('click', startNavigation);
cancelNavBtn.addEventListener('click', cancelNavigation);
nextStepBtn.addEventListener('click', advanceNextStep);

muteBtn.addEventListener('click', () => {
  isMuted = !isMuted;
  muteBtn.textContent = isMuted ? '🔇' : '🔊';
  if (!isMuted) speak("Voice enabled");
});

// ── Initialization ───────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadServerInfo();
  initCamera();
});
