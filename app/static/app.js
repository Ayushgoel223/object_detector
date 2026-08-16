/**
 * BlindAid Mobile Web Application Client
 * Real-time Camera Processing + Robust Mobile Audio Synthesis (Web Audio Chime + Web Speech API)
 */

// ── DOM Elements ─────────────────────────────────────────────────────────────
const webcamVideo     = document.getElementById('webcam');
const outputCanvas    = document.getElementById('outputCanvas');
const ctx             = outputCanvas.getContext('2d');

const modeBadge       = document.getElementById('modeBadge');
const muteBtn         = document.getElementById('muteBtn');
const unlockBanner    = document.getElementById('unlockBanner');
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
let isVoiceUnlocked   = false;

let frameCount        = 0;
let fpsTimer          = Date.now();
let lastSpokenText    = "";
let lastSpokenTime    = 0;

// ── Web Audio Chime Generator ────────────────────────────────────────────────
let audioCtx = null;

function playAudioChime(isCritical = false) {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = isCritical ? 'sawtooth' : 'sine';
    osc.frequency.setValueAtTime(isCritical ? 880 : 587.33, now); // A5 or D5 tone
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + (isCritical ? 0.3 : 0.18));

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start(now);
    osc.stop(now + (isCritical ? 0.3 : 0.18));
  } catch (e) {
    console.log("[Audio] WebAudio chime notice:", e);
  }
}

// ── Web Speech API Voice Engine ──────────────────────────────────────────────
const synth = window.speechSynthesis;
let availableVoices = [];

function loadVoices() {
  if (synth) {
    availableVoices = synth.getVoices();
  }
}

if (synth) {
  synth.onvoiceschanged = loadVoices;
  loadVoices();
}

const voicePlayer = document.getElementById('voicePlayer');

function playMp3Audio(b64Audio) {
  if (!voicePlayer || !b64Audio) return false;
  try {
    voicePlayer.src = b64Audio;
    const playPromise = voicePlayer.play();
    if (playPromise !== undefined) {
      playPromise.catch(e => {
        console.log("[Audio] HTML5 audio play blocked/waiting for touch:", e);
      });
    }
    return true;
  } catch (e) {
    console.log("[Audio] MP3 playback error:", e);
    return false;
  }
}

function unlockVoice() {
  if (isVoiceUnlocked) return;
  isVoiceUnlocked = true;

  if (unlockBanner) {
    unlockBanner.style.background = "#00e5ff";
    unlockBanner.style.color = "#000";
    unlockBanner.textContent = "🔊 LIVE PHONE VOICE IS ACTIVE";
  }

  // Play silent buffer on voicePlayer to unlock mobile HTML5 audio permissions
  if (voicePlayer) {
    voicePlayer.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAA==";
    voicePlayer.play().catch(e => {});
  }

  playAudioChime(false);
  speak("Voice audio active. Point phone camera forward.", true);
}

// Global touch/click listener in CAPTURE phase to guarantee unlocking mobile audio
window.addEventListener('click', unlockVoice, true);
window.addEventListener('touchstart', unlockVoice, true);

function speak(text, isCritical = false, audioB64 = null) {
  if (isMuted || !text || !text.trim()) return;

  const now = Date.now();

  // Deduplicate identical non-critical messages within 2.5 seconds
  if (text === lastSpokenText && !isCritical && (now - lastSpokenTime) < 2500) {
    return;
  }

  // 1. Update Text UI Banner
  if (speechText) speechText.textContent = text;
  if (speechBanner) {
    if (isCritical) {
      speechBanner.classList.add('critical');
    } else {
      speechBanner.classList.remove('critical');
    }
  }

  // 2. Play Audio Tone Chime
  playAudioChime(isCritical);

  // 3. Prefer Real MP3 Audio Stream (100% working on all mobile OS over HTTP/HTTPS)
  let mp3Played = false;
  if (audioB64) {
    mp3Played = playMp3Audio(audioB64);
  }

  // 4. Fallback to Web Speech Synthesis API if no MP3 available
  if (!mp3Played && synth) {
    try {
      synth.resume(); // Wake up mobile Chrome/Safari audio engine

      if (isCritical) {
        synth.cancel(); // Interrupt only for critical emergencies!
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 1.0;
      utterance.volume = 1.0;

      if (availableVoices.length === 0) loadVoices();
      const preferredVoice = availableVoices.find(
        v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha') || v.default)
      ) || availableVoices.find(v => v.lang.startsWith('en'));

      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      synth.speak(utterance);
    } catch (err) {
      console.error("[TTS] Speech Synthesis error:", err);
    }
  }

  lastSpokenText = text;
  lastSpokenTime = now;
}

// ── Init Rear Camera ──────────────────────────────────────────────────────────
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
    
    // Start continuous processing loop (~7 FPS)
    setInterval(captureAndProcessFrame, 150);
  } catch (err) {
    console.error("[Camera] Error:", err);
    if (speechText) speechText.textContent = "Camera error. Grant camera permissions.";
    alert("Camera permission denied. Please allow camera access in browser settings.");
  }
}

// ── Frame Processing Loop ────────────────────────────────────────────────────
async function captureAndProcessFrame() {
  if (isProcessingFrame || !webcamVideo.videoWidth) return;
  isProcessingFrame = true;

  // Render raw video to canvas
  ctx.drawImage(webcamVideo, 0, 0, outputCanvas.width, outputCanvas.height);

  // Convert canvas frame to base64 JPEG
  const imageB64 = outputCanvas.toDataURL('image/jpeg', 0.60);

  try {
    const response = await fetch('/api/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageB64 })
    });

    if (!response.ok) throw new Error("Server error");
    const data = await response.json();

    // 1. Draw Path Corridors & Bounding Boxes on Canvas
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

    // 3. Live Voice Output (Obstacles + Directions)
    if (data.voice_msg) {
      speak(data.voice_msg, data.is_critical, data.audio_b64);
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

    speak(step.instruction, true);
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

muteBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  isMuted = !isMuted;
  muteBtn.textContent = isMuted ? '🔇' : '🔊';
  if (isMuted) {
    muteBtn.classList.add('muted');
  } else {
    muteBtn.classList.remove('muted');
    speak("Voice active and unmuted.", true);
  }
});

// ── Initialization ───────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadServerInfo();
  initCamera();
});
