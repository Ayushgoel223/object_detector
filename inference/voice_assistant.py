"""
BlindAid - Voice Assistant Module
===================================
Uses win32com (Windows SAPI) directly for maximum reliability.
Fallback: pyttsx3 -> gTTS

win32com.client.Dispatch('SAPI.SpVoice') is the most reliable
TTS on Windows - no threading COM issues when used correctly.
"""

import threading
import queue
import time
import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


# ── Priority Levels ────────────────────────────────────────────────────────────

class MsgPriority(Enum):
    CRITICAL = 0
    NORMAL   = 1


@dataclass
class VoiceMessage:
    text: str
    priority: MsgPriority
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def __lt__(self, other):
        return self.priority.value < other.priority.value


# ── Voice Assistant ────────────────────────────────────────────────────────────

class VoiceAssistant:
    """
    Non-blocking, rate-limited TTS using Windows SAPI directly via win32com.
    Falls back to pyttsx3, then gTTS if win32com unavailable.
    """

    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        vc = cfg["voice"]
        self.rate              = vc.get("rate", 165)
        self.volume            = vc.get("volume", 1.0)
        self.min_interval      = vc["min_interval_seconds"]
        self.critical_interval = vc["critical_interval_seconds"]
        self.max_queue         = vc["max_queue_size"]

        # Convert pyttsx3 rate (wpm) to SAPI rate (-10 to 10)
        # 165 wpm ≈ SAPI rate 2
        self.sapi_rate = max(-5, min(5, int((self.rate - 150) / 15)))

        self._queue            = queue.PriorityQueue(maxsize=self.max_queue + 5)
        self._last_spoken_time = 0.0
        self._last_message     = ""
        self._stop_event       = threading.Event()
        self._thread           = None

        print(f"[Voice] Initialized. Rate={self.rate} wpm, SAPI rate={self.sapi_rate}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="VoiceWorker"
        )
        self._thread.start()
        print("[Voice] Background TTS thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)
        print("[Voice] TTS thread stopped.")

    def speak(self, text: str, critical: bool = False):
        if not text or not text.strip():
            return
        priority = MsgPriority.CRITICAL if critical else MsgPriority.NORMAL
        if text == self._last_message and not critical:
            return
        msg = VoiceMessage(text=text, priority=priority)
        try:
            self._queue.put_nowait((priority.value, msg))
        except queue.Full:
            if critical:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait((priority.value, msg))
                except Exception:
                    pass

    def speak_now(self, text: str):
        """Synchronous — blocks until done. Used for startup/shutdown."""
        print(f"[Voice] Speaking: \"{text}\"")
        self._sapi_speak(text)

    def is_busy(self) -> bool:
        return not self._queue.empty()

    # ── Worker Thread ──────────────────────────────────────────────────────────

    def _worker(self):
        """
        Worker thread: creates its OWN SAPI speaker instance.
        Each thread must create its own COM object.
        """
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
        except Exception:
            com_initialized = False

        # Create SAPI speaker on THIS thread
        speaker = self._create_sapi_speaker()

        while not self._stop_event.is_set():
            try:
                _, msg = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now      = time.time()
            interval = (self.critical_interval
                        if msg.priority == MsgPriority.CRITICAL
                        else self.min_interval)
            elapsed  = now - self._last_spoken_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

            print(f"[Voice] Speaking: \"{msg.text}\"")
            self._speak_with(speaker, msg.text)
            self._last_spoken_time = time.time()
            self._last_message     = msg.text
            self._queue.task_done()

        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ── TTS Backends ───────────────────────────────────────────────────────────

    def _create_sapi_speaker(self):
        """Create a SAPI SpVoice COM object. Returns None on failure."""
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Volume = int(self.volume * 100)
            speaker.Rate   = self.sapi_rate
            print("[Voice] SAPI SpVoice created on thread.")
            return speaker
        except Exception as e:
            print(f"[Voice] SAPI creation failed: {e}. Will use pyttsx3.")
            return None

    def _speak_with(self, speaker, text: str):
        """Speak using SAPI speaker, fallback to pyttsx3, then gTTS."""
        # Try SAPI first
        if speaker is not None:
            try:
                speaker.Speak(text)
                return
            except Exception as e:
                print(f"[Voice] SAPI speak error: {e}")

        # Fallback: pyttsx3
        try:
            import pyttsx3
            import pythoncom
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return
        except Exception as e:
            print(f"[Voice] pyttsx3 fallback error: {e}")

        # Fallback: gTTS (online)
        self._gtts_speak(text)

    def _sapi_speak(self, text: str):
        """Synchronous SAPI speak for main thread (speak_now)."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Volume = int(self.volume * 100)
            speaker.Rate   = self.sapi_rate
            speaker.Speak(text)
            return
        except Exception as e:
            print(f"[Voice] speak_now SAPI error: {e}")

        # Fallback: pyttsx3 on main thread
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[Voice] speak_now pyttsx3 error: {e}")
            print(f"[AUDIO FALLBACK] {text}")

    def _gtts_speak(self, text: str):
        """Online TTS fallback."""
        try:
            from gtts import gTTS
            tmp = Path("_blindaid_tts.mp3")
            gTTS(text=text, lang="en", slow=False).save(str(tmp))
            os.system(f'start /min "" "{tmp.resolve()}"')
            time.sleep(max(1.5, len(text) * 0.07))
            try:
                tmp.unlink()
            except Exception:
                pass
        except Exception as e:
            print(f"[Voice] gTTS error: {e}")
            print(f"[AUDIO FALLBACK] {text}")
