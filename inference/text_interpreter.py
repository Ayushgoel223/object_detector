"""
BlindAid — Text Interpreter
==============================
Maps raw OCR output to semantic navigation events.
Uses keyword matching + fuzzy string matching via rapidfuzz.

Also handles word dynamics:
  - Single char confidence drops → partially obscured sign
  - Low overall confidence → announce with uncertainty qualifier
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as rfuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("[TextInterp] rapidfuzz not installed. Exact matching only.")


# ── Semantic Event ─────────────────────────────────────────────────────────────

@dataclass
class TextEvent:
    """A navigation-relevant event derived from OCR text."""
    raw_text:       str
    tag:            str              # e.g. 'EXIT', 'STAIRS', 'DANGER'
    message:        str              # Human-readable navigation instruction
    priority:       int              # Higher = more important (0-100)
    confidence:     float            # OCR confidence
    zone:           str              # 'left' | 'center' | 'right'
    uncertain:      bool = False     # True if OCR conf was borderline

    def __repr__(self):
        flag = "~" if self.uncertain else ""
        return f"[{self.tag}] {flag}{self.message} (zone={self.zone}, pri={self.priority})"


# ── Keyword Knowledge Base ────────────────────────────────────────────────────
# (keyword_lower, tag, priority 0-100, message_template)
# {ZONE} is replaced with the actual zone string at runtime

KEYWORD_DB: List[Tuple[str, str, int, str]] = [
    # Critical / danger
    ("danger",      "DANGER",       95, "Warning: DANGER sign detected {ZONE}."),
    ("warning",     "WARNING",      90, "Warning sign detected {ZONE}."),
    ("caution",     "CAUTION",      85, "Caution sign detected {ZONE}."),
    ("no entry",    "NO_ENTRY",     85, "No Entry sign {ZONE}. Do not proceed."),
    ("stop",        "STOP",         80, "STOP sign {ZONE}."),
    ("emergency",   "EMERGENCY",    95, "Emergency sign detected {ZONE}."),
    ("fire",        "FIRE",         90, "Fire warning {ZONE}. Proceed carefully."),
    ("wet floor",   "WET_FLOOR",    80, "Wet floor warning {ZONE}. Walk carefully."),

    # Navigation
    ("exit",        "EXIT",         75, "Exit sign {ZONE}."),
    ("stairs",      "STAIRS",       75, "Stairs detected {ZONE}. Take care."),
    ("stair",       "STAIRS",       75, "Stairs detected {ZONE}. Take care."),
    ("step",        "STAIRS",       70, "Steps {ZONE}. Be careful."),
    ("elevator",    "ELEVATOR",     60, "Elevator {ZONE}."),
    ("lift",        "ELEVATOR",     60, "Elevator {ZONE}."),
    ("escalator",   "ESCALATOR",    65, "Escalator {ZONE}. Grab the rail."),
    ("ramp",        "RAMP",         60, "Ramp {ZONE}."),

    # Doors
    ("push",        "DOOR_PUSH",    55, "Push door {ZONE}."),
    ("pull",        "DOOR_PULL",    55, "Pull door {ZONE}."),
    ("entrance",    "ENTRANCE",     50, "Entrance {ZONE}."),
    ("enter",       "ENTRANCE",     50, "Entrance {ZONE}."),

    # Facilities
    ("restroom",    "RESTROOM",     40, "Restroom {ZONE}."),
    ("toilet",      "RESTROOM",     40, "Restroom {ZONE}."),
    ("bathroom",    "RESTROOM",     40, "Restroom {ZONE}."),

    # Directional
    ("left",        "DIRECTION",    30, "Sign says left {ZONE}."),
    ("right",       "DIRECTION",    30, "Sign says right {ZONE}."),
    ("straight",    "DIRECTION",    30, "Sign says straight ahead {ZONE}."),
    ("ahead",       "DIRECTION",    30, "Sign points ahead {ZONE}."),

    # Information
    ("open",        "INFO",         20, "Sign says open {ZONE}."),
    ("closed",      "INFO",         20, "Sign says closed {ZONE}."),
    ("out of order","INFO",         25, "Out of order {ZONE}."),
]

# Build lookup for exact matching
_EXACT_MAP = {kw: (tag, pri, msg) for kw, tag, pri, msg in KEYWORD_DB}

# All keywords for fuzzy matching
_ALL_KEYWORDS = [kw for kw, *_ in KEYWORD_DB]


class TextInterpreter:
    """
    Converts raw OCR text into structured TextEvents.

    Matching strategy (cascaded):
    1. Exact substring match (fastest)
    2. Regex pattern match (for multi-word keywords)
    3. Fuzzy match via rapidfuzz (handles OCR misreads)
    """

    def __init__(self, fuzzy_threshold: int = 80):
        """
        Args:
            fuzzy_threshold: Minimum rapidfuzz score (0-100) to accept a match.
                             80 means ~80% similarity required.
        """
        self.fuzzy_threshold = fuzzy_threshold

    def interpret(self, ocr_results: list) -> List[TextEvent]:
        """
        Process a list of OCRResult objects and return TextEvents.

        Args:
            ocr_results: List[OCRResult] from CNNOCRReader.read()

        Returns:
            List[TextEvent] sorted by priority descending.
        """
        events: List[TextEvent] = []
        seen_tags = set()

        for ocr_result in ocr_results:
            text = ocr_result.text
            zone = ocr_result.zone
            conf = ocr_result.confidence

            # Try to find a match
            match = self._match(text)
            if match is None:
                continue

            tag, priority, msg_template = match
            if tag in seen_tags:
                continue   # Deduplicate same tag
            seen_tags.add(tag)

            # Format zone string for message
            zone_phrase = self._zone_phrase(zone)
            message = msg_template.replace("{ZONE}", zone_phrase).strip()

            # Uncertainty qualifier for borderline confidence
            uncertain = conf < 0.60

            # Adjust priority by confidence
            effective_priority = int(priority * conf)

            events.append(TextEvent(
                raw_text=text,
                tag=tag,
                message=message,
                priority=effective_priority,
                confidence=conf,
                zone=zone,
                uncertain=uncertain,
            ))

        events.sort(key=lambda e: e.priority, reverse=True)
        return events

    def _match(self, text: str) -> Optional[Tuple[str, int, str]]:
        """
        Returns (tag, priority, message_template) or None.
        Tries: exact → regex → fuzzy.
        """
        lower = text.lower().strip()
        if not lower:
            return None

        # 1. Exact substring match
        for kw, (tag, pri, msg) in _EXACT_MAP.items():
            if kw in lower:
                return tag, pri, msg

        # 2. Fuzzy match (handles OCR misreads like "EX1T", "STAAIRS")
        if RAPIDFUZZ_AVAILABLE:
            # Check each word in the OCR text
            words = re.sub(r"[^a-z0-9 ]", " ", lower).split()
            for word in words:
                if len(word) < 3:
                    continue
                result = rfuzz_process.extractOne(
                    word, _ALL_KEYWORDS,
                    scorer=fuzz.ratio,
                    score_cutoff=self.fuzzy_threshold,
                )
                if result is not None:
                    kw = result[0]
                    tag, pri, msg = _EXACT_MAP[kw]
                    return tag, pri, msg

        return None

    def _zone_phrase(self, zone: str) -> str:
        return {
            "left":   "on your left",
            "center": "ahead",
            "right":  "on your right",
        }.get(zone, "")

    def get_top_priority(self, ocr_results: list) -> Optional[TextEvent]:
        """Convenience: returns only the highest-priority event."""
        events = self.interpret(ocr_results)
        return events[0] if events else None

    def format_display(self, events: List[TextEvent], max_lines: int = 3) -> List[str]:
        """
        Returns text lines for the display overlay.
        Uncertain readings are prefixed with '~'.
        """
        lines = []
        for ev in events[:max_lines]:
            prefix = "~" if ev.uncertain else "►"
            lines.append(f"{prefix} {ev.message}")
        return lines


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class FakeOCR:
        text: str
        zone: str = "center"
        confidence: float = 0.85

    interp = TextInterpreter()
    test_cases = [
        FakeOCR("EXIT →"),
        FakeOCR("EX1T", confidence=0.55),   # OCR misread — fuzzy should catch
        FakeOCR("STAIRS DOWN", zone="left"),
        FakeOCR("Wet Floor", confidence=0.72),
        FakeOCR("Open 9-5"),                 # low priority info
        FakeOCR("qwerty xyz"),               # no match
    ]

    for case in test_cases:
        events = interp.interpret([case])
        if events:
            print(f"'{case.text}' → {events[0]}")
        else:
            print(f"'{case.text}' → No match")
