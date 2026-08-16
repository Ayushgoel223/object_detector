"""
BlindAid - Navigation Command Parser
======================================
Parses typed text commands from the user into navigation intents.

Supports:
  "go to cafeteria"          → NAVIGATE_TO: cafeteria
  "take me to the office"    → NAVIGATE_TO: office
  "i am at the entrance"     → SET_LOCATION: entrance
  "where am i"               → WHERE_AM_I
  "next step"                → NEXT_STEP
  "repeat"                   → REPEAT
  "cancel" / "stop"          → CANCEL
  "list places"              → LIST_PLACES
  "help"                     → HELP
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class CommandType(Enum):
    NAVIGATE_TO   = "navigate_to"
    SET_LOCATION  = "set_location"
    WHERE_AM_I    = "where_am_i"
    NEXT_STEP     = "next_step"
    REPEAT        = "repeat"
    CANCEL        = "cancel"
    LIST_PLACES   = "list_places"
    HELP          = "help"
    UNKNOWN       = "unknown"


@dataclass
class ParsedCommand:
    type: CommandType
    argument: Optional[str] = None   # location name if relevant
    raw_input: str = ""


class CommandParser:
    """
    Parses typed navigation commands into structured intents.
    Uses simple keyword + regex matching (no LLM needed for commands).
    """

    NAVIGATE_PATTERNS = [
        r"(?:go|take me|navigate|walk|head|i want to go)\s+to\s+(?:the\s+)?(.+)",
        r"(?:find|show me|where is)\s+(?:the\s+)?(.+)",
        r"(?:to|towards?)\s+(?:the\s+)?(.+)",
    ]

    SET_LOCATION_PATTERNS = [
        r"i(?:'m| am) (?:at|in|near|by|standing at)\s+(?:the\s+)?(.+)",
        r"(?:i'm at|at|in|currently at)\s+(?:the\s+)?(.+)",
        r"(?:my location is|i am currently at|set (?:my )?(?:current )?(?:location|position) (?:to|as))\s+(?:the\s+)?(.+)",
        r"(?:location|position|i'm)\s+(.+)",
    ]

    WHERE_AM_I_KEYWORDS = [
        "where am i", "where are we", "my location", "current location",
        "what location", "where", "am i", "where i am"
    ]

    NEXT_STEP_KEYWORDS = [
        "next", "next step", "continue", "done", "ok", "okay",
        "proceed", "go", "forward", "advance", "got it", "understood"
    ]

    REPEAT_KEYWORDS = [
        "repeat", "again", "say again", "what did you say",
        "say that again", "pardon", "what"
    ]

    CANCEL_KEYWORDS = [
        "cancel", "stop navigation", "abort", "quit navigation",
        "stop", "end", "exit navigation", "never mind"
    ]

    LIST_KEYWORDS = [
        "list", "places", "locations", "what places", "where can i go",
        "show places", "available", "list places", "all locations"
    ]

    HELP_KEYWORDS = [
        "help", "commands", "what can you do", "instructions",
        "how to", "what do i say", "options"
    ]

    def parse(self, text: str) -> ParsedCommand:
        """Parse a text command into a ParsedCommand."""
        text = text.strip()
        lower = text.lower().strip('.,!?')

        # WHERE AM I
        for kw in self.WHERE_AM_I_KEYWORDS:
            if kw in lower and lower.count(" ") <= 3:
                return ParsedCommand(CommandType.WHERE_AM_I, raw_input=text)

        # LIST PLACES
        for kw in self.LIST_KEYWORDS:
            if lower == kw or lower.startswith(kw):
                return ParsedCommand(CommandType.LIST_PLACES, raw_input=text)

        # HELP
        for kw in self.HELP_KEYWORDS:
            if lower == kw or lower.startswith(kw):
                return ParsedCommand(CommandType.HELP, raw_input=text)

        # CANCEL
        for kw in self.CANCEL_KEYWORDS:
            if lower == kw or lower.startswith(kw):
                return ParsedCommand(CommandType.CANCEL, raw_input=text)

        # NEXT STEP
        for kw in self.NEXT_STEP_KEYWORDS:
            if lower == kw:
                return ParsedCommand(CommandType.NEXT_STEP, raw_input=text)

        # REPEAT
        for kw in self.REPEAT_KEYWORDS:
            if lower == kw or lower.startswith(kw):
                return ParsedCommand(CommandType.REPEAT, raw_input=text)

        # NAVIGATE TO
        for pattern in self.NAVIGATE_PATTERNS:
            match = re.match(pattern, lower)
            if match:
                destination = match.group(1).strip().rstrip('.,!?')
                return ParsedCommand(CommandType.NAVIGATE_TO, argument=destination, raw_input=text)

        # SET LOCATION
        for pattern in self.SET_LOCATION_PATTERNS:
            match = re.match(pattern, lower)
            if match:
                location = match.group(1).strip().rstrip('.,!?')
                return ParsedCommand(CommandType.SET_LOCATION, argument=location, raw_input=text)

        # If short text, might be a location name (e.g. just "cafeteria" → navigate there)
        if len(lower.split()) <= 3 and not any(
            c in lower for c in ['?', 'where', 'how', 'what']
        ):
            return ParsedCommand(CommandType.NAVIGATE_TO, argument=lower, raw_input=text)

        return ParsedCommand(CommandType.UNKNOWN, raw_input=text)

    def get_help_text(self) -> str:
        return """
Navigation Commands:
  go to [place]        — Navigate to a location
  i am at [place]      — Set your current position
  next step            — Advance to next navigation step
  repeat               — Repeat the current instruction
  list places          — Show all available locations
  where am i           — Tell current location
  cancel               — Cancel current navigation
  help                 — Show this help
""".strip()
