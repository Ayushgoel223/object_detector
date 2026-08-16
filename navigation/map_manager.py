"""
BlindAid - Map Manager
========================
Parses a floor plan image using Google Gemini Vision API.
Extracts rooms, corridors, landmarks, and connections.
Builds a navigable graph from the floor plan.
Caches the result so it doesn't re-parse every run.

Usage:
    manager = MapManager()
    manager.load_map("data/floorplan.jpg")
    locations = manager.get_location_names()
    print(locations)  # ['entrance', 'cafeteria', 'office', ...]
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
import networkx as nx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


class MapManager:
    """
    Manages the floor plan map.
    Uses Gemini Vision to parse the image into a navigable graph.
    """

    GEMINI_PARSE_PROMPT = """
You are analyzing a floor plan image for a blind navigation assistant.

Extract all rooms/spaces and how they connect to each other.

Return ONLY a valid JSON object with this exact structure:
{
  "building_name": "name of the building if shown",
  "locations": [
    {
      "id": "snake_case_unique_id",
      "name": "Human Readable Room Name",
      "type": "room|corridor|stairs|exit|bathroom|entrance",
      "aliases": ["alternative names the user might say"],
      "description": "one sentence description for blind user"
    }
  ],
  "connections": [
    {
      "from": "location_id_1",
      "to": "location_id_2",
      "distance_steps": 10,
      "turn_direction": "straight|left|right|slight_left|slight_right",
      "landmark_hint": "optional: what the user will notice at this point (door, turn, wall end)"
    }
  ]
}

Rules:
- distance_steps: realistic walking steps between rooms (1 step ≈ 75cm)
- All connections are bidirectional
- Include stairs, exits, balconies, bathrooms as locations
- aliases: include common names someone might say ("loo" for bathroom, "main door" for entrance)
- Be practical for a blind person using audio guidance
"""

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = ROOT / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.graph: Optional[nx.Graph] = None
        self.locations: dict = {}      # id -> location dict
        self.building_name: str = "Building"
        self._map_loaded: bool = False
        self._api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ── Public API ──────────────────────────────────────────────────────────────

    def load_map(self, image_path: str, force_reparse: bool = False) -> bool:
        """
        Load and parse a floor plan image.
        Uses cached JSON if available (avoids repeated API calls).

        Returns True on success, False on failure.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[Map] ERROR: Floor plan not found: {image_path}")
            return False

        cache_file = self.cache_dir / f"{image_path.stem}_map.json"

        if cache_file.exists() and not force_reparse:
            print(f"[Map] Loading cached map: {cache_file}")
            return self._load_from_cache(cache_file)

        print(f"[Map] Parsing floor plan with Gemini Vision: {image_path.name}")
        map_data = self._parse_with_gemini(image_path)

        if map_data is None:
            print("[Map] Gemini parse failed. Trying fallback text description.")
            return False

        # Cache the result
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, indent=2)
        print(f"[Map] Map cached to: {cache_file}")

        return self._build_graph(map_data)

    def load_map_from_text(self, description: str) -> bool:
        """
        Parse a text description of the building instead of an image.
        Useful when no floor plan image is available.
        """
        print("[Map] Parsing text description with Gemini...")
        map_data = self._parse_text_with_gemini(description)
        if map_data:
            return self._build_graph(map_data)
        return False

    def load_map_from_json(self, json_path: str) -> bool:
        """Load a pre-existing map JSON directly."""
        return self._load_from_cache(Path(json_path))

    def get_location_names(self) -> list:
        """Return all location names the user can navigate to."""
        return [loc["name"] for loc in self.locations.values()]

    def find_location(self, user_input: str) -> Optional[str]:
        """
        Find the best matching location ID from a user's typed input.
        Handles partial matches and aliases.

        Returns location ID or None.
        """
        query = user_input.strip().lower()
        best_match = None
        best_score = 0

        for loc_id, loc in self.locations.items():
            # Check name
            name_lower = loc["name"].lower()
            score = self._match_score(query, name_lower)

            # Check aliases
            for alias in loc.get("aliases", []):
                score = max(score, self._match_score(query, alias.lower()))

            if score > best_score:
                best_score = score
                best_match = loc_id

        if best_score >= 0.4:
            return best_match
        return None

    def get_location_description(self, loc_id: str) -> str:
        """Get the description of a location for voice output."""
        loc = self.locations.get(loc_id, {})
        return loc.get("description", loc.get("name", loc_id))

    def is_loaded(self) -> bool:
        return self._map_loaded

    # ── Graph Building ─────────────────────────────────────────────────────────

    def _build_graph(self, map_data: dict) -> bool:
        try:
            self.building_name = map_data.get("building_name", "Building")
            self.graph = nx.Graph()
            self.locations = {}

            for loc in map_data.get("locations", []):
                loc_id = loc["id"]
                self.locations[loc_id] = loc
                self.graph.add_node(
                    loc_id,
                    name=loc["name"],
                    type=loc.get("type", "room"),
                    description=loc.get("description", ""),
                )

            for conn in map_data.get("connections", []):
                from_id = conn["from"]
                to_id   = conn["to"]
                if from_id in self.locations and to_id in self.locations:
                    self.graph.add_edge(
                        from_id, to_id,
                        distance=conn.get("distance_steps", 10),
                        turn=conn.get("turn_direction", "straight"),
                        hint=conn.get("landmark_hint", ""),
                    )

            self._map_loaded = True
            n_loc = len(self.locations)
            n_conn = self.graph.number_of_edges()
            print(f"[Map] Graph built: {n_loc} locations, {n_conn} connections.")
            print(f"[Map] Locations: {', '.join(self.get_location_names())}")
            return True

        except Exception as e:
            print(f"[Map] Graph build error: {e}")
            return False

    def _load_from_cache(self, cache_file: Path) -> bool:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                map_data = json.load(f)
            return self._build_graph(map_data)
        except Exception as e:
            print(f"[Map] Cache load error: {e}")
            return False

    # ── Gemini API ─────────────────────────────────────────────────────────────

    def _parse_with_gemini(self, image_path: Path) -> Optional[dict]:
        if not self._api_key or self._api_key == "your_gemini_api_key_here":
            print("[Map] No Gemini API key set. Add it to .env file.")
            return None
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()

            raw = None
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self._api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        self.GEMINI_PARSE_PROMPT,
                    ]
                )
                raw = response.text.strip()
            except Exception as e1:
                print(f"[Map] google.genai error: {e1}, trying google.generativeai...")
                import google.generativeai as genai_old
                from PIL import Image
                genai_old.configure(api_key=self._api_key)
                model = genai_old.GenerativeModel("gemini-2.5-flash")
                img = Image.open(image_path)
                response = model.generate_content([self.GEMINI_PARSE_PROMPT, img])
                raw = response.text.strip()

            if not raw:
                return None

            # Strip markdown code fences if present
            if "```" in raw:
                parts = raw.split("```")
                for p in parts:
                    p_str = p.strip()
                    if p_str.startswith("json"):
                        p_str = p_str[4:].strip()
                    if p_str.startswith("{") and p_str.endswith("}"):
                        raw = p_str
                        break

            map_data = json.loads(raw)
            print(f"[Map] Gemini parsed {len(map_data.get('locations', []))} locations.")
            return map_data

        except json.JSONDecodeError as e:
            print(f"[Map] Gemini returned invalid JSON: {e}")
            print(f"[Map] Raw response: {raw[:300] if raw else 'None'}")
            return None
        except Exception as e:
            print(f"[Map] Gemini API error: {e}")
            return None

    def _parse_text_with_gemini(self, description: str) -> Optional[dict]:
        if not self._api_key or self._api_key == "your_gemini_api_key_here":
            return None
        try:
            prompt = self.GEMINI_PARSE_PROMPT + f"\n\nText description of the building:\n{description}"
            raw = None
            try:
                from google import genai
                client = genai.Client(api_key=self._api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                raw = response.text.strip()
            except Exception as e1:
                import google.generativeai as genai_old
                genai_old.configure(api_key=self._api_key)
                model = genai_old.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                raw = response.text.strip()

            if not raw:
                return None

            if "```" in raw:
                parts = raw.split("```")
                for p in parts:
                    p_str = p.strip()
                    if p_str.startswith("json"):
                        p_str = p_str[4:].strip()
                    if p_str.startswith("{") and p_str.endswith("}"):
                        raw = p_str
                        break

            return json.loads(raw.strip())
        except Exception as e:
            print(f"[Map] Text parse error: {e}")
            return None

    # ── Fuzzy Matching ─────────────────────────────────────────────────────────

    def _match_score(self, query: str, target: str) -> float:
        """Simple string similarity: exact > startswith > contains."""
        if query == target:
            return 1.0
        if target.startswith(query) or query.startswith(target):
            return 0.8
        if query in target or target in query:
            return 0.6
        # Word-level overlap
        q_words = set(query.split())
        t_words = set(target.split())
        overlap = q_words & t_words
        if overlap:
            return 0.5 * len(overlap) / max(len(q_words), len(t_words))
        return 0.0
