"""
BlindAid - Route Planner
==========================
Takes a floor plan graph (from MapManager) and computes
step-by-step walking directions between two locations.

Uses Dijkstra shortest path weighted by distance in steps.

Usage:
    planner = RoutePlanner(map_manager)
    steps = planner.get_route("entrance", "cafeteria")
    for step in steps:
        print(step.instruction)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import networkx as nx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class NavigationStep:
    """A single step in a navigation route."""
    step_number: int
    instruction: str           # Voice instruction: "Turn left at the door"
    from_location: str         # Location ID
    to_location: str           # Location ID
    from_name: str             # Human-readable name
    to_name: str               # Human-readable name
    distance_steps: int        # Walking steps for this segment
    turn_direction: str        # straight / left / right
    landmark_hint: str         # "you'll feel the wall end" etc.
    is_final: bool             # True for the last step (arrival)


class RoutePlanner:
    """
    Computes and manages navigation routes through a building.

    State machine:
        IDLE → NAVIGATING (step 1) → ... → NAVIGATING (step N) → ARRIVED
    """

    def __init__(self, map_manager):
        self.map_manager = map_manager
        self.graph = map_manager.graph
        self.locations = map_manager.locations

        # Navigation state
        self.current_route: List[NavigationStep] = []
        self.current_step_index: int = 0
        self.origin_id: Optional[str] = None
        self.destination_id: Optional[str] = None
        self.is_navigating: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan_route(self, from_id: str, to_id: str) -> Optional[List[NavigationStep]]:
        """
        Plan a route from from_id to to_id.
        Returns list of NavigationStep objects, or None if no path found.
        """
        if not self.map_manager.is_loaded():
            print("[Route] No map loaded.")
            return None

        if from_id not in self.graph or to_id not in self.graph:
            print(f"[Route] Location not in graph: {from_id} or {to_id}")
            return None

        if from_id == to_id:
            return []

        try:
            path = nx.dijkstra_path(
                self.graph, from_id, to_id, weight="distance"
            )
        except nx.NetworkXNoPath:
            print(f"[Route] No path from {from_id} to {to_id}")
            return None

        steps = self._path_to_steps(path)

        # Store as active route
        self.current_route = steps
        self.current_step_index = 0
        self.origin_id = from_id
        self.destination_id = to_id
        self.is_navigating = True

        return steps

    def get_current_step(self) -> Optional[NavigationStep]:
        """Get the current step the user should be executing."""
        if not self.is_navigating or not self.current_route:
            return None
        if self.current_step_index >= len(self.current_route):
            return None
        return self.current_route[self.current_step_index]

    def advance_step(self) -> Optional[NavigationStep]:
        """
        Mark current step as done and move to the next.
        Returns the next step, or None if arrived.
        """
        if not self.is_navigating:
            return None

        self.current_step_index += 1

        if self.current_step_index >= len(self.current_route):
            self.is_navigating = False
            dest_name = self.locations.get(self.destination_id, {}).get("name", "destination")
            return None  # Arrived

        return self.get_current_step()

    def cancel_navigation(self):
        """Cancel current navigation."""
        self.is_navigating = False
        self.current_route = []
        self.current_step_index = 0

    def get_remaining_steps(self) -> int:
        """How many steps remain including the current one."""
        if not self.is_navigating:
            return 0
        return len(self.current_route) - self.current_step_index

    def get_progress_summary(self) -> str:
        """E.g. 'Step 2 of 4 — Heading to cafeteria'"""
        if not self.is_navigating:
            return "Not navigating"
        total = len(self.current_route)
        current = self.current_step_index + 1
        dest = self.locations.get(self.destination_id, {}).get("name", "destination")
        return f"Step {current} of {total} — Heading to {dest}"

    def format_full_route(self) -> str:
        """Return the full route as a human-readable string."""
        if not self.current_route:
            return "No route planned."
        lines = []
        for step in self.current_route:
            prefix = "->" if not step.is_final else "[DEST]"
            lines.append(f"  {prefix} Step {step.step_number}: {step.instruction}")
        return "\n".join(lines)

    # ── Step Building ──────────────────────────────────────────────────────────

    def _path_to_steps(self, path: List[str]) -> List[NavigationStep]:
        steps = []
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id   = path[i + 1]
            from_name = self.locations.get(from_id, {}).get("name", from_id)
            to_name   = self.locations.get(to_id, {}).get("name", to_id)

            edge_data = self.graph.get_edge_data(from_id, to_id) or {}
            distance   = edge_data.get("distance", 10)
            turn       = edge_data.get("turn", "straight")
            hint       = edge_data.get("hint", "")
            is_final   = (i == len(path) - 2)

            instruction = self._build_instruction(
                step_num=i + 1,
                from_name=from_name,
                to_name=to_name,
                turn=turn,
                distance=distance,
                hint=hint,
                is_final=is_final,
            )

            steps.append(NavigationStep(
                step_number=i + 1,
                instruction=instruction,
                from_location=from_id,
                to_location=to_id,
                from_name=from_name,
                to_name=to_name,
                distance_steps=distance,
                turn_direction=turn,
                landmark_hint=hint,
                is_final=is_final,
            ))

        return steps

    def _build_instruction(
        self,
        step_num: int,
        from_name: str,
        to_name: str,
        turn: str,
        distance: int,
        hint: str,
        is_final: bool,
    ) -> str:
        """Build a natural-language voice instruction for one navigation step."""

        # Turn phrase
        turn_phrases = {
            "straight":     "Go straight",
            "left":         "Turn left",
            "right":        "Turn right",
            "slight_left":  "Bear slightly left",
            "slight_right": "Bear slightly right",
        }
        turn_phrase = turn_phrases.get(turn, "Continue")

        # Distance phrase
        if distance <= 3:
            dist_phrase = "a few steps"
        elif distance <= 8:
            dist_phrase = f"about {distance} steps"
        else:
            dist_phrase = f"about {distance} steps"

        if is_final:
            instruction = f"{turn_phrase} for {dist_phrase}. {to_name} will be on your {'left' if turn == 'left' else 'right' if turn == 'right' else 'side'}. You have arrived."
            if hint:
                instruction += f" {hint}."
        else:
            instruction = f"{turn_phrase} for {dist_phrase} toward {to_name}."
            if hint:
                instruction += f" {hint}."

        return instruction
