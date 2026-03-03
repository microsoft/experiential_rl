from __future__ import annotations

from typing import Any, Sequence

from rllm.agents.agent import Action
from rllm.environments.base.base_env import BaseEnv


class SokobanEnv(BaseEnv):
    """A lightweight Sokoban environment with text rendering for LLM control."""

    DEFAULT_LAYOUT: list[str] = [
        "#######",
        "#@ $. #",
        "#  #  #",
        "#     #",
        "#     #",
        "#######",
    ]

    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        3: "Right",
        4: "Up",
    }
    MOVE_DELTAS = {
        1: (0, -1),   # Left
        2: (1, 0),    # Down
        3: (0, 1),    # Right
        4: (-1, 0),   # Up
    }

    INVALID_ACTION = 0

    DEFAULT_SYMBOLS: dict[str, str] = {
        "player": "@",
        "player_on_goal": "+",
        "box": "$",
        "box_on_goal": "*",
        "goal": ".",
        "wall": "#",
        "floor": " ",
    }

    def __init__(
        self,
        layout: Sequence[str] | None = None,
        max_steps: int | None = 64,
        symbol_map: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.initial_layout: list[str] = list(layout) if layout is not None else list(self.DEFAULT_LAYOUT)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.seed = seed
        self.symbol_map = self.DEFAULT_SYMBOLS.copy()
        if symbol_map:
            self.symbol_map.update({k: str(v) for k, v in symbol_map.items()})
        self._apply_symbol_map()
        self.task_metadata: dict[str, Any] = {}

        # Runtime state
        self.height: int = 0
        self.width: int = 0
        self.walls: set[tuple[int, int]] = set()
        self.goals: set[tuple[int, int]] = set()
        self.boxes: set[tuple[int, int]] = set()
        self.player: tuple[int, int] | None = None
        self.step_count: int = 0

        self.reset()

    def _apply_symbol_map(self) -> None:
        wall = self.symbol_map.get("wall", self.DEFAULT_SYMBOLS["wall"])
        floor = self.symbol_map.get("floor", self.DEFAULT_SYMBOLS["floor"])
        goal = self.symbol_map.get("goal", self.DEFAULT_SYMBOLS["goal"])
        box = self.symbol_map.get("box", self.DEFAULT_SYMBOLS["box"])
        box_on_goal = self.symbol_map.get("box_on_goal", self.DEFAULT_SYMBOLS["box_on_goal"])
        player = self.symbol_map.get("player", self.DEFAULT_SYMBOLS["player"])
        player_on_goal = self.symbol_map.get("player_on_goal", self.DEFAULT_SYMBOLS["player_on_goal"])

        self.GRID_LOOKUP = {
            0: f" {floor} \t",
            1: f" {wall} \t",
            2: f" {goal} \t",
            3: f" {box} \t",
            4: f" {box_on_goal} \t",
            5: f" {player} \t",
            6: f" {player_on_goal} \t",
        }

    def _parse_layout(self, layout_lines: Sequence[str]) -> None:
        if not layout_lines:
            raise ValueError("Layout must contain at least one row.")
        width = len(layout_lines[0])
        if any(len(row) != width for row in layout_lines):
            raise ValueError("All layout rows must have the same width.")

        self.height = len(layout_lines)
        self.width = width
        self.walls.clear()
        self.goals.clear()
        self.boxes.clear()
        self.player = None

        for r, line in enumerate(layout_lines):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == ".":
                    self.goals.add((r, c))
                elif ch == "$":
                    self.boxes.add((r, c))
                elif ch == "*":
                    self.boxes.add((r, c))
                    self.goals.add((r, c))
                elif ch == "@":
                    self.player = (r, c)
                elif ch == "+":
                    self.player = (r, c)
                    self.goals.add((r, c))
                elif ch == " ":
                    continue
                else:
                    raise ValueError(f"Unsupported character '{ch}' in layout.")

        if self.player is None:
            raise ValueError("Layout must include a player position marked with '@' or '+'.")

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        r, c = position
        return 0 <= r < self.height and 0 <= c < self.width

    def _can_occupy(self, position: tuple[int, int]) -> bool:
        if not self._in_bounds(position):
            return False
        if position in self.walls:
            return False
        if position in self.boxes:
            return False
        return True

    def _state_grid(self) -> list[list[int]]:
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for r, c in self.walls:
            grid[r][c] = 1
        for r, c in self.goals:
            grid[r][c] = 2
        for r, c in self.boxes:
            grid[r][c] = 3 if grid[r][c] != 2 else 4
        if self.player is not None:
            r, c = self.player
            if grid[r][c] == 2:
                grid[r][c] = 6
            else:
                grid[r][c] = 5
        return grid

    def reset(self, task: dict[str, Any] | None = None):
        task = task or {}
        layout = task.get("layout", self.initial_layout)
        symbol_override = task.get("symbol_map")
        if symbol_override:
            self.symbol_map.update({k: str(v) for k, v in symbol_override.items()})
            self._apply_symbol_map()

        self.max_steps = int(task.get("max_steps", self.max_steps)) if task.get("max_steps", None) is not None else self.max_steps
        self.step_count = 0
        self._parse_layout(layout)
        self.task_metadata = {k: v for k, v in task.items() if k not in {"layout", "max_steps", "symbol_map"}}
        return self.render(), {}

    def finished(self) -> bool:
        return self.success() or (self.max_steps is not None and self.step_count >= self.max_steps)

    def success(self) -> bool:
        return self.goals and self.goals.issuperset(self.boxes) and len(self.boxes) == len(self.goals)

    def step(self, action: int):
        if self.success():
            return self.render(), 1.0, True, {"action_is_effective": False, "max_steps": self.max_steps}

        if isinstance(action, Action):
            action = action.action
        action = int(action) if action is not None else self.INVALID_ACTION

        reward = 0.0
        info = {"action_is_effective": False, "max_steps": self.max_steps}

        if action in self.MOVE_DELTAS:
            dr, dc = self.MOVE_DELTAS[action]
            target = (self.player[0] + dr, self.player[1] + dc)  # type: ignore[index]

            if self._in_bounds(target) and target not in self.walls:
                if target in self.boxes:
                    push_to = (target[0] + dr, target[1] + dc)
                    if self._can_occupy(push_to):
                        self.boxes.remove(target)
                        self.boxes.add(push_to)
                        self.player = target
                        info["action_is_effective"] = True
                elif self._can_occupy(target):
                    self.player = target
                    info["action_is_effective"] = True

        self.step_count += 1
        done = self.finished()
        if self.success():
            reward = 1.0

        return self.render(), reward, done, info

    def render(self, mode: str = "tiny_rgb_array"):
        assert mode in ["tiny_rgb_array", "list", "state"]
        state = self._state_grid()
        if mode == "state":
            return state
        if mode == "list":
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in state]
        lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
        return "\n".join("".join(lookup(cell) for cell in row) for row in state)

    @staticmethod
    def from_dict(env_info: dict[str, Any]) -> "SokobanEnv":
        kwargs: dict[str, Any] = {}
        for key in ("layout", "max_steps", "seed", "symbol_map"):
            if key in env_info and env_info[key] is not None:
                kwargs[key] = env_info[key]
        env = SokobanEnv(**kwargs)
        env.task_metadata = {k: v for k, v in env_info.items() if k not in kwargs}
        return env

    @staticmethod
    def is_multithread_safe() -> bool:
        # Pure Python state; no shared mutable state across instances.
        return True
