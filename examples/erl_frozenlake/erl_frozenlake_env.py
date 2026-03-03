from __future__ import annotations

from typing import Any

from examples.erl_frozenlake.erl_frozenlake_agent import DEFAULT_SYMBOLS
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv


class ErlFrozenLakeEnv(FrozenLakeEnv):
    """Thin FrozenLakeEnv wrapper for ERL workflows with flexible symbol mapping."""

    def __init__(self, **kwargs: Any) -> None:
        symbol_map = kwargs.pop("symbol_map", None)
        super().__init__(**kwargs)
        self.symbol_map = DEFAULT_SYMBOLS.copy()
        if symbol_map:
            self.symbol_map.update({k: str(v) for k, v in symbol_map.items()})
        self._apply_symbol_map()
        self.task_metadata: dict[str, Any] = {}

    def _apply_symbol_map(self) -> None:
        player = self.symbol_map.get("player", DEFAULT_SYMBOLS["player"])
        frozen = self.symbol_map.get("frozen", DEFAULT_SYMBOLS["frozen"])
        hole = self.symbol_map.get("hole", DEFAULT_SYMBOLS["hole"])
        goal = self.symbol_map.get("goal", DEFAULT_SYMBOLS["goal"])
        player_hole = self.symbol_map.get("player_hole", DEFAULT_SYMBOLS["player_hole"])
        player_goal = self.symbol_map.get("player_goal", DEFAULT_SYMBOLS["player_goal"])

        # Override lookup tables for rendering
        self.GRID_LOOKUP = {
            0: f" {player} \t",
            1: f" {frozen} \t",
            2: f" {hole} \t",
            3: f" {goal} \t",
            4: f" {player_hole} \t",
            5: f" {player_goal} \t",
        }

    @staticmethod
    def from_dict(env_info: dict[str, Any]) -> "ErlFrozenLakeEnv":
        kwargs: dict[str, Any] = {}
        for key in ("size", "p", "seed", "max_steps", "is_slippery", "desc"):
            if key in env_info and env_info[key] is not None:
                kwargs[key] = env_info[key]
        if "symbol_map" in env_info:
            kwargs["symbol_map"] = env_info["symbol_map"]

        env = ErlFrozenLakeEnv(**kwargs)
        env.task_metadata = {k: v for k, v in env_info.items() if k not in kwargs}
        return env
