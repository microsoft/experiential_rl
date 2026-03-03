from __future__ import annotations

from typing import Any, Sequence

from examples.erl_sokoban.erl_sokoban_agent import DEFAULT_SYMBOLS
from rllm.environments.sokoban.sokoban import SokobanEnv


class ErlSokobanEnv(SokobanEnv):
    """Thin SokobanEnv wrapper for ERL workflows with flexible symbol mapping."""

    def __init__(self, layout: Sequence[str] | None = None, **kwargs: Any) -> None:
        symbol_map = kwargs.pop("symbol_map", None)
        merged_symbols = DEFAULT_SYMBOLS.copy()
        if symbol_map:
            merged_symbols.update({k: str(v) for k, v in symbol_map.items()})
        super().__init__(layout=layout, symbol_map=merged_symbols, **kwargs)
        self.symbol_map = merged_symbols
        self.task_metadata: dict[str, Any] = {}

    @staticmethod
    def from_dict(env_info: dict[str, Any]) -> "ErlSokobanEnv":
        kwargs: dict[str, Any] = {}
        for key in ("layout", "max_steps", "seed", "symbol_map"):
            if key in env_info and env_info[key] is not None:
                kwargs[key] = env_info[key]

        env = ErlSokobanEnv(**kwargs)
        env.task_metadata = {k: v for k, v in env_info.items() if k not in kwargs}
        return env
