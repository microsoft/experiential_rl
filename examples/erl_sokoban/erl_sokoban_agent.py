from __future__ import annotations

from typing import Any

from rllm.agents.agent import Trajectory
from rllm.agents.sokoban_agent import SokobanAgent

DEFAULT_SYMBOLS: dict[str, str] = {
    "player": "A",
    "player_on_goal": "a",
    "box": "B",
    "box_on_goal": "b",
    "goal": "C",
    "wall": "E",
    "floor": "D",
}

# DEFAULT_PROMPT_TEMPLATE = """
# You are an agent playing Sokoban on a grid, acting as a reasoning engine.
# Your decisions are based on your current game rules (your best guess of how the game works)
# and your strategic playbook (your learned strategies). These may be incomplete or incorrect.
# Your only way to interact with the environment is by choosing your NEXT ACTION.

# Symbols:
# - {wall}: wall
# - {floor}: open floor tile
# - {goal}: goal position
# - {box}: box on floor
# - {box_on_goal}: box already on a goal
# - {player}: you
# - {player_on_goal}: you standing on a goal

# Rules (Sokoban):
# - Move with Up, Down, Left, Right.
# - You may push exactly one adjacent box if the square beyond it is free (not a wall or another box).
# - You cannot pull boxes or walk through walls or boxes.
# - Win condition: every box rests on a goal tile.

# Instructions:
# 1. Analyze State: Summarize the current state.
# 2. Predict Long-term Value of Outcomes: Evaluate the strategic value and potential of the current state for the future.
# 3. Predict Immediate Consequences (World Model Simulation): For the top two candidate actions, predict their consequences using a "result-because" structure.
# 4. Select the Best Action: Choose the action leading to the most advantageous future state.

# Your response MUST strictly follow this structure:
# <reason>
# **1. Analysis of the Current State:**
# [Summary of the board state.]

# **2. Prediction of the Value of Current States:**
# [Assessment of the state's strategic value.]
# - **Value:** High / Medium / Low value with justification.

# **3. Prediction of Immediate Consequences:**
# [Analyze ONLY the top 2 candidate actions using the "result-because" structure.]
# - **Action A:** result-because structure.
# - **Action B:** result-because structure.
# </reason>

# Then output the NEXT ACTION inside triple backticks, like this:
# ```Up```

# Always remember:
# - Valid actions: Up, Down, Left, Right.
# - Think step by step, but make the final line only the next action wrapped in triple backticks, e.g., ```Up```.
# - Ensure the move actually progresses the puzzle (do not walk into walls or push boxes into blocked spaces).
# """

DEFAULT_PROMPT_TEMPLATE = """
You are an agent playing a game on a grid, acting as a reasoning engine.
Your decisions are based on your current game rules (your best guess of how the game works)
and your strategic playbook (your learned strategies). These may be incomplete or incorrect.
Your only way to interact with the environment is by choosing your NEXT ACTION.

Instructions:
1. Analyze State: Summarize the current state.
2. Predict Long-term Value of Outcomes (Value Function Evaluation): Evaluate the strategic value
   and potential of the current state for the future.
3. Predict Immediate Consequences (World Model Simulation): For the top two candidate actions,
   predict their consequences using a "result-because" structure.
4. Select the Best Action: Choose the action leading to the most advantageous future state.

Your response MUST strictly follow this structure:
<reason>
**1. Analysis of the Current State:**
[Summary of the board state.]

**2. Prediction of the Value of Current States:**
[Assessment of the state's strategic value.]
- **Value:** High / Medium / Low value with justification.

**3. Prediction of Immediate Consequences:**
[Analyze ONLY the top 2 candidate actions using the "result-because" structure.]
- **Action A:** result-because structure.
- **Action B:** result-because structure.
</reason>

Then output the NEXT ACTION inside triple backticks, like this:
```Up```

Always remember:
- Valid actions: Up, Down, Left, Right.
- Think step by step, but make the final line only the next action wrapped in triple backticks, e.g., ```Up```.
"""


def build_system_prompt(symbol_map: dict[str, str] | None = None, template: str | None = None) -> str:
    merged = DEFAULT_SYMBOLS.copy()
    if symbol_map:
        merged.update({k: str(v) for k, v in symbol_map.items()})
    prompt_template = template or DEFAULT_PROMPT_TEMPLATE
    try:
        return prompt_template.format(**merged)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f"Missing symbol '{missing}' required by the Sokoban prompt template.") from exc


class ErlSokobanAgent(SokobanAgent):
    """Sokoban agent variant that supports runtime prompt overrides for ERL reflection."""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_history: bool | None = True,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        symbol_map: dict[str, str] | None = None,
    ) -> None:
        self.symbol_map = DEFAULT_SYMBOLS.copy()
        if symbol_map:
            self.symbol_map.update({k: str(v) for k, v in symbol_map.items()})

        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        computed_prompt = system_prompt or build_system_prompt(self.symbol_map, self.prompt_template)

        self.system_prompt_override = computed_prompt
        super().__init__(
            max_steps=max_steps,
            use_accumulate_history=use_accumulate_history,
            system_prompt=self.system_prompt_override,
            symbol_map=self.symbol_map,
        )

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt_override}]
        self.step = 0
        self.current_observation: Any = None
