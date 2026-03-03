from __future__ import annotations

from typing import Any

from rllm.agents.agent import Trajectory
from rllm.agents.frozenlake_agent import FrozenLakeAgent

# Default symbols used throughout the prompts and environment rendering
# DEFAULT_SYMBOLS: dict[str, str] = {
#     "player": "P",
#     "goal": "G",
#     "hole": "O",
#     "frozen": "_",
#     "player_hole": "X",
#     "player_goal": "√",
# }

# DEFAULT_PROMPT_TEMPLATE = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant navigating a frozen lake.

# FrozenLake Quick Guide
# Goal: Reach the goal ({goal}). Player ({player}) and Goal ({goal}) must overlap.

# Symbols:
# {frozen} Frozen | {hole} Hole | {goal} Goal | {player} Player

# Rules:
# 1. Avoid falling into holes ({hole}).
# 2. Frozen tiles are slippery; you may move perpendicular to your intended direction.

# Valid Actions (separated by | ):
# Up | Down | Left | Right

# Rewards:
# Fall into hole: 0
# Reach goal: +1.0

# You will be provided the current observation; decide the next action.
# Show your thought process, then place the final action inside ``` ```.
# Only output the NEXT ACTION in ``` ```. For example, ```Up```.
# Plan ahead and reach the goal in the minimum number of steps.
# """

DEFAULT_SYMBOLS: dict[str, str] = {
    "player": "A",
    "goal": "B",
    "hole": "C",
    "frozen": "D",
    "player_hole": "X",
    "player_goal": "√",
}

# DEFAULT_PROMPT_TEMPLATE = """You are an agent playing a game on a grid, acting as a reasoning engine.

# Valid Actions (separated by | ):
# Up | Down | Left | Right

# You will be provided the current observation; decide the next action.
# Show your thought process, then place the final action inside ``` ```.
# Only output the NEXT ACTION in ``` ```. For example, ```Up```.
# Plan ahead and reach the goal in the minimum number of steps.
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
        raise ValueError(f"Missing symbol '{missing}' required by the FrozenLake prompt template.") from exc


class ErlFrozenLakeAgent(FrozenLakeAgent):
    """FrozenLake agent variant that supports runtime prompt overrides for ERL reflection."""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool | None = True,
        use_multistep_prompt: bool | None = False,
        use_accumulate_history: bool | None = True,
        system_prompt: str | None = None,
        multishot_system_prompt: str | None = None,
        prompt_template: str | None = None,
        symbol_map: dict[str, str] | None = None,
    ):
        self.symbol_map = DEFAULT_SYMBOLS.copy()
        if symbol_map:
            self.symbol_map.update({k: str(v) for k, v in symbol_map.items()})

        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        computed_prompt = system_prompt or build_system_prompt(self.symbol_map, self.prompt_template)

        self.system_prompt_override = computed_prompt
        self.multishot_system_prompt_override = multishot_system_prompt

        super().__init__(
            max_steps=max_steps,
            use_accumulate_thinking=use_accumulate_thinking,
            use_multistep_prompt=use_multistep_prompt,
            use_accumulate_history=use_accumulate_history,
        )

    def reset(self) -> None:
        self._trajectory = Trajectory()
        system_prompt = self.system_prompt_override or self.SYSTEM_PROMPT
        multishot_prompt = self.multishot_system_prompt_override or self.MULTI_SHOT_SYSTEM_PROMPT
        self.messages = [
            {
                "role": "system",
                "content": system_prompt if not self.multistep_prompt else multishot_prompt,
            }
        ]
        self.step = 0
        self.current_observation: Any = None
