from __future__ import annotations

import copy
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.environments.sokoban import SokobanEnv


DEFAULT_SYSTEM_PROMPT = """You are a Sokoban puzzle solver playing on a grid.

Symbols:
- {wall}: wall | {floor}: empty floor
- {goal}: goal location
- {box}: box on floor | {box_on_goal}: box already on a goal
- {player}: your position | {player_on_goal}: you standing on a goal

Rules:
1) Move using Up / Down / Left / Right.
2) You may push exactly one adjacent box if the square beyond it is free.
3) You cannot pull boxes or walk through walls or boxes.
4) The puzzle is solved when every box sits on a goal.

Always show your reasoning, then put ONLY the next move inside triple backticks, e.g., ```Left```.
"""


class SokobanAgent(BaseAgent):
    """Sokoban agent that produces a single action per model turn."""

    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_history: bool | None = True,
        system_prompt: str | None = None,
        symbol_map: dict[str, str] | None = None,
    ):
        self.symbol_map = SokobanEnv.DEFAULT_SYMBOLS.copy()
        if symbol_map:
            self.symbol_map.update({k: str(v) for k, v in symbol_map.items()})

        template_kwargs = {k: str(v) for k, v in self.symbol_map.items()}
        prompt = system_prompt or self.SYSTEM_PROMPT.format(**template_kwargs)

        self.system_prompt_override = prompt
        self.max_steps = max_steps
        self.accumulate_history = use_accumulate_history

        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step: int = 0
        self.current_observation: Any = None
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if self._trajectory.steps:
            cur_step = self._trajectory.steps[-1]
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        current_obs_str = str(observation)
        prompt_lines = [
            f"Current Board ({self.step}):",
            current_obs_str,
            "Puzzle not solved yet. Provide the next move.",
        ]

        if self._trajectory.steps and self._trajectory.steps[-1].action is not None:
            last_obs = self._trajectory.steps[-1].observation
            if last_obs == current_obs_str:
                prompt_lines.append("Your last move had no effect (likely hit a wall or tried to push a blocked box). Ensure you output a valid direction: Up, Down, Left, or Right.")
            elif info.get("action_is_effective"):
                prompt_lines.append("Last move succeeded (moved or pushed a box); puzzle still unsolved—plan the next move.")

        if self.max_steps is not None:
            remaining = self.max_steps - self.step
            if remaining > 0:
                prompt_lines.append(f"Maximum remaining moves: {remaining}.")

        self.messages.append({"role": "user", "content": "\n".join(prompt_lines)})
        self.current_observation = current_obs_str

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        direction_map = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = response
        action_str = str(SokobanEnv.INVALID_ACTION)

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = response.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = response[:last_match_index].strip()

            extracted = last_match_content.lower()
            if extracted in direction_map:
                action_str = str(direction_map[extracted])
            elif extracted.isdigit() and int(extracted) in direction_map.values():
                action_str = str(int(extracted))

        return thought, action_str

    def _process_action_for_validation(self, response: str) -> str:
        _, action_str = self._parse_model_response(response)
        return action_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        thought, action_str = self._parse_model_response(response)
        self.messages.append({"role": "assistant", "content": response})

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)
        self.step += 1
        return Action(action=action_str)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self.accumulate_history:
            return self.messages
        if len(self.messages) <= 1:
            return self.messages
        return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt_override}]
        self.step = 0
        self.current_observation = None
