from __future__ import annotations

import re
from typing import Any

from rllm.agents.agent import Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine

DEFAULT_UPDATER_SYSTEM_PROMPT = (
    "You are an expert prompt updater. Analyze recent trajectories, rewards, and feedback to improve "
    "the solver system prompt. Return ONLY the revised prompt wrapped in <prompt>...</prompt> tags."
)

_PROMPT_PATTERN = re.compile(r"<prompt>(.*?)</prompt>", flags=re.IGNORECASE | re.DOTALL)


def extract_prompt_from_response(text: str) -> str | None:
    """Extract a revised prompt from updater output."""
    match = _PROMPT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


class ErlPromptUpdater:
    """Shared prompt-updater utility used by task-specific ERL workflows."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        system_prompt: str = DEFAULT_UPDATER_SYSTEM_PROMPT,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.9}

    async def propose_prompt(self, state: str, current_prompt: str) -> tuple[str, Trajectory]:
        """Ask the updater policy for a revised prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state},
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, **self.sampling_params)
        content = output.content or output.text or ""
        extracted_prompt = extract_prompt_from_response(content)
        new_prompt = extracted_prompt if extracted_prompt else current_prompt

        step = Step(
            chat_completions=messages + [{"role": "assistant", "content": content, "reasoning": output.reasoning}],
            thought=output.reasoning,
            action=new_prompt,
            model_output=output,
            info={"previous_prompt": current_prompt},
        )
        trajectory = Trajectory(name="erl_updater", steps=[step], info={"previous_prompt": current_prompt})
        return new_prompt, trajectory
