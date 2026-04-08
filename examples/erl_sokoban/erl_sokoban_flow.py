from __future__ import annotations

from collections import deque
import uuid
import copy
from typing import Any, Callable, Sequence

from rllm.agents.agent import Episode, Trajectory
from rllm.engine import AgentExecutionEngine, AgentWorkflowEngine, RolloutEngine
from rllm.workflows.cumulative_workflow import CumulativeWorkflow
from rllm.workflows.workflow import Workflow

from examples.erl.erl_flow import ErlPromptUpdater
from examples.erl_sokoban.erl_sokoban_agent import ErlSokobanAgent, build_system_prompt
from examples.erl_sokoban.erl_sokoban_env import ErlSokobanEnv

FeedbackFunction = Callable[[dict[str, Any], Trajectory], str]

DEFAULT_SOLVER_SYSTEM_PROMPT = build_system_prompt()
DEFAULT_UPDATER_SYSTEM_PROMPT = (
    "You are a chief scientific strategist and master tactician. "
    "Your mission is to analyze extensive field data from numerous operations to distill and refine the "
    "Master Rulebook of a complex game. "
    "You will be presented with a large collection of highly successful trajectories and critical failure "
    "trajectories, collected over a long period. "
    "Your primary task is to perform a deep, comparative analysis to understand the fundamental principles "
    "of victory and defeat. Act as a grand strategist, identifying universal patterns and high-level causal "
    "relationships. Your goal is to synthesize these insights to produce the next generation's Master Rulebook, "
    "making it more robust, accurate, and effective. "
    "Core Principles: Think Long-Term—focus on universal, strategic truths that hold across diverse scenarios; "
    "Learn from Contrast—extract insights by comparing winners and losers; Synthesize and Consolidate—produce a "
    "single unified theory; Be Authoritative and Concise—state rules as definitive principles. "
    "Your output MUST be a single consolidated <prompt> block representing the new Master Rulebook:\n"
    "<prompt>\n"
    "<game_rules>\n"
    "**1. Symbol Meanings:** [Clarify what each key symbol represents within the game world (entities, terrain types, goals, risks, etc.), and explicitly identify which symbol corresponds to the player/agent.]\n"
    "**2. Information & Interpretation:** [Define how elements reliably inform about the game state.]\n"
    "**3. Gameplay & Actions:** [Define the core mechanics and interactions.]\n"
    "**4. Action Effects:** [Describe the predictable outcomes of actions.]\n"
    "**5. Game Objective & Termination:** [State the ultimate win/loss conditions.]\n"
    "</game_rules>\n"
    "<strategy>\n"
    "**1. Core Strategies:** [Describe foundational, high-level strategic priorities that lead to victory.]\n"
    "**2. Tactical Tips:** [List widely applicable, advantageous situational plays.]\n"
    "</strategy>\n"
    "</prompt>"
)

ACTION_CODE_TO_TEXT = {
    "1": "Left",
    "2": "Down",
    "3": "Right",
    "4": "Up",
}
ACTION_CODE_TO_TEXT.update({1: "Left", 2: "Down", 3: "Right", 4: "Up"})

ACTION_AND_FORMAT_REMINDER = (
    "Always remember:\n"
    "- Valid actions: Up, Down, Left, Right.\n"
    "- Only push a box if the space behind it is free.\n"
    "- Think step by step, but make the final line only the next action wrapped in triple backticks, e.g., ```Up```."
)

SECOND_ATTEMPT_GENERIC_INSTRUCTION = (
    "You are provided with the model's past attempt data, including observations, actions, rewards, and feedback. "
    "Use this information as context to make a better next-attempt decision policy. "
    "Follow the action/output format exactly."
)


def default_feedback_fn(task: dict[str, Any], trajectory: Trajectory) -> str:
    action_steps = [step for step in trajectory.steps if step.action is not None]
    steps_taken = len(action_steps)
    succeeded = trajectory.reward >= 1.0
    last_step = action_steps[-1] if action_steps else (trajectory.steps[-1] if trajectory.steps else None)
    last_info = dict(last_step.info or {}) if last_step else {}
    max_steps = last_info.get("max_steps")

    def describe_outcome() -> str:
        if last_step is None:
            return "No valid actions were recorded."
        if succeeded:
            return "Solved the puzzle (all boxes placed on goals)."
        if max_steps and steps_taken >= max_steps:
            return f"Hit the max step limit ({max_steps})."
        if last_step.done:
            return "Episode ended without solving the puzzle."
        return "Episode ended unexpectedly (likely stalled)."

    ineffective_moves = sum(1 for step in action_steps if step.info.get("action_is_effective") is False)
    trailing_ineffective = 0
    for step in reversed(action_steps):
        if step.info.get("action_is_effective") is False:
            trailing_ineffective += 1
        else:
            break

    last_action_code = last_step.action if last_step and last_step.action is not None else None
    last_action_text = ACTION_CODE_TO_TEXT.get(last_action_code, ACTION_CODE_TO_TEXT.get(str(last_action_code), str(last_action_code))) if last_action_code is not None else None

    outcome = describe_outcome()
    if not succeeded and ineffective_moves:
        outcome += f" {ineffective_moves} action(s) had no effect (likely ran into a wall or tried to push a blocked box)."
        if trailing_ineffective >= 2:
            outcome += f" Last {trailing_ineffective} action(s) in a row did nothing."
    if last_action_text:
        outcome += f" Last action taken: {last_action_text}."

    final_state_hint: str | None = None
    if last_step:
        if max_steps and steps_taken >= max_steps:
            final_state_hint = "board unsolved when max steps were reached"
    if final_state_hint:
        outcome += f" Final state: {final_state_hint}."

    return f"{outcome} Reward={float(trajectory.reward):.2f}, steps={steps_taken}."


class ErlSokobanStateBuilder:
    """Compose prompt updater context from Sokoban attempts."""

    CANONICAL_SYMBOL_BY_KEY = {
        "wall": "#",
        "floor": " ",
        "goal": ".",
        "box": "$",
        "box_on_goal": "*",
        "player": "@",
        "player_on_goal": "+",
    }
    CANONICAL_KEY_BY_SYMBOL = {value: key for key, value in CANONICAL_SYMBOL_BY_KEY.items()}

    def __init__(self, max_examples: int = 8, symbol_map: dict[str, str] | None = None):
        self.max_examples = max_examples
        self.symbol_map = {
            key: str((symbol_map or {}).get(key, value))
            for key, value in self.CANONICAL_SYMBOL_BY_KEY.items()
        }

    def _render_layout(self, layout_rows: Sequence[str], symbol_override: dict[str, Any] | None = None) -> str:
        active_symbol_map = self.symbol_map.copy()
        if symbol_override:
            active_symbol_map.update({key: str(value) for key, value in symbol_override.items() if value is not None})

        rendered_rows: list[str] = []
        for row in layout_rows:
            rendered_rows.append(
                "".join(
                    active_symbol_map.get(self.CANONICAL_KEY_BY_SYMBOL.get(ch, ""), ch)
                    for ch in str(row)
                )
            )
        return "\n".join(rendered_rows)

    def __call__(self, initial_prompt: str, batch: Sequence[dict[str, Any]], attempts: Sequence[dict[str, Any]], metrics: dict[str, float]) -> str:
        lines: list[str] = [
            "## Inferred information from past attempts (may be inaccurate)",
            initial_prompt.strip(),
            "",
            "## Batch Metrics",
        ]

        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.4f}")

        lines.extend(["", "## Recent Attempts"])

        for idx, attempt in enumerate(attempts[-self.max_examples :], start=1):
            task = attempt.get("example", {})
            lines.append(f"### Attempt {idx}")
            layout_name = task.get("layout_name") or task.get("name") or task.get("id") or "unknown"
            max_steps = task.get("max_steps", "<default>")
            lines.append(f"Layout: {layout_name} | max_steps: {max_steps}")
            layout_rows = task.get("layout")
            if layout_rows:
                rendered_layout = self._render_layout(layout_rows, task.get("symbol_map"))
                lines.append(rendered_layout)
                lines.append("")

            def append_trace(trace: list[dict[str, Any]], label: str) -> None:
                if not trace:
                    lines.append(f"{label}: <no steps recorded>")
                    return
                lines.append(label + ":")
                for turn, entry in enumerate(trace, start=1):
                    obs = entry.get("observation") or "<missing observation>"
                    action_text = entry.get("action_text") or entry.get("action_code") or "<no action>"
                    reward = entry.get("reward", 0.0)
                    effective = entry.get("action_effective")
                    done = entry.get("done")
                    status_bits = f"reward={reward:.2f}"
                    if effective is not None:
                        status_bits += f", effective={bool(effective)}"
                    if done is not None:
                        status_bits += f", done={bool(done)}"

                    action_comment = ""
                    if effective is False:
                        action_comment = "The agent did not move (likely hit a wall or tried to push into a blocked space)."
                    elif effective is True:
                        if reward > 0 or done:
                            action_comment = "The agent solved the puzzle (all boxes on goals)."
                        else:
                            action_comment = "The agent moved or pushed a box; puzzle not solved yet."
                    lines.append(f"- Observation {turn} (seen before choosing an action):\n{obs}")
                    lines.append(f"- Action {turn} (taken after Observation {turn}): {action_text} | {status_bits}")
                    if action_comment:
                        lines.append(f"  -> {action_comment}")

            append_trace(attempt.get("first_trace", []), "Attempt #1 Trace")
            lines.append(f"Reward (attempt #1): {attempt.get('first_reward', 0.0):.4f} | Correct: {attempt.get('first_correct', False)}")

            if "second_trace" in attempt:
                append_trace(attempt.get("second_trace", []), "Attempt #2 Trace")
                lines.append(f"Reward (attempt #2): {attempt.get('second_reward', 0.0):.4f} | Correct: {attempt.get('second_correct', False)}")

            feedback = attempt.get("feedback")
            if feedback:
                lines.append(f"Feedback: {feedback}")
            lines.append("")

        lines.append("")
        lines.append("Think step by step and provide an improved Sokoban instruction enclosed in <prompt>...</prompt> tags.")
        return "\n".join(lines)


class ErlSokobanWorkflow(Workflow):
    """Reflection-enhanced workflow tailored for the Sokoban environment."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        initial_system_prompt: str = DEFAULT_SOLVER_SYSTEM_PROMPT,
        feedback_fn: FeedbackFunction | None = None,
        state_builder: ErlSokobanStateBuilder | None = None,
        train_dataset: Sequence[dict[str, Any]] | None = None,
        batch_size: int = 1,
        max_concurrency: int = 1024,
        agent_class: type[ErlSokobanAgent] = ErlSokobanAgent,
        env_class: type[ErlSokobanEnv] = ErlSokobanEnv,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        solver_tokenizer=None,
        solver_sampling_params: dict[str, Any] | None = None,
        solver_rollout_engine_args: dict[str, Any] | None = None,
        solver_engine_name: str = "workflow",
        workflow_config=None,
        updater_rollout_engine: RolloutEngine | None = None,
        updater_rollout_engine_config: dict[str, Any] | None = None,
        updater_sampling_params: dict[str, Any] | None = None,
        updater_engine_name: str = "openai",
        max_response_length: int = 16384,
        max_prompt_length: int = 4096,
        success_reward_threshold: float = 1.0,
        train_first_attempt: bool = True,
        train_second_attempt_raw: bool = False,
        train_second_attempt_distilled: bool = True,
        train_updater: bool = False,
        train_first_attempt_adv_estimator: str | None = None,
        train_second_attempt_raw_adv_estimator: str | None = None,
        train_second_attempt_distilled_adv_estimator: str | None = None,
        train_updater_adv_estimator: str | None = None,
        no_memory: bool = False,
        no_reflection: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine, executor, **kwargs)
        self.initial_system_prompt = initial_system_prompt.strip()
        self.agent_args_template = dict(agent_args or {})
        self.env_args = dict(env_args or {})
        self.feedback_fn = feedback_fn or default_feedback_fn
        self.state_builder = state_builder or ErlSokobanStateBuilder(
            symbol_map=self.env_args.get("symbol_map") or self.agent_args_template.get("symbol_map")
        )
        self.batch_size = max(1, batch_size)
        self.max_concurrency = max(1, max_concurrency)
        self.updater_engine_name = (updater_engine_name or "openai").lower()
        self.agent_class = agent_class
        self.env_class = env_class
        self.solver_engine_name = solver_engine_name
        self.workflow_config = workflow_config
        self.solver_tokenizer = solver_tokenizer
        self.solver_sampling_params = dict(solver_sampling_params or {})
        self.solver_rollout_engine_args = dict(solver_rollout_engine_args or {})
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.success_reward_threshold = success_reward_threshold
        self.train_first_attempt = bool(train_first_attempt)
        self.train_second_attempt_raw = bool(train_second_attempt_raw)
        self.train_second_attempt_distilled = bool(train_second_attempt_distilled)
        self.train_updater = bool(train_updater)
        self.train_first_attempt_adv_estimator = train_first_attempt_adv_estimator
        self.train_second_attempt_raw_adv_estimator = train_second_attempt_raw_adv_estimator
        self.train_second_attempt_distilled_adv_estimator = train_second_attempt_distilled_adv_estimator
        self.train_updater_adv_estimator = train_updater_adv_estimator
        self.no_memory = bool(no_memory)
        self.no_reflection = bool(no_reflection)

        if self.updater_engine_name == "verl":
            self.updater_rollout_engine = rollout_engine
        elif updater_rollout_engine is not None:
            self.updater_rollout_engine = updater_rollout_engine
        elif updater_rollout_engine_config is not None:
            self.updater_rollout_engine = self._instantiate_rollout_engine(updater_rollout_engine_config)
        else:
            self.updater_rollout_engine = rollout_engine

        self.updater = ErlPromptUpdater(self.updater_rollout_engine, system_prompt=DEFAULT_UPDATER_SYSTEM_PROMPT, sampling_params=updater_sampling_params)

        self.train_dataset = list(train_dataset) if train_dataset is not None else []
        self._train_queue: deque[dict[str, Any]] = deque()
        if self.train_dataset:
            self._reset_train_queue()

        self.active_prompt = self._ensure_action_output_reminder(self.initial_system_prompt)
        self.last_improved_prompt: str | None = None

    async def run(self, task: dict[str, Any], uid: str, **kwargs: Any) -> Episode:
        self.reset(task, uid)

        is_validation = bool(getattr(self.rollout_engine, "validate", False) or kwargs.get("validate", False))
        minibatch = self._extract_minibatch(task, use_training_queue=not is_validation)
        if not minibatch:
            return Episode(id=uid, task=task, is_correct=False, trajectories=[], metrics={}, info={})

        # Always use the initial prompt for the first attempt (both training and validation).
        current_prompt = self.initial_system_prompt

        first_trajectories = await self._run_solver_batch(current_prompt, minibatch, validate=is_validation)
        attempts: list[dict[str, Any]] = []
        first_rewards: list[float] = []
        first_successes = 0

        solver_trajectories: list[Trajectory] = []

        for idx, (example, trajectory) in enumerate(zip(minibatch, first_trajectories, strict=True)):
            trajectory.name = f"solver_attempt_attempt1_batch{idx}"
            self._tag_adv_estimator(trajectory, attempt_label="first", override=self.train_first_attempt_adv_estimator)
            trace = self._extract_trace(trajectory)
            reward = float(trajectory.reward)
            success = reward >= self.success_reward_threshold
            first_rewards.append(reward)
            if success:
                first_successes += 1

            initial_obs = trajectory.steps[0].observation if trajectory.steps else ""
            feedback = self.feedback_fn(example, trajectory)

            attempt_record = {
                "example": example,
                "initial_observation": initial_obs,
                "first_trace": trace,
                "first_reward": reward,
                "first_correct": success,
                "feedback": feedback,
                "first_response": trajectory.steps[-1].model_response if trajectory.steps else "",
            }
            attempts.append(attempt_record)

            if is_validation:
                solver_trajectories.append(trajectory)
            elif self.train_first_attempt:
                trajectory.name = f"{trajectory.name}_attempt1_batch{idx}"
                solver_trajectories.append(trajectory)

        avg_first_reward = sum(first_rewards) / len(first_rewards) if first_rewards else 0.0
        first_success_rate = first_successes / len(first_rewards) if first_rewards else 0.0
        compute_second_metrics = self.train_second_attempt_raw or self.train_second_attempt_distilled
        metrics = {
            "avg_first_reward": avg_first_reward,
            "first_success_rate": first_success_rate,
            "batch_size": float(len(minibatch)),
        }
        all_first_attempts_correct = first_success_rate == 1.0 and len(first_rewards) == len(minibatch)

        state: str | None = None
        logged_updater_traj: Trajectory | None = None
        # Reflect using the most recently improved prompt (if any), otherwise fall back to the initial prompt.
        base_prompt_for_update = self.initial_system_prompt if self.no_memory else (self.last_improved_prompt or self.initial_system_prompt)
        improved_prompt = base_prompt_for_update
        avg_second_reward = avg_first_reward
        second_success_rate = first_success_rate

        if not is_validation:
            # Skip updater/second pass when we're only training on first attempts.
            if all_first_attempts_correct:
                # Nothing to improve; mirror first-attempt results and skip reflection/reruns.
                for attempt_record in attempts:
                    attempt_record["second_trace"] = attempt_record.get("first_trace", [])
                    attempt_record["second_reward"] = attempt_record.get("first_reward", 0.0)
                    attempt_record["second_correct"] = attempt_record.get("first_correct", False)
                    attempt_record["second_response"] = attempt_record.get("first_response", "")
                improved_prompt = base_prompt_for_update
                second_success_rate = first_success_rate
                avg_second_reward = avg_first_reward
                state = None
            elif not (compute_second_metrics or self.train_updater):
                # Mirror first-attempt results for reporting and proceed without rerunning.
                for attempt_record in attempts:
                    attempt_record["second_trace"] = attempt_record.get("first_trace", [])
                    attempt_record["second_reward"] = attempt_record.get("first_reward", 0.0)
                    attempt_record["second_correct"] = attempt_record.get("first_correct", False)
                    attempt_record["second_response"] = attempt_record.get("first_response", "")
                improved_prompt = base_prompt_for_update
                second_success_rate = first_success_rate
                avg_second_reward = avg_first_reward
                state = None
            else:
                state = self.state_builder(base_prompt_for_update, minibatch, attempts, metrics)
                if self.no_reflection:
                    improved_prompt = self._build_generic_second_attempt_prompt(state)
                else:
                    # Build updater state and ask for an improved prompt (for logging/training).
                    improved_prompt, updater_traj = await self.updater.propose_prompt(state, base_prompt_for_update)
                    improved_prompt = self._ensure_action_output_reminder(improved_prompt)
                    if updater_traj is not None:
                        if updater_traj.steps:
                            for step in updater_traj.steps:
                                step.reward = 0.0
                        self._tag_adv_estimator(updater_traj, attempt_label="updater", override=self.train_updater_adv_estimator)
                        logged_updater_traj = updater_traj
                        if self.train_updater:
                            solver_trajectories.append(updater_traj)

                # If we are not running second-attempt rollouts, just carry first metrics forward.
                if not compute_second_metrics:
                    for attempt_record in attempts:
                        attempt_record["second_trace"] = attempt_record.get("first_trace", [])
                        attempt_record["second_reward"] = attempt_record.get("first_reward", 0.0)
                        attempt_record["second_correct"] = attempt_record.get("first_correct", False)
                        attempt_record["second_response"] = attempt_record.get("first_response", "")
                    second_success_rate = first_success_rate
                    avg_second_reward = avg_first_reward
                else:
                    # Re-run only the failed examples with the improved prompt.
                    pending_indices = [i for i, record in enumerate(attempts) if not record.get("first_correct", False)]
                    # If all first attempts succeeded, skip running a second attempt.
                    second_trajectories: list[Trajectory] = []
                    if pending_indices:
                        pending_batch = [attempts[i]["example"] for i in pending_indices]
                        second_trajectories = await self._run_solver_batch(improved_prompt, pending_batch)

                    second_rewards: list[float] = []
                    second_successes = 0
                    second_attempt_success = False

                    # Pre-fill second-attempt fields for already-correct attempts (no rerun needed).
                    for attempt_record in attempts:
                        if attempt_record.get("first_correct", False):
                            attempt_record["second_trace"] = attempt_record.get("first_trace", [])
                            attempt_record["second_reward"] = attempt_record.get("first_reward", 0.0)
                            attempt_record["second_correct"] = True
                            attempt_record["second_response"] = attempt_record.get("first_response", "")
                            second_rewards.append(float(attempt_record["second_reward"]))
                            second_successes += 1

                    # Assign results for rerun attempts.
                    for local_idx, global_idx in enumerate(pending_indices):
                        attempt_record = attempts[global_idx]
                        trajectory = second_trajectories[local_idx]
                        trajectory.name = f"{trajectory.name}_attempt2_batch{global_idx}"
                        self._tag_adv_estimator(trajectory, attempt_label="second_raw", override=self.train_second_attempt_raw_adv_estimator)
                        trace = self._extract_trace(trajectory)
                        reward = float(trajectory.reward)
                        success = reward >= self.success_reward_threshold
                        second_rewards.append(reward)
                        if success:
                            second_successes += 1
                            second_attempt_success = True

                        attempt_record["second_trace"] = trace
                        attempt_record["second_reward"] = reward
                        attempt_record["second_correct"] = success
                        attempt_record["second_response"] = trajectory.steps[-1].model_response if trajectory.steps else ""

                        # Train on second attempt raw
                        if self.train_second_attempt_raw:
                            solver_trajectories.append(trajectory)

                        # Train on second attempt distilled
                        if self.train_second_attempt_distilled:
                            # We need deep copy since list is mutable
                            distilled = copy.deepcopy(trajectory)
                            for step in distilled.steps:
                                step.model_output = None
                                # replace the original system prompt with the inital system prompt
                                if step.chat_completions and step.chat_completions[0].get("role") == "system":
                                    step.chat_completions[0]["content"] = self.initial_system_prompt
                            distilled.name = f"{distilled.name}_distilled"
                            self._tag_adv_estimator(distilled, attempt_label="second_distilled", override=self.train_second_attempt_distilled_adv_estimator)
                            solver_trajectories.append(distilled)

                    avg_second_reward = sum(second_rewards) / len(second_rewards) if second_rewards else 0.0
                    second_success_rate = second_successes / len(second_rewards) if second_rewards else 0.0

                # Only persist the improved prompt if it actually produced a successful second attempt.
                if (not self.no_memory) and (not self.no_reflection) and pending_indices and second_attempt_success:
                    self.last_improved_prompt = improved_prompt

        # Align updater rewards with solver rewards (use second-attempt reward if present, else first-attempt).
        if logged_updater_traj is not None:
            updater_reward = avg_second_reward if compute_second_metrics else avg_first_reward
            logged_updater_traj.reward = updater_reward
            for step in logged_updater_traj.steps:
                step.reward = updater_reward

        for traj in solver_trajectories:
            self.adjust_step_rewards(traj)
            self.compute_trajectory_reward(traj)

        if is_validation:
            metrics_payload = {
                "avg_reward": avg_first_reward,
                "success_rate": first_success_rate,
                "batch_size": float(len(minibatch)),
            }
        elif compute_second_metrics:
            metrics_payload = {
                "avg_first_reward": avg_first_reward,
                "avg_second_reward": avg_second_reward,
                "first_success_rate": first_success_rate,
                "second_success_rate": second_success_rate,
                "batch_size": float(len(minibatch)),
            }
        else:
            # Only first-attempt training/reporting; omit second metrics to avoid None entries.
            metrics_payload = {
                "avg_first_reward": avg_first_reward,
                "first_success_rate": first_success_rate,
                "batch_size": float(len(minibatch)),
            }

        success_rate_for_correct = (
            second_success_rate if (not is_validation and compute_second_metrics) else first_success_rate
        )
        episode = Episode(
            id=uid,
            task=task,
            is_correct=success_rate_for_correct == 1.0,
            trajectories=list(solver_trajectories),
            metrics=metrics_payload,
            info={
                "initial_prompt": current_prompt,
                "active_prompt": self.last_improved_prompt or current_prompt,
                "improved_prompt": improved_prompt,
                "batch_size": len(minibatch),
                "updater_state": None if self.no_reflection else state,
                "attempt_summaries": [
                    {
                        "seed": record["example"].get("seed"),
                        "size": record["example"].get("size"),
                        "p": record["example"].get("p"),
                        "initial_observation": record.get("initial_observation"),
                        "first_actions": record.get("first_actions"),
                        "first_reward": record.get("first_reward"),
                        "first_correct": record.get("first_correct"),
                        "second_actions": record.get("second_actions"),
                        "second_reward": record.get("second_reward"),
                        "second_correct": record.get("second_correct"),
                        "feedback": record.get("feedback"),
                    }
                    for record in attempts
                ],
            },
        )

        if logged_updater_traj is not None:
            episode.trajectories.append(logged_updater_traj)

        return episode

    async def _run_solver_batch(self, system_prompt: str, batch: Sequence[dict[str, Any]], validate: bool = False) -> list[Trajectory]:
        if not batch:
            return []

        agent_args = dict(self.agent_args_template)
        agent_args["system_prompt"] = self._ensure_action_output_reminder(system_prompt)
        max_steps = self._get_configured_max_steps(agent_args)

        if self.solver_engine_name in {"workflow", "cumulative"}:
            resolved_max_steps = max_steps
            if resolved_max_steps is None:
                resolved_max_steps = self.env_args.get("max_steps") or self.agent_args_template.get("max_steps")
                resolved_max_steps = int(resolved_max_steps) if resolved_max_steps is not None else None

            workflow_args = {
                "agent_cls": self.agent_class,
                "env_cls": self.env_class,
                "agent_args": agent_args,
                "env_args": self.env_args,
                "max_steps": resolved_max_steps or 5,
            }
            workflow_engine = AgentWorkflowEngine(
                workflow_cls=CumulativeWorkflow,
                workflow_args=workflow_args,
                rollout_engine=self.rollout_engine,
                config=self.workflow_config,
                n_parallel_tasks=min(self.max_concurrency, max(1, len(batch))),
                retry_limit=1,
            )
            episodes = await workflow_engine.execute_tasks(list(batch))
            trajectories: list[Trajectory] = []
            for episode in episodes:
                if episode.trajectories:
                    traj = episode.trajectories[0]
                else:
                    traj = Trajectory(
                        uid=str(uuid.uuid4()),
                        name="solver_attempt",
                        task=episode.task,
                        steps=[],
                        reward=0.0,
                        info={"warning": "empty_trajectory_from_workflow"},
                    )
                traj.name = "solver_attempt"
                trajectories.append(traj)
            return trajectories

        engine_kwargs: dict[str, Any] = {
            "agent_class": self.agent_class,
            "env_class": self.env_class,
            "agent_args": agent_args,
            "env_args": self.env_args,
            "n_parallel_agents": min(self.max_concurrency, max(1, len(batch))),
            "max_response_length": self.max_response_length,
            "max_prompt_length": self.max_prompt_length,
        }
        if max_steps is not None:
            engine_kwargs["max_steps"] = max_steps

        if self.solver_engine_name == "verl":
            rollout_manager = getattr(self.rollout_engine, "rollout_manager", self.rollout_engine)
            tokenizer = self.solver_tokenizer or getattr(self.rollout_engine, "tokenizer", None)
            verl_sampling_params: dict[str, Any] = {"meta_info": {"validate": bool(validate)}}
            engine_kwargs.update(
                engine_name="verl",
                rollout_engine=rollout_manager,
                tokenizer=tokenizer,
                config=self.workflow_config,
                sampling_params=verl_sampling_params,
            )
        else:
            engine_kwargs.update(
                engine_name="openai",
                tokenizer=self.solver_tokenizer,
                sampling_params=dict(self.solver_sampling_params),
                rollout_engine_args=dict(self.solver_rollout_engine_args),
            )

        engine = AgentExecutionEngine(**engine_kwargs)

        trajectories = await engine.execute_tasks(list(batch))
        for traj in trajectories:
            traj.name = "solver_attempt"
        return trajectories

    def _extract_minibatch(self, task: dict[str, Any], use_training_queue: bool = True) -> list[dict[str, Any]]:
        provided_batch: list[dict[str, Any]] = []
        if isinstance(task, dict) and task.get("batch"):
            provided_batch = list(task["batch"])
        elif task:
            provided_batch = [task]

        if not use_training_queue:
            return provided_batch

        batch: list[dict[str, Any]] = list(provided_batch)

        # Top up from the training queue until we meet the configured batch size.
        while len(batch) < self.batch_size:
            supplemental = self._pop_train_example()
            if supplemental is None:
                break
            batch.append(supplemental)

        # If nothing could be sampled (e.g., empty train set), fall back to provided task(s).
        if not batch:
            batch = provided_batch
        return batch

    def _reset_train_queue(self) -> None:
        shuffled = self.train_dataset.copy()
        import random

        random.shuffle(shuffled)
        self._train_queue = deque(shuffled)

    def _sample_minibatch(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while len(batch) < self.batch_size:
            item = self._pop_train_example()
            if item is None:
                break
            batch.append(item)
        return batch

    def _pop_train_example(self) -> dict[str, Any] | None:
        if not self.train_dataset:
            return None
        if not self._train_queue:
            self._reset_train_queue()
            if not self._train_queue:
                return None
        return self._train_queue.popleft()

    def _get_configured_max_steps(self, agent_args: dict[str, Any]) -> int | None:
        max_steps = agent_args.get("max_steps") or self.env_args.get("max_steps")
        if max_steps is not None:
            return int(max_steps)

        workflow_cfg = getattr(self.workflow_config, "workflow", None) if self.workflow_config is not None else None
        if workflow_cfg is not None:
            candidate = getattr(workflow_cfg, "agent_max_steps", None) or getattr(workflow_cfg, "env_max_steps", None)
            if candidate is not None:
                return int(candidate)
        return None

    def _extract_trace(self, trajectory: Trajectory) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for step in trajectory.steps:
            action_code = step.action
            action_text = ACTION_CODE_TO_TEXT.get(action_code, ACTION_CODE_TO_TEXT.get(str(action_code), str(action_code)))
            info = step.info or {}
            entry = {
                "observation": step.observation,
                "action_code": action_code,
                "action_text": action_text,
                "reward": float(step.reward),
                "done": bool(step.done),
                "action_effective": info.get("action_is_effective"),
            }
            trace.append(entry)
        return trace

    @staticmethod
    def _combine_prompts(initial: str, improved: str) -> str:
        improved = (improved or "").strip()
        if not improved:
            return initial
        if not initial.strip():
            return improved
        separator = "\n\n" if not initial.endswith("\n") else "\n"
        return f"{initial}{separator}{improved}"

    @staticmethod
    def _ensure_action_output_reminder(prompt: str) -> str:
        base_prompt = (prompt or "").rstrip()
        reminder = ACTION_AND_FORMAT_REMINDER.strip()
        if not base_prompt:
            return reminder
        if base_prompt.endswith(reminder):
            return base_prompt
        return f"{base_prompt}\n\n{reminder}"

    def _build_generic_second_attempt_prompt(self, state: str) -> str:
        improved_prompt = (
            f"{self.initial_system_prompt.strip()}\n\n"
            f"{SECOND_ATTEMPT_GENERIC_INSTRUCTION}\n\n"
            f"{state.strip()}"
        )
        return self._ensure_action_output_reminder(improved_prompt)

    @staticmethod
    def _instantiate_rollout_engine(config: dict[str, Any]) -> RolloutEngine:
        module_path = config.get("module")
        class_name = config.get("class")
        kwargs = dict(config.get("kwargs", {}))

        if module_path is None or class_name is None:
            raise ValueError("updater_rollout_engine_config must include 'module' and 'class' keys")

        tokenizer_config = config.get("tokenizer_config")
        if tokenizer_config is not None:
            tokenizer_module = tokenizer_config.get("module")
            tokenizer_class = tokenizer_config.get("class")
            tokenizer_kwargs = tokenizer_config.get("kwargs", {})

            if tokenizer_module is None or tokenizer_class is None:
                raise ValueError("tokenizer_config must include 'module' and 'class' keys")

            import importlib

            tok_module = importlib.import_module(tokenizer_module)
            tokenizer_cls = getattr(tok_module, tokenizer_class)

            if hasattr(tokenizer_cls, "from_pretrained"):
                tokenizer = tokenizer_cls.from_pretrained(**tokenizer_kwargs)
            else:
                tokenizer = tokenizer_cls(**tokenizer_kwargs)
            kwargs["tokenizer"] = tokenizer

        import importlib

        module = importlib.import_module(module_path)
        engine_cls = getattr(module, class_name)
        return engine_cls(**kwargs)

    @staticmethod
    def _tag_adv_estimator(trajectory: Trajectory, attempt_label: str | None, override: str | None) -> None:
        """Annotate trajectory/steps with attempt label and adv estimator override for downstream training."""
        if attempt_label:
            trajectory.info.setdefault("attempt", attempt_label)
        if override:
            trajectory.info["adv_estimator"] = override
        for step in trajectory.steps:
            if attempt_label and "attempt" not in step.info:
                step.info["attempt"] = attempt_label
            if override:
                step.info["adv_estimator"] = override
