from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import socket
import threading
import uuid
from collections import deque
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from examples.erl.erl_flow import ErlPromptUpdater
from examples.sdk.langgraph.local_retrieval_tool import LocalRetrievalTool, to_langchain_tool
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn
from rllm.sdk import get_chat_client, get_chat_client_async
from rllm.sdk.session.base import _ensure_tracer_initialized
from rllm.sdk.proxy.proxy_manager import ProxyManager, VerlProxyManager
from rllm.sdk.session import SESSION_BACKEND
from rllm.utils import colorful_print
from rllm.workflows.workflow import Workflow

FeedbackFunction = Callable[[dict[str, Any], dict[str, Any], RewardOutput], str]

DEFAULT_SOLVER_SYSTEM_PROMPT = """You are a helpful assistant who answers questions directly and efficiently.

Provide your final answer in \\boxed{} format."""

DEFAULT_UPDATER_SYSTEM_PROMPT = (
    "You are an expert prompt updater for multi-hop QA with retrieval. "
    "You will analyze recent trajectories, tool calls, and rewards to improve the solver's system prompt. "
    "Your priority is to reliably trigger retrieval when the answer is not directly stated in the question, "
    "form concise queries, and synthesize evidence across sources. "
    "When failures occur, explicitly add rules that prevent repeating them (e.g., missing tool calls, "
    "hallucinated facts, or unboxed final answers). "
    "Keep the prompt short, actionable, and reusable. "
    "Output ONLY the improved system prompt wrapped in <prompt>...</prompt> tags."
)

BOXED_REMINDER = "Always put your final answer in \\boxed{} format."
SECOND_ATTEMPT_GENERIC_INSTRUCTION = (
    "You are provided with the model's past attempt data, including observations, actions, rewards, and feedback. "
    "Use this information as context to make a better next-attempt decision policy. "
    "Follow the action/output format exactly."
)


_PROXY_LOCK = threading.Lock()
_PROXY_MANAGER: ProxyManager | None = None


def _bool_from_env(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _maybe_start_litellm_proxy(
    rollout_engine: RolloutEngine | None,
    model_name: str,
    *,
    proxy_mode: str | None = None,
    proxy_host: str | None = None,
    proxy_port: int | None = None,
    admin_token: str | None = None,
    db_path: str | None = None,
    project: str | None = None,
    snapshot_dir: str | None = None,
) -> ProxyManager | None:
    global _PROXY_MANAGER
    if _PROXY_MANAGER is not None:
        return _PROXY_MANAGER

    if not _bool_from_env(os.getenv("ERL_HOTPOT_PROXY_AUTOSTART", "true"), default=True):
        return None

    host = proxy_host or os.getenv("ERL_HOTPOT_PROXY_HOST", os.getenv("LITELLM_PROXY_HOST", "127.0.0.1"))
    port = proxy_port or int(os.getenv("ERL_HOTPOT_PROXY_PORT", os.getenv("LITELLM_PROXY_PORT", "4000")))
    proxy_mode = proxy_mode or os.getenv("ERL_HOTPOT_PROXY_MODE", os.getenv("LITELLM_PROXY_MODE", "external"))
    admin_token = admin_token or os.getenv("ERL_HOTPOT_PROXY_ADMIN_TOKEN", os.getenv("LITELLM_PROXY_ADMIN_TOKEN", "my-shared-secret"))
    db_path = db_path or os.getenv("ERL_HOTPOT_PROXY_DB_PATH", os.getenv("LITELLM_PROXY_DB_PATH"))
    project = project or os.getenv("ERL_HOTPOT_PROXY_PROJECT", os.getenv("LITELLM_PROXY_PROJECT", "erl-hotpot"))
    snapshot_dir = snapshot_dir or os.getenv("ERL_HOTPOT_PROXY_SNAPSHOT_DIR", os.getenv("LITELLM_PROXY_STATE_DIR"))
    add_logprobs = _bool_from_env(os.getenv("ERL_HOTPOT_PROXY_ADD_LOGPROBS", os.getenv("LITELLM_PROXY_ADD_LOGPROBS", "false")))
    add_return_token_ids = _bool_from_env(
        os.getenv("ERL_HOTPOT_PROXY_ADD_RETURN_TOKEN_IDS", os.getenv("LITELLM_PROXY_ADD_RETURN_TOKEN_IDS", "true")),
        default=True,
    )
    requires_sync_storage = SESSION_BACKEND == "opentelemetry"

    with _PROXY_LOCK:
        if _PROXY_MANAGER is not None:
            return _PROXY_MANAGER
        if rollout_engine is None or type(rollout_engine).__name__ != "VerlEngine":
            print("[erl_hotpot] Proxy autostart skipped; rollout_engine is not VerlEngine.")
            return None

        manager = VerlProxyManager(
            rollout_engine=rollout_engine,
            model_name=model_name,
            proxy_host=host,
            proxy_port=port,
            admin_token=admin_token,
            proxy_access_log=False,
            add_logprobs=add_logprobs,
        )
        config_payload = manager.build_proxy_config()

        if requires_sync_storage and proxy_mode == "external":
            print("[erl_hotpot] OpenTelemetry sessions require sync tracer. Consider proxy_mode='subprocess'.")

        if proxy_mode == "subprocess":
            manager.start_proxy_subprocess(
                config=config_payload,
                db_path=db_path,
                project=project,
                snapshot_directory=snapshot_dir,
                sync_tracer=requires_sync_storage,
                add_logprobs=add_logprobs,
                add_return_token_ids=add_return_token_ids,
            )
        elif proxy_mode == "external":
            manager.reload_proxy_config(config=config_payload)
        else:
            raise ValueError(f"Unknown proxy mode: {proxy_mode}. Must be 'external' or 'subprocess'")

        _PROXY_MANAGER = manager
        return _PROXY_MANAGER


class ErlHotpotStateBuilder:
    """Compose prompt updater context from HotPotQA attempts."""

    def __init__(self, max_examples: int = 8):
        self.max_examples = max_examples

    def __call__(
        self,
        initial_prompt: str,
        batch: Sequence[dict[str, Any]],
        attempts: Sequence[dict[str, Any]],
        metrics: dict[str, float],
    ) -> str:
        lines: list[str] = [
            "## Current System Prompt",
            initial_prompt.strip(),
            "",
            "## Batch Metrics",
        ]

        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.4f}")

        lines.append("")
        lines.append("## Recent Attempts")

        for idx, attempt in enumerate(attempts[-self.max_examples :], start=1):
            question = (attempt.get("question") or "").strip()
            final_response = (attempt.get("first_response") or "").strip() or "<empty response>"
            tool_calls = attempt.get("first_tool_calls", 0)
            trajectory_lines = _format_trajectory_for_state(attempt.get("first_messages", []))

            lines.append(f"### Attempt {idx}")
            lines.append(f"Question:\n{question}")
            lines.append("Trajectory:")
            lines.extend(trajectory_lines or ["<no trajectory recorded>"])
            lines.append(f"Final response:\n{final_response}")
            lines.append(f"Final answer: {attempt.get('first_final_answer')}")
            lines.append(f"Tool calls: {tool_calls} | Turns: {attempt.get('first_num_turns')} | Timed out: {attempt.get('first_timed_out')}")
            lines.append(
                f"Reward: {attempt.get('first_reward', 0.0):.4f} | Correct: {attempt.get('first_correct', False)}"
            )
            feedback = attempt.get("feedback")
            if feedback:
                lines.append(f"Feedback: {feedback}")
            lines.append("")

        lines.append("")
        lines.append("Think step by step and provide an improved system prompt enclosed in <prompt>...</prompt> tags.")
        return "\n".join(lines)


class ErlHotpotSearchAgent:
    """LangGraph-based search agent runner with configurable prompts."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        use_proxy: bool,
        temperature: float,
        max_tokens: int,
        retriever_server_url: str,
        retriever_max_results: int,
        retriever_timeout: float,
    ) -> None:
        self.retriever_tool = to_langchain_tool(
            server_url=retriever_server_url,
            max_results=retriever_max_results,
            timeout=retriever_timeout,
        )
        self.openai_tools = [convert_to_openai_tool(self.retriever_tool)]

        sync_client = get_chat_client(
            api_key=api_key,
            base_url=base_url,
            use_proxy=use_proxy,
        )
        async_client = get_chat_client_async(
            api_key=api_key,
            base_url=base_url,
            use_proxy=use_proxy,
        )

        self.response_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=sync_client,
            async_client=async_client,
        )
        _ensure_tracer_initialized("erl_hotpot")

        self.graph = self._build_graph()

    def _build_graph(self):
        async def agent_step(state: MessagesState):
            response = await self.response_model.bind_tools([self.retriever_tool]).ainvoke(state["messages"])
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_step)
        workflow.add_node("tools", ToolNode([self.retriever_tool]))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    async def run(self, question: str, system_prompt: str, max_turns: int) -> dict[str, Any]:
        trace_messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        raw_agent_outputs: list[str] = []
        num_turns = 0
        timed_out = False
        final_answer = None

        async for chunk in self.graph.astream(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
            },
            {"recursion_limit": max_turns * 2 + 5},
        ):
            for node_name, update in chunk.items():
                if node_name == "agent":
                    num_turns += 1
                    if num_turns > max_turns:
                        timed_out = True
                        break

                if "messages" in update and update["messages"]:
                    # Keep behavior consistent with examples/sdk/langgraph/search_agent_langgraph.py:
                    # consume the latest emitted message from each node update.
                    last_msg = update["messages"][-1]
                    trace_messages.append(last_msg)
                    msg_dict = _langchain_message_to_dict(last_msg)

                    content = msg_dict.get("content") or ""
                    if node_name == "agent":
                        raw_agent_outputs.append(content)
                    match = re.search(r"\\boxed\{([^}]+)\}", content)
                    if match:
                        final_answer = match.group(1)

            if timed_out:
                break

        messages = _safe_convert_to_openai_messages(trace_messages)
        final_response = raw_agent_outputs[-1] if raw_agent_outputs else (messages[-1].get("content") if messages else "")

        return {
            "question": question,
            "final_answer": final_answer,
            "final_response": final_response,
            "messages": messages,
            "raw_agent_outputs": raw_agent_outputs,
            "num_turns": num_turns,
            "timed_out": timed_out,
        }


class ErlHotpotRolloutAgent:
    """RolloutEngine-backed search agent for training (no external HTTP calls)."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        retriever_server_url: str,
        retriever_max_results: int,
        retriever_timeout: float,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.retriever = LocalRetrievalTool(
            server_url=retriever_server_url,
            max_results=retriever_max_results,
            timeout=retriever_timeout,
        )
        self.openai_tools = [self.retriever.json]

    async def run(self, question: str, system_prompt: str, max_turns: int) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        raw_agent_outputs: list[str] = []
        num_turns = 0
        timed_out = False
        final_answer = None

        for _ in range(max_turns):
            num_turns += 1
            output = await self.rollout_engine.get_model_response(messages, tools=[self.retriever.json])
            content = output.content or output.text or ""
            raw_tool_calls = output.tool_calls or []
            tool_calls = _normalize_tool_calls(raw_tool_calls)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            raw_agent_outputs.append(content)

            match = re.search(r"\\boxed\\{([^}]+)\\}", content)
            if match:
                final_answer = match.group(1)

            if not tool_calls:
                break

            for call in tool_calls:
                tool_name = call.get("name") or call.get("function", {}).get("name")
                args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_output = ""
                if tool_name == self.retriever.name:
                    query = args.get("query") or ""
                    top_k = args.get("top_k")
                    result = self.retriever.forward(query=query, top_k=top_k)
                    tool_output = result.output or result.error or ""
                else:
                    tool_output = f"Unknown tool: {tool_name}"

                tool_msg: dict[str, Any] = {"role": "tool", "content": tool_output}
                tool_call_id = call.get("id") or call.get("tool_call_id")
                if tool_call_id:
                    tool_msg["tool_call_id"] = tool_call_id
                messages.append(tool_msg)

        if num_turns >= max_turns and final_answer is None:
            timed_out = True

        final_response = raw_agent_outputs[-1] if raw_agent_outputs else ""
        return {
            "question": question,
            "final_answer": final_answer,
            "final_response": final_response,
            "messages": messages,
            "raw_agent_outputs": raw_agent_outputs,
            "num_turns": num_turns,
            "timed_out": timed_out,
        }


class ErlHotpotWorkflow(Workflow):
    """Reflection-enhanced workflow tailored for HotPotQA search agents."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        initial_system_prompt: str = DEFAULT_SOLVER_SYSTEM_PROMPT,
        feedback_fn: FeedbackFunction | None = None,
        state_builder: ErlHotpotStateBuilder | None = None,
        train_dataset: Sequence[dict[str, Any]] | None = None,
        batch_size: int = 1,
        max_concurrency: int = 32,
        solver_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        solver_base_url: str = "http://localhost:4000/v1",
        solver_api_key: str = "",
        solver_use_proxy: bool = True,
        solver_temperature: float = 0.7,
        solver_max_tokens: int = 2048,
        solver_engine_name: str = "sdk",
        solver_rollout_engine: RolloutEngine | None = None,
        max_turns: int = 5,
        retriever_server_url: str = "http://127.0.0.1:9002",
        retriever_max_results: int = 5,
        retriever_timeout: float = 30.0,
        updater_engine_name: str = "openai",
        updater_rollout_engine: RolloutEngine | None = None,
        updater_rollout_engine_config: dict[str, Any] | None = None,
        updater_sampling_params: dict[str, Any] | None = None,
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
        strict_correctness: bool = True,
        no_memory: bool = False,
        no_reflection: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine, executor, **kwargs)
        self.initial_system_prompt = self._ensure_boxed_reminder(initial_system_prompt.strip())
        self.feedback_fn = feedback_fn or default_feedback_fn
        self.state_builder = state_builder or ErlHotpotStateBuilder()
        self.batch_size = max(1, batch_size)
        self.max_concurrency = max(1, max_concurrency)
        self.max_turns = max(1, int(max_turns))
        self.success_reward_threshold = success_reward_threshold
        self.train_first_attempt = bool(train_first_attempt)
        self.train_second_attempt_raw = bool(train_second_attempt_raw)
        self.train_second_attempt_distilled = bool(train_second_attempt_distilled)
        self.train_updater = bool(train_updater)
        self.train_first_attempt_adv_estimator = train_first_attempt_adv_estimator
        self.train_second_attempt_raw_adv_estimator = train_second_attempt_raw_adv_estimator
        self.train_second_attempt_distilled_adv_estimator = train_second_attempt_distilled_adv_estimator
        self.train_updater_adv_estimator = train_updater_adv_estimator
        self.strict_correctness = bool(strict_correctness)
        self.no_memory = bool(no_memory)
        self.no_reflection = bool(no_reflection)

        self.reward_fn = RewardSearchFn(RewardConfig())

        self.solver_engine_name = (solver_engine_name or "sdk").lower()
        if self.solver_engine_name == "sdk" and solver_use_proxy:
            _maybe_start_litellm_proxy(
                rollout_engine,
                solver_model,
                proxy_mode=kwargs.get("proxy_mode"),
                proxy_host=kwargs.get("proxy_host"),
                proxy_port=kwargs.get("proxy_port"),
                admin_token=kwargs.get("proxy_admin_token"),
                db_path=kwargs.get("proxy_db_path"),
                project=kwargs.get("proxy_project"),
                snapshot_dir=kwargs.get("proxy_snapshot_dir"),
            )
        resolved_solver_engine = solver_rollout_engine or (rollout_engine if self.solver_engine_name == "verl" else None)
        if resolved_solver_engine is not None:
            self.search_agent = ErlHotpotRolloutAgent(
                rollout_engine=resolved_solver_engine,
                retriever_server_url=retriever_server_url,
                retriever_max_results=retriever_max_results,
                retriever_timeout=retriever_timeout,
            )
        else:
            self.search_agent = ErlHotpotSearchAgent(
                model=solver_model,
                base_url=solver_base_url,
                api_key=solver_api_key,
                use_proxy=solver_use_proxy,
                temperature=solver_temperature,
                max_tokens=solver_max_tokens,
                retriever_server_url=retriever_server_url,
                retriever_max_results=retriever_max_results,
                retriever_timeout=retriever_timeout,
            )


        self.updater_engine_name = (updater_engine_name or "openai").lower()
        if self.updater_engine_name == "verl":
            self.updater_rollout_engine = rollout_engine
        elif updater_rollout_engine is not None:
            self.updater_rollout_engine = updater_rollout_engine
        elif updater_rollout_engine_config is not None:
            self.updater_rollout_engine = self._instantiate_rollout_engine(updater_rollout_engine_config)
        else:
            self.updater_rollout_engine = rollout_engine

        self.updater = ErlPromptUpdater(
            self.updater_rollout_engine,
            system_prompt=DEFAULT_UPDATER_SYSTEM_PROMPT,
            sampling_params=updater_sampling_params,
        )

        self.train_dataset = list(train_dataset) if train_dataset is not None else []
        self._train_queue: deque[dict[str, Any]] = deque()
        if self.train_dataset:
            self._reset_train_queue()

        self.last_improved_prompt: str | None = None
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self._solver_tool_schemas = _extract_solver_tool_schemas(self.search_agent)

    async def run(self, task: dict[str, Any], uid: str, **kwargs: Any) -> Episode:
        self.reset(task, uid)

        is_validation = bool(getattr(self.rollout_engine, "validate", False) or kwargs.get("validate", False))
        minibatch = self._extract_minibatch(task, use_training_queue=not is_validation)
        if not minibatch:
            return Episode(id=uid, task=task, is_correct=False, trajectories=[], metrics={}, info={})

        current_prompt = self.initial_system_prompt
        first_results = await self._run_solver_batch(current_prompt, minibatch)

        attempts: list[dict[str, Any]] = []
        first_rewards: list[float] = []
        first_successes = 0

        solver_trajectories: list[Trajectory] = []

        for idx, result in enumerate(first_results):
            example = result["example"]
            question = result["question"]
            ground_truth = self._extract_ground_truth(example)

            response_text = result["final_response"] or ""
            reward_output = self._evaluate_reward(ground_truth, response_text, result["timed_out"])
            feedback = self.feedback_fn(example, result, reward_output)

            attempt_record = {
                "example": example,
                "question": question,
                "first_response": response_text,
                "first_final_answer": result.get("final_answer"),
                "first_reward": float(reward_output.reward),
                "first_correct": bool(reward_output.is_correct),
                "first_num_turns": result.get("num_turns"),
                "first_timed_out": result.get("timed_out"),
                "first_messages": result.get("messages", []),
                "first_raw_agent_outputs": result.get("raw_agent_outputs", []),
                "first_tool_calls": _count_tool_calls(result.get("messages", [])),
                "feedback": feedback,
                "reward_metadata": reward_output.metadata,
            }
            attempts.append(attempt_record)

            first_rewards.append(float(reward_output.reward))
            if reward_output.is_correct:
                first_successes += 1

            if is_validation:
                traj = self._build_trajectory(
                    attempt_label="validation",
                    batch_index=idx,
                    question=question,
                    system_prompt=current_prompt,
                    messages=result.get("messages", []),
                    response_text=response_text,
                    reward_output=reward_output,
                    timed_out=result.get("timed_out"),
                    num_turns=result.get("num_turns"),
                    validation=True,
                    use_messages_as_is=True,
                )
                solver_trajectories.append(traj)
            elif self.train_first_attempt:
                traj = self._build_trajectory(
                    attempt_label="first",
                    batch_index=idx,
                    question=question,
                    system_prompt=current_prompt,
                    messages=result.get("messages", []),
                    response_text=response_text,
                    reward_output=reward_output,
                    timed_out=result.get("timed_out"),
                    num_turns=result.get("num_turns"),
                    use_messages_as_is=True,
                )
                self._tag_adv_estimator(traj, attempt_label="first", override=self.train_first_attempt_adv_estimator)
                solver_trajectories.append(traj)

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
        base_prompt_for_update = self.initial_system_prompt if self.no_memory else (self.last_improved_prompt or self.initial_system_prompt)
        improved_prompt = base_prompt_for_update
        avg_second_reward = avg_first_reward
        second_success_rate = first_success_rate

        if not is_validation:
            if all_first_attempts_correct:
                for attempt_record in attempts:
                    _mirror_first_to_second(attempt_record)
                improved_prompt = base_prompt_for_update
                second_success_rate = first_success_rate
                avg_second_reward = avg_first_reward
                state = None
            elif not (compute_second_metrics or self.train_updater):
                for attempt_record in attempts:
                    _mirror_first_to_second(attempt_record)
                improved_prompt = base_prompt_for_update
                second_success_rate = first_success_rate
                avg_second_reward = avg_first_reward
                state = None
            else:
                state = self.state_builder(base_prompt_for_update, minibatch, attempts, metrics)
                if self.no_reflection:
                    improved_prompt = self._build_generic_second_attempt_prompt(state)
                else:
                    improved_prompt, updater_traj = await self.updater.propose_prompt(state, base_prompt_for_update)
                    improved_prompt = self._ensure_boxed_reminder(improved_prompt)
                    if updater_traj is not None:
                        if updater_traj.steps:
                            for step in updater_traj.steps:
                                step.reward = 0.0
                        self._tag_adv_estimator(updater_traj, attempt_label="updater", override=self.train_updater_adv_estimator)
                        logged_updater_traj = updater_traj
                        if self.train_updater:
                            solver_trajectories.append(updater_traj)

                if not compute_second_metrics:
                    for attempt_record in attempts:
                        _mirror_first_to_second(attempt_record)
                    second_success_rate = first_success_rate
                    avg_second_reward = avg_first_reward
                else:
                    pending_indices = [i for i, record in enumerate(attempts) if not record.get("first_correct", False)]
                    second_results: list[dict[str, Any]] = []
                    if pending_indices:
                        pending_batch = [attempts[i]["example"] for i in pending_indices]
                        second_results = await self._run_solver_batch(improved_prompt, pending_batch)

                    second_rewards: list[float] = []
                    second_successes = 0
                    second_attempt_success = False

                    for attempt_record in attempts:
                        if attempt_record.get("first_correct", False):
                            _mirror_first_to_second(attempt_record)
                            second_rewards.append(float(attempt_record["second_reward"]))
                            second_successes += 1

                    for local_idx, global_idx in enumerate(pending_indices):
                        attempt_record = attempts[global_idx]
                        result = second_results[local_idx]
                        question = attempt_record["question"]
                        ground_truth = self._extract_ground_truth(attempt_record["example"])
                        response_text = result["final_response"] or ""
                        reward_output = self._evaluate_reward(ground_truth, response_text, result["timed_out"])

                        attempt_record.update(
                            {
                                "second_response": response_text,
                                "second_final_answer": result.get("final_answer"),
                                "second_reward": float(reward_output.reward),
                                "second_correct": bool(reward_output.is_correct),
                                "second_num_turns": result.get("num_turns"),
                                "second_timed_out": result.get("timed_out"),
                                "second_messages": result.get("messages", []),
                                "second_raw_agent_outputs": result.get("raw_agent_outputs", []),
                                "second_tool_calls": _count_tool_calls(result.get("messages", [])),
                                "second_reward_metadata": reward_output.metadata,
                            }
                        )

                        second_rewards.append(float(reward_output.reward))
                        if reward_output.is_correct:
                            second_successes += 1
                            second_attempt_success = True

                        second_trajectory = self._build_trajectory(
                            attempt_label="second_raw",
                            batch_index=global_idx,
                            question=question,
                            system_prompt=improved_prompt,
                            messages=result.get("messages", []),
                            response_text=response_text,
                            reward_output=reward_output,
                            timed_out=result.get("timed_out"),
                            num_turns=result.get("num_turns"),
                            use_messages_as_is=True,
                        )

                        if self.train_second_attempt_raw:
                            traj = copy.deepcopy(second_trajectory)
                            self._tag_adv_estimator(traj, attempt_label="second_raw", override=self.train_second_attempt_raw_adv_estimator)
                            traj.name = f"{traj.name}_attempt2_batch{global_idx}"
                            solver_trajectories.append(traj)

                        if self.train_second_attempt_distilled:
                            distilled = copy.deepcopy(second_trajectory)
                            self._replace_trajectory_system_prompt(distilled, current_prompt)
                            self._tag_adv_estimator(
                                distilled,
                                attempt_label="second_distilled",
                                override=self.train_second_attempt_distilled_adv_estimator,
                            )
                            distilled.name = f"{distilled.name}_distilled"
                            solver_trajectories.append(distilled)

                    avg_second_reward = sum(second_rewards) / len(second_rewards) if second_rewards else 0.0
                    second_success_rate = second_successes / len(second_rewards) if second_rewards else 0.0

                if (not self.no_memory) and (not self.no_reflection) and pending_indices and second_attempt_success:
                    self.last_improved_prompt = improved_prompt

        if logged_updater_traj is not None and self.train_updater:
            updater_reward = avg_second_reward if compute_second_metrics else avg_first_reward
            logged_updater_traj.reward = updater_reward
            for step in logged_updater_traj.steps:
                step.reward = updater_reward

        for traj in solver_trajectories:
            self.adjust_step_rewards(traj)
            self.compute_trajectory_reward(traj)

        steps_used = sum(len(traj.steps) for traj in solver_trajectories)
        steps_collected = steps_used
        num_trajectories = len(solver_trajectories)
        all_tool_calls = [record.get("first_tool_calls", 0) for record in attempts]
        avg_tool_calls = sum(all_tool_calls) / len(all_tool_calls) if all_tool_calls else 0.0
        max_tool_calls = max(all_tool_calls, default=0)

        if is_validation:
            metrics_payload = {
                "avg_reward": avg_first_reward,
                "success_rate": first_success_rate,
                "first_success_rate": first_success_rate,
                "steps_collected": float(steps_collected),
                "steps_used": float(steps_used),
                "num_trajectories": float(num_trajectories),
                "avg_tool_calls": float(avg_tool_calls),
                "max_tool_calls": float(max_tool_calls),
                "batch_size": float(len(minibatch)),
            }
        elif compute_second_metrics:
            metrics_payload = {
                "avg_first_reward": avg_first_reward,
                "avg_second_reward": avg_second_reward,
                "first_success_rate": first_success_rate,
                "second_success_rate": second_success_rate,
                "steps_collected": float(steps_collected),
                "steps_used": float(steps_used),
                "num_trajectories": float(num_trajectories),
                "avg_tool_calls": float(avg_tool_calls),
                "max_tool_calls": float(max_tool_calls),
                "batch_size": float(len(minibatch)),
            }
        else:
            metrics_payload = {
                "avg_first_reward": avg_first_reward,
                "first_success_rate": first_success_rate,
                "steps_collected": float(steps_collected),
                "steps_used": float(steps_used),
                "num_trajectories": float(num_trajectories),
                "avg_tool_calls": float(avg_tool_calls),
                "max_tool_calls": float(max_tool_calls),
                "batch_size": float(len(minibatch)),
            }

        success_rate_for_correct = second_success_rate if (not is_validation and compute_second_metrics) else first_success_rate
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
                        "question": record.get("question"),
                        "first_reward": record.get("first_reward"),
                        "first_correct": record.get("first_correct"),
                        "second_reward": record.get("second_reward"),
                        "second_correct": record.get("second_correct"),
                        "feedback": record.get("feedback"),
                    }
                    for record in attempts
                ],
            },
        )

        return episode

    async def _run_solver_batch(self, system_prompt: str, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not batch:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: list[dict[str, Any] | None] = [None] * len(batch)

        async def call(idx: int, example: dict[str, Any]):
            question = self._extract_question(example)
            async with semaphore:
                result = await self.search_agent.run(question, system_prompt, self.max_turns)
            return idx, example, question, result

        tasks = [asyncio.create_task(call(idx, example)) for idx, example in enumerate(batch)]

        for task in asyncio.as_completed(tasks):
            idx, example, question, result = await task
            results[idx] = {
                "example": example,
                "question": question,
                **result,
            }

        return [res for res in results if res is not None]

    def _extract_minibatch(self, task: dict[str, Any], use_training_queue: bool = True) -> list[dict[str, Any]]:
        provided_batch: list[dict[str, Any]] = []
        if isinstance(task, dict) and task.get("batch"):
            provided_batch = list(task["batch"])
        elif task:
            provided_batch = [task]

        if not use_training_queue:
            return provided_batch

        batch: list[dict[str, Any]] = list(provided_batch)
        while len(batch) < self.batch_size:
            supplemental = self._pop_train_example()
            if supplemental is None:
                break
            batch.append(supplemental)

        if not batch:
            batch = provided_batch
        return batch

    def _reset_train_queue(self) -> None:
        shuffled = self.train_dataset.copy()
        import random

        random.shuffle(shuffled)
        self._train_queue = deque(shuffled)

    def _pop_train_example(self) -> dict[str, Any] | None:
        if not self.train_dataset:
            return None
        if not self._train_queue:
            self._reset_train_queue()
            if not self._train_queue:
                return None
        return self._train_queue.popleft()

    @staticmethod
    def _extract_question(task: dict[str, Any]) -> str:
        return task.get("question") or task.get("input") or task.get("prompt") or ""

    @staticmethod
    def _extract_ground_truth(task: dict[str, Any]) -> str | list[str] | None:
        return task.get("ground_truth") or task.get("answer") or task.get("target")

    def _evaluate_reward(self, ground_truth: str | list[str] | None, response_text: str, timed_out: bool) -> RewardOutput:
        if timed_out:
            return RewardOutput(reward=0.0, metadata={"reason": "timeout"}, is_correct=False)
        if ground_truth is None:
            return RewardOutput(reward=0.0, metadata={"reason": "missing_ground_truth"}, is_correct=False)

        reward_input = RewardInput(task_info={"ground_truth": ground_truth}, action=response_text)
        reward_output = self.reward_fn(reward_input)
        if self.strict_correctness:
            reward_output.is_correct = reward_output.reward >= 1.0
        return reward_output

    def _build_trajectory(
        self,
        attempt_label: str,
        batch_index: int,
        question: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        response_text: str,
        reward_output: RewardOutput,
        timed_out: bool,
        num_turns: int | None,
        validation: bool = False,
        use_messages_as_is: bool = False,
    ) -> Trajectory:
        if use_messages_as_is:
            full_messages = _ensure_prefixed_messages(system_prompt, question, messages)
            if response_text:
                full_messages = _ensure_assistant_message(full_messages, response_text)
        else:
            full_messages = _merge_with_prompt(system_prompt, question, messages, response_text)

        if not full_messages:
            full_messages = _distilled_messages(system_prompt, question, response_text)
        full_messages = self._attach_tool_schemas(full_messages)

        if not any(msg.get("role") == "assistant" for msg in full_messages):
            colorful_print(
                f"[erl_hotpot] Dropping attempt '{attempt_label}' (batch_index={batch_index}) due to missing assistant message.",
                fg="yellow",
            )
            return Trajectory(
                uid=str(uuid.uuid4()),
                name="solver_attempt",
                task={"question": question},
                steps=[],
                reward=0.0,
                info={"attempt": attempt_label, "batch_index": batch_index, "warning": "no_assistant_message"},
            )

        steps: list[Step] = []
        cumulative: list[dict[str, Any]] = []
        assistant_msgs: list[dict[str, Any]] = []
        for msg in full_messages:
            cumulative.append(msg)
            if msg.get("role") == "assistant":
                assistant_msgs.append(msg)
                content = (msg.get("content") or "").strip()
                tool_calls = msg.get("tool_calls") or []
                action_text = content
                if not action_text and tool_calls:
                    # Keep tool-only assistant turns aligned with chat parser rendering.
                    action_text = self._render_assistant_response_for_step(msg, tool_calls)
                step_messages = list(cumulative)
                if step_messages and step_messages[0].get("role") != "system":
                    step_messages = _ensure_prefixed_messages(system_prompt, question, step_messages)
                step_messages = self._attach_tool_schemas(step_messages)
                step = Step(
                    chat_completions=step_messages,
                    thought="",
                    action=action_text,
                    model_response=action_text,
                    model_output=None,
                    info={
                        "attempt": attempt_label,
                        "batch_index": batch_index,
                        "question": question,
                        "num_turns": num_turns,
                        "timed_out": timed_out,
                        "reward_metadata": reward_output.metadata,
                        "is_correct": bool(reward_output.is_correct),
                        "validation": validation,
                    },
                )
                step.reward = 0.0
                step.done = False
                steps.append(step)

        if not steps:
            if full_messages and full_messages[0].get("role") != "system":
                full_messages = _ensure_prefixed_messages(system_prompt, question, full_messages)
            full_messages = self._attach_tool_schemas(full_messages)
            full_messages = _ensure_assistant_message(full_messages, response_text)
            step = Step(
                chat_completions=full_messages,
                thought="",
                action=response_text,
                model_response=response_text,
                model_output=None,
                info={
                    "attempt": attempt_label,
                    "batch_index": batch_index,
                    "question": question,
                    "num_turns": num_turns,
                    "timed_out": timed_out,
                    "reward_metadata": reward_output.metadata,
                    "is_correct": bool(reward_output.is_correct),
                    "validation": validation,
                },
            )
            step.reward = float(reward_output.reward)
            step.done = True
            steps = [step]
        else:
            steps[-1].reward = float(reward_output.reward)
            steps[-1].done = True

        traj = Trajectory(
            uid=str(uuid.uuid4()),
            name="solver_attempt",
            task={"question": question},
            steps=steps,
            reward=float(reward_output.reward),
            info={"attempt": attempt_label, "batch_index": batch_index},
        )
        return traj

    def _render_assistant_response_for_step(self, assistant_msg: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str:
        parser = getattr(self.rollout_engine, "chat_parser", None)
        if parser is None:
            return json.dumps(tool_calls, ensure_ascii=True)

        try:
            rendered = parser.parse([assistant_msg], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
            generation_prompt = getattr(parser, "generation_prompt", "")
            if generation_prompt and rendered.startswith(generation_prompt):
                rendered = rendered[len(generation_prompt) :]
            return rendered
        except Exception:
            return json.dumps(tool_calls, ensure_ascii=True)

    def _attach_tool_schemas(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages or not self._solver_tool_schemas:
            return list(messages)

        # Normalize to OpenAI tool schema format to match bind_tools behavior.
        normalized_tools: list[dict[str, Any]] = []
        for tool in self._solver_tool_schemas:
            if isinstance(tool, dict):
                if tool.get("type") == "function" and "function" in tool:
                    normalized_tools.append(tool)
                elif "name" in tool and "parameters" in tool:
                    normalized_tools.append({"type": "function", "function": dict(tool)})
                else:
                    normalized_tools.append(tool)
            else:
                try:
                    normalized_tools.append(convert_to_openai_tool(tool))
                except Exception:
                    # Fall back to raw tool representation if conversion fails.
                    normalized_tools.append(tool)

        updated = [dict(message) for message in messages]
        system_idx = next((idx for idx, msg in enumerate(updated) if msg.get("role") == "system"), None)
        if system_idx is None:
            # Insert a system message placeholder if missing so tools can be attached.
            updated.insert(0, {"role": "system", "content": ""})
            system_idx = 0

        system_msg = dict(updated[system_idx])
        if system_msg.get("functions") is None:
            system_msg["functions"] = json.dumps(normalized_tools, ensure_ascii=True)
            updated[system_idx] = system_msg

        return updated

    def _replace_trajectory_system_prompt(self, trajectory: Trajectory, system_prompt: str) -> None:
        for step in trajectory.steps:
            if step.chat_completions and step.chat_completions[0].get("role") == "system":
                step.chat_completions[0]["content"] = system_prompt

    @staticmethod
    def _ensure_boxed_reminder(prompt: str) -> str:
        base_prompt = (prompt or "").rstrip()
        reminder = BOXED_REMINDER.strip()
        if not base_prompt:
            return reminder
        if "\\boxed" in base_prompt:
            return base_prompt
        return f"{base_prompt}\n\n{reminder}"

    def _build_generic_second_attempt_prompt(self, state: str) -> str:
        improved_prompt = (
            f"{self.initial_system_prompt.strip()}\n\n"
            f"{SECOND_ATTEMPT_GENERIC_INSTRUCTION}\n\n"
            f"{state.strip()}"
        )
        return self._ensure_boxed_reminder(improved_prompt)

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
        if attempt_label:
            trajectory.info.setdefault("attempt", attempt_label)
        if override:
            trajectory.info["adv_estimator"] = override
        for step in trajectory.steps:
            if attempt_label and "attempt" not in step.info:
                step.info["attempt"] = attempt_label
            if override:
                step.info["adv_estimator"] = override


def default_feedback_fn(task: dict[str, Any], attempt: dict[str, Any], reward: RewardOutput) -> str:
    status = "correct" if reward.is_correct else "incorrect"
    if attempt.get("timed_out"):
        status = "timeout"

    extracted = reward.metadata.get("extracted_answer") if reward.metadata else None
    details = reward.metadata.get("evaluation_method") or reward.metadata.get("error") if reward.metadata else None

    parts = [status.upper(), f"reward={reward.reward:.2f}"]
    if extracted:
        parts.append(f"extracted={extracted}")
    if details:
        parts.append(f"details={details}")
    return " | ".join(parts)


def _distilled_messages(system_prompt: str, question: str, response_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]


def _ensure_prefixed_messages(system_prompt: str, question: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if not messages:
        return base
    if messages[0].get("role") == "system":
        return list(messages)
    return base + list(messages)


def _ensure_assistant_message(messages: list[dict[str, Any]], response_text: str) -> list[dict[str, Any]]:
    if any(msg.get("role") == "assistant" for msg in messages):
        return list(messages)
    updated = list(messages)
    updated.append({"role": "assistant", "content": response_text})
    return updated


def _merge_with_prompt(system_prompt: str, question: str, messages: list[dict[str, Any]], response_text: str) -> list[dict[str, Any]]:
    merged = _ensure_prefixed_messages(system_prompt, question, messages)

    if merged and merged[-1].get("role") == "assistant" and (merged[-1].get("content") or "") == response_text:
        return merged

    merged.append({"role": "assistant", "content": response_text})
    return _ensure_assistant_message(merged, response_text)


def _mirror_first_to_second(record: dict[str, Any]) -> None:
    record["second_response"] = record.get("first_response", "")
    record["second_final_answer"] = record.get("first_final_answer")
    record["second_reward"] = record.get("first_reward", 0.0)
    record["second_correct"] = record.get("first_correct", False)
    record["second_num_turns"] = record.get("first_num_turns")
    record["second_timed_out"] = record.get("first_timed_out")
    record["second_messages"] = record.get("first_messages", [])
    record["second_raw_agent_outputs"] = record.get("first_raw_agent_outputs", [])
    record["second_tool_calls"] = record.get("first_tool_calls", 0)
    record["second_reward_metadata"] = record.get("reward_metadata")


def _count_tool_calls(messages: Sequence[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            count += len(tool_calls)
    return count


def _langchain_message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", None) or getattr(message, "type", None) or "assistant"
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"

    content = message.content if hasattr(message, "content") else str(message)
    msg_dict: dict[str, Any] = {"role": role, "content": content}

    name = getattr(message, "name", None)
    if name:
        msg_dict["name"] = name

    tool_calls = getattr(message, "tool_calls", None)
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not tool_calls:
        tool_calls = additional_kwargs.get("tool_calls")
    if tool_calls:
        msg_dict["tool_calls"] = _normalize_tool_calls(tool_calls)

    function_call = additional_kwargs.get("function_call")
    if function_call:
        msg_dict["function_call"] = function_call

    tool_call_id = additional_kwargs.get("tool_call_id")
    if tool_call_id:
        msg_dict["tool_call_id"] = tool_call_id

    return msg_dict


def _safe_convert_to_openai_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    try:
        converted = convert_to_openai_messages(list(messages))
        if isinstance(converted, dict):
            return [converted]
        return list(converted)
    except Exception:
        return [_langchain_message_to_dict(message) for message in messages]


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if hasattr(call, "model_dump"):
            normalized.append(call.model_dump())
        elif hasattr(call, "dict"):
            normalized.append(call.dict())
        elif isinstance(call, dict):
            normalized.append(call)
        else:
            normalized.append({"raw": str(call)})
    return normalized


def _format_trajectory_for_state(messages: Sequence[dict[str, Any]]) -> list[str]:
    if not messages:
        return []

    lines: list[str] = []
    for idx, msg in enumerate(messages, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if content:
            lines.append(f"- Turn {idx} [{role}]: {content}")
        else:
            lines.append(f"- Turn {idx} [{role}]: <no content>")

        tool_calls = msg.get("tool_calls") or []
        for call_idx, call in enumerate(tool_calls, start=1):
            name = call.get("name") or call.get("function", {}).get("name") or "<unknown>"
            args = call.get("arguments") or call.get("function", {}).get("arguments") or call.get("args")
            lines.append(f"  tool_call {call_idx}: name={name} args={args}")

        function_call = msg.get("function_call")
        if function_call:
            name = function_call.get("name") or "<unknown>"
            args = function_call.get("arguments")
            lines.append(f"  function_call: name={name} args={args}")

        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            lines.append(f"  tool_call_id: {tool_call_id}")

    return lines


def _extract_solver_tool_schemas(agent: Any) -> list[dict[str, Any]]:
    openai_tools = getattr(agent, "openai_tools", None)
    if isinstance(openai_tools, list) and openai_tools:
        return [dict(tool) if isinstance(tool, dict) else tool for tool in openai_tools]
    retriever = getattr(agent, "retriever", None)
    if retriever is not None and hasattr(retriever, "json"):
        return [retriever.json]
    return []
