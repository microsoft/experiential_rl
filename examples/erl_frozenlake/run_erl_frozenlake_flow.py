import argparse
import asyncio
import json
import os
from copy import deepcopy
from datetime import datetime
from math import ceil

from transformers import AutoTokenizer

from examples.erl_frozenlake.erl_frozenlake_agent import DEFAULT_SYMBOLS as DEFAULT_SYMBOL_MAP, build_system_prompt
from examples.erl_frozenlake.erl_frozenlake_flow import ErlFrozenLakeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine


def load_tasks(train_examples: list[dict], num_tasks: int, batch_size: int, repeats: int = 1) -> list[dict]:
    """Prepare batched FrozenLake tasks for workflow execution."""
    slice_size = min(num_tasks, len(train_examples))
    selected = [train_examples[idx] for idx in range(slice_size)]

    expanded: list[dict] = []
    for example in selected:
        for _ in range(max(1, repeats)):
            expanded.append(deepcopy(example))

    batch_size = max(1, batch_size)
    batches: list[dict] = []
    for batch_index in range(ceil(len(expanded) / batch_size) if expanded else 0):
        start = batch_index * batch_size
        end = start + batch_size
        batch_examples = expanded[start:end]
        if not batch_examples:
            continue
        batches.append({"batch": batch_examples, "batch_index": batch_index})
    return batches


def build_symbol_map_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "player": args.symbol_player,
        "goal": args.symbol_goal,
        "hole": args.symbol_hole,
        "frozen": args.symbol_frozen,
        "player_hole": args.symbol_player_hole,
        "player_goal": args.symbol_player_goal,
    }


def resolve_system_prompt(args: argparse.Namespace, symbol_map: dict[str, str]) -> str:
    if args.system_prompt and args.system_prompt_file:
        print("Both --system-prompt and --system-prompt-file were provided; using --system-prompt.")
    if args.system_prompt:
        return args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return build_system_prompt(symbol_map=symbol_map)


def summarize_results(episodes: list, output_dir: str | None = None) -> None:
    """Summarize FrozenLake ERL workflow outcomes and optionally persist logs."""
    if not episodes:
        print("No episodes generated.")
        return

    first_rewards = [episode.metrics.get("avg_first_reward", episode.metrics.get("avg_reward", 0.0)) for episode in episodes]
    second_rewards = [episode.metrics.get("avg_second_reward", episode.metrics.get("avg_reward", 0.0)) for episode in episodes]
    second_success = [episode.metrics.get("second_success_rate", episode.metrics.get("success_rate", 0.0)) for episode in episodes]

    print("=== FrozenLake ERL Summary ===")
    print(f"Episodes processed: {len(episodes)}")
    print(f"Average first-attempt reward: {sum(first_rewards) / len(first_rewards):.4f}")
    print(f"Average second-attempt reward: {sum(second_rewards) / len(second_rewards):.4f}")
    print(f"Second-attempt success rate: {sum(second_success) / len(second_success):.4%}")

    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(output_dir, f"erl_frozenlake_results_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([episode.to_dict() for episode in episodes], f, indent=2)
    print(f"\nEpisode logs saved to {results_path}")


async def run_erl_frozenlake(args: argparse.Namespace) -> None:
    updater_model = args.updater_model or args.model
    updater_tokenizer = AutoTokenizer.from_pretrained(updater_model)
    updater_engine = OpenAIEngine(
        model=updater_model,
        tokenizer=updater_tokenizer,
        max_prompt_length=args.updater_max_prompt_length,
        max_response_length=args.updater_max_response_length,
        base_url=args.updater_base_url or args.base_url,
        api_key=args.updater_api_key or args.api_key,
        sampling_params={"temperature": args.updater_temperature, "top_p": args.updater_top_p},
    )

    solver_model = args.solver_model or args.model
    solver_tokenizer = AutoTokenizer.from_pretrained(solver_model)
    solver_engine = OpenAIEngine(
        model=solver_model,
        tokenizer=solver_tokenizer,
        max_prompt_length=args.solver_max_prompt_length,
        max_response_length=args.solver_max_response_length,
        base_url=args.solver_base_url or args.base_url,
        api_key=args.solver_api_key or args.api_key,
        sampling_params={"temperature": args.solver_temperature, "top_p": args.solver_top_p},
    )

    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    train_examples = train_dataset.get_data()
    if args.train_limit is not None:
        train_examples = train_examples[: args.train_limit]

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    symbol_map = build_symbol_map_from_args(args)
    system_prompt = resolve_system_prompt(args, symbol_map)

    workflow_args = {
        "initial_system_prompt": system_prompt,
        "train_dataset": list(train_examples),
        "batch_size": args.batch_size,
        "max_concurrency": args.max_concurrency,
        "agent_args": {
            "max_steps": args.max_steps,
            "use_accumulate_history": True,
            "symbol_map": symbol_map,
            "system_prompt": system_prompt,
        },
        "env_args": {
            "max_steps": args.max_steps,
            "is_slippery": args.is_slippery,
            "symbol_map": symbol_map,
        },
        "solver_tokenizer": solver_tokenizer,
        "solver_sampling_params": {
            "model": solver_model,
            "temperature": args.solver_temperature,
            "top_p": args.solver_top_p,
        },
        "solver_rollout_engine_args": {
            "base_url": args.solver_base_url or args.base_url,
            "api_key": args.solver_api_key or args.api_key,
        },
        "updater_rollout_engine": updater_engine,
        "updater_sampling_params": {
            "temperature": args.updater_temperature,
            "top_p": args.updater_top_p,
        },
        "max_response_length": args.solver_max_response_length,
        "max_prompt_length": args.solver_max_prompt_length,
        "success_reward_threshold": args.success_reward_threshold,
    }

    engine = AgentWorkflowEngine(
        workflow_cls=ErlFrozenLakeWorkflow,
        workflow_args=workflow_args,
        rollout_engine=solver_engine,
        config=None,
        n_parallel_tasks=1,
        retry_limit=3,
    )

    tasks = load_tasks(train_examples, num_tasks=args.num_tasks, batch_size=args.batch_size, repeats=args.repeats)
    print(f"Prepared {len(tasks)} FrozenLake batches (batch size={args.batch_size})")

    episodes = await engine.execute_tasks(tasks)
    summarize_results(episodes, output_dir=args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ERL workflow on FrozenLake tasks.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Default model identifier for both solver and updater")
    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1", help="Default inference server base URL")
    parser.add_argument("--api-key", type=str, default="None", help="Default API key for inference servers")

    parser.add_argument("--solver-model", type=str, default=None, help="Override model identifier for the solver")
    parser.add_argument("--solver-base-url", type=str, default=None, help="Override base URL for the solver endpoint")
    parser.add_argument("--solver-api-key", type=str, default=None, help="Override API key for the solver endpoint")
    parser.add_argument("--solver-temperature", type=float, default=0.7, help="Sampling temperature for solver calls")
    parser.add_argument("--solver-top-p", type=float, default=0.8, help="Top-p nucleus for solver calls")
    parser.add_argument("--solver-max-prompt-length", type=int, default=4096, help="Maximum solver prompt length")
    parser.add_argument("--solver-max-response-length", type=int, default=16384, help="Maximum solver completion length")

    parser.add_argument("--system-prompt", type=str, default=None, help="Override solver system prompt with a literal string")
    parser.add_argument("--system-prompt-file", type=str, default=None, help="Path to a text file containing a custom solver system prompt")
    parser.add_argument("--symbol-player", type=str, default=DEFAULT_SYMBOL_MAP["player"], help="Symbol representing the player")
    parser.add_argument("--symbol-goal", type=str, default=DEFAULT_SYMBOL_MAP["goal"], help="Symbol representing the goal")
    parser.add_argument("--symbol-hole", type=str, default=DEFAULT_SYMBOL_MAP["hole"], help="Symbol representing a hole")
    parser.add_argument("--symbol-frozen", type=str, default=DEFAULT_SYMBOL_MAP["frozen"], help="Symbol representing a frozen tile")
    parser.add_argument("--symbol-player-hole", type=str, default=DEFAULT_SYMBOL_MAP["player_hole"], help="Symbol representing the player after falling in a hole")
    parser.add_argument("--symbol-player-goal", type=str, default=DEFAULT_SYMBOL_MAP["player_goal"], help="Symbol representing the player on the goal")

    parser.add_argument("--updater-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Override model identifier for the fixed updater")
    parser.add_argument("--updater-base-url", type=str, default="http://localhost:30001/v1", help="Override base URL for the updater endpoint")
    parser.add_argument("--updater-api-key", type=str, default=None, help="Override API key for the updater endpoint")
    parser.add_argument("--updater-temperature", type=float, default=0.7, help="Sampling temperature for updater calls")
    parser.add_argument("--updater-top-p", type=float, default=0.8, help="Top-p nucleus for updater calls")
    parser.add_argument("--updater-max-prompt-length", type=int, default=16384, help="Maximum updater prompt length")
    parser.add_argument("--updater-max-response-length", type=int, default=4096, help="Maximum updater completion length")

    parser.add_argument("--num-tasks", type=int, default=5, help="Number of unique training tasks to include before batching")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat each task")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tasks per ERL batch")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument("--max-concurrency", type=int, default=128, help="Maximum simultaneous solver rollouts")
    parser.add_argument("--output-dir", type=str, default="logs", help="Directory to store results")

    parser.add_argument("--max-steps", type=int, default=8, help="Maximum environment steps per FrozenLake episode")
    parser.add_argument("--is-slippery", action="store_true", help="Enable stochastic FrozenLake transitions")
    parser.add_argument("--success-reward-threshold", type=float, default=1.0, help="Reward threshold that counts as success")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    cli_args = parse_args()
    asyncio.run(run_erl_frozenlake(cli_args))
