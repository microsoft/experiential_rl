import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from examples.erl_hotpot.erl_hotpot_flow import DEFAULT_SOLVER_SYSTEM_PROMPT, ErlHotpotWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine


def resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt and args.system_prompt_file:
        print("Both --system-prompt and --system-prompt-file provided; using --system-prompt.")
    if args.system_prompt:
        return args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_SOLVER_SYSTEM_PROMPT


def build_engine(model: str, base_url: str, api_key: str, max_prompt_length: int, max_response_length: int, temperature: float, top_p: float) -> OpenAIEngine:
    tokenizer = AutoTokenizer.from_pretrained(model)
    return OpenAIEngine(
        model=model,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        base_url=base_url,
        api_key=api_key,
        sampling_params={"temperature": temperature, "top_p": top_p},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERL Hotpot search workflow and print updater state.")

    parser.add_argument("--model", default=os.getenv("ERL_HOTPOT_SOLVER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--base-url", default=os.getenv("ERL_HOTPOT_SOLVER_BASE_URL", "http://localhost:4000/v1"))
    parser.add_argument("--api-key", default=os.getenv("ERL_HOTPOT_SOLVER_API_KEY", ""))
    parser.add_argument("--use-proxy", action="store_true", default=os.getenv("ERL_HOTPOT_SOLVER_USE_PROXY", "true").lower() in {"1", "true", "yes"})
    parser.add_argument("--temperature", type=float, default=float(os.getenv("ERL_HOTPOT_SOLVER_TEMPERATURE", "0.7")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("ERL_HOTPOT_SOLVER_MAX_TOKENS", "2048")))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("ERL_HOTPOT_MAX_TURNS", "5")))

    parser.add_argument("--retriever-url", default=os.getenv("ERL_HOTPOT_RETRIEVER_URL", "http://127.0.0.1:9002"))
    parser.add_argument("--retriever-max-results", type=int, default=int(os.getenv("ERL_HOTPOT_RETRIEVER_MAX_RESULTS", "5")))
    parser.add_argument("--retriever-timeout", type=float, default=float(os.getenv("ERL_HOTPOT_RETRIEVER_TIMEOUT", "30")))

    parser.add_argument("--updater-model", default=os.getenv("ERL_HOTPOT_UPDATER_MODEL", os.getenv("ERL_UPDATER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")))
    parser.add_argument("--updater-base-url", default=os.getenv("ERL_HOTPOT_UPDATER_BASE_URL", os.getenv("ERL_UPDATER_BASE_URL")))
    parser.add_argument("--updater-api-key", default=os.getenv("ERL_HOTPOT_UPDATER_API_KEY", os.getenv("ERL_UPDATER_API_KEY")))
    parser.add_argument("--updater-temperature", type=float, default=float(os.getenv("ERL_HOTPOT_UPDATER_TEMPERATURE", os.getenv("ERL_UPDATER_TEMPERATURE", "0.5"))))
    parser.add_argument("--updater-top-p", type=float, default=float(os.getenv("ERL_HOTPOT_UPDATER_TOP_P", os.getenv("ERL_UPDATER_TOP_P", "0.9"))))
    parser.add_argument("--updater-max-prompt-length", type=int, default=int(os.getenv("ERL_HOTPOT_UPDATER_MAX_PROMPT_LENGTH", os.getenv("ERL_UPDATER_MAX_PROMPT_LENGTH", "4096"))))
    parser.add_argument("--updater-max-response-length", type=int, default=int(os.getenv("ERL_HOTPOT_UPDATER_MAX_RESPONSE_LENGTH", os.getenv("ERL_UPDATER_MAX_RESPONSE_LENGTH", "1024"))))

    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--system-prompt-file", default=None)

    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--dataset", default="hotpotqa-small")
    parser.add_argument("--split", default="test")

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    dataset = DatasetRegistry.load_dataset(args.dataset, args.split)
    examples = dataset.get_data()[: max(1, args.num_examples)]

    updater_base_url = args.updater_base_url or args.base_url
    updater_api_key = args.updater_api_key or args.api_key
    updater_engine = build_engine(
        model=args.updater_model,
        base_url=updater_base_url,
        api_key=updater_api_key,
        max_prompt_length=args.updater_max_prompt_length,
        max_response_length=args.updater_max_response_length,
        temperature=args.updater_temperature,
        top_p=args.updater_top_p,
    )

    system_prompt = resolve_system_prompt(args)

    workflow = ErlHotpotWorkflow(
        rollout_engine=updater_engine,
        executor=ThreadPoolExecutor(max_workers=4),
        initial_system_prompt=system_prompt,
        batch_size=1,
        max_concurrency=8,
        solver_model=args.model,
        solver_base_url=args.base_url,
        solver_api_key=args.api_key,
        solver_use_proxy=args.use_proxy,
        solver_temperature=args.temperature,
        solver_max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        retriever_server_url=args.retriever_url,
        retriever_max_results=args.retriever_max_results,
        retriever_timeout=args.retriever_timeout,
        updater_engine_name="openai",
        updater_rollout_engine=updater_engine,
        updater_sampling_params={"temperature": args.updater_temperature, "top_p": args.updater_top_p},
    )

    for idx, example in enumerate(examples, start=1):
        uid = f"hotpot_demo_{idx}"
        episode = await workflow.run(task=example, uid=uid)
        print("\n" + "=" * 80)
        print(f"Example {idx}: {example.get('question')}")
        print("-" * 80)
        print("Updater state:\n")
        updater_state = episode.info.get("updater_state")
        if updater_state:
            print(updater_state)
        else:
            print("<no updater state>")

        print("\nImproved prompt:\n")
        print(episode.info.get("improved_prompt"))


if __name__ == "__main__":
    asyncio.run(main())
