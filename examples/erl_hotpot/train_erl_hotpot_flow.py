import os
from typing import Any

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from examples.erl_hotpot.erl_hotpot_flow import DEFAULT_SOLVER_SYSTEM_PROMPT, ErlHotpotWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def _maybe_get(cfg: DictConfig | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _get_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_initial_prompt(config: DictConfig) -> str:
    workflow_cfg = getattr(config, "workflow", None)
    literal_cfg = _maybe_get(workflow_cfg, "system_prompt", None)
    file_cfg = _maybe_get(workflow_cfg, "system_prompt_file", None)

    literal_env = os.getenv("ERL_HOTPOT_SYSTEM_PROMPT")
    file_env = os.getenv("ERL_HOTPOT_SYSTEM_PROMPT_FILE")

    if literal_cfg:
        return literal_cfg
    if literal_env:
        return literal_env

    prompt_file = file_cfg or file_env
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()

    return DEFAULT_SOLVER_SYSTEM_PROMPT


def build_updater_engine_config(config: DictConfig) -> dict[str, dict[str, object]]:
    workflow_cfg = getattr(config, "workflow", None)
    updater_cfg = getattr(workflow_cfg, "updater", None)

    engine_name = _maybe_get(updater_cfg, "engine_name", os.getenv("ERL_HOTPOT_UPDATER_ENGINE", os.getenv("ERL_UPDATER_ENGINE", "openai")))
    if str(engine_name).lower() == "verl":
        return {}

    default_model = os.getenv("ERL_HOTPOT_UPDATER_MODEL", os.getenv("ERL_UPDATER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    default_base_url = os.getenv("ERL_HOTPOT_UPDATER_BASE_URL", os.getenv("ERL_UPDATER_BASE_URL", "http://localhost:30000/v1"))
    default_api_key = os.getenv("ERL_HOTPOT_UPDATER_API_KEY", os.getenv("ERL_UPDATER_API_KEY", "None"))
    default_max_prompt = int(os.getenv("ERL_HOTPOT_UPDATER_MAX_PROMPT_LENGTH", os.getenv("ERL_UPDATER_MAX_PROMPT_LENGTH", "4096")))
    default_max_response = int(os.getenv("ERL_HOTPOT_UPDATER_MAX_RESPONSE_LENGTH", os.getenv("ERL_UPDATER_MAX_RESPONSE_LENGTH", "1024")))

    model_name = _maybe_get(updater_cfg, "model", default_model)
    base_url = _maybe_get(updater_cfg, "base_url", default_base_url)
    api_key = _maybe_get(updater_cfg, "api_key", default_api_key)
    max_prompt_length = _maybe_get(updater_cfg, "max_prompt_length", default_max_prompt)
    max_response_length = _maybe_get(updater_cfg, "max_response_length", default_max_response)

    temperature_cfg = _maybe_get(updater_cfg, "temperature", None)
    top_p_cfg = _maybe_get(updater_cfg, "top_p", None)
    temperature_env = os.getenv("ERL_HOTPOT_UPDATER_TEMPERATURE", os.getenv("ERL_UPDATER_TEMPERATURE"))
    top_p_env = os.getenv("ERL_HOTPOT_UPDATER_TOP_P", os.getenv("ERL_UPDATER_TOP_P"))

    sampling_params: dict[str, float] = {}
    if temperature_cfg is not None:
        sampling_params["temperature"] = float(temperature_cfg)
    elif temperature_env is not None:
        sampling_params["temperature"] = float(temperature_env)

    if top_p_cfg is not None:
        sampling_params["top_p"] = float(top_p_cfg)
    elif top_p_env is not None:
        sampling_params["top_p"] = float(top_p_env)

    return {
        "module": "rllm.engine.rollout.openai_engine",
        "class": "OpenAIEngine",
        "kwargs": {
            "model": model_name,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "base_url": base_url,
            "api_key": api_key,
            "sampling_params": sampling_params,
        },
        "tokenizer_config": {
            "module": "transformers",
            "class": "AutoTokenizer",
            "kwargs": {"pretrained_model_name_or_path": model_name},
        },
    }


def build_updater_sampling_params(config: DictConfig) -> dict[str, float]:
    workflow_cfg = getattr(config, "workflow", None)
    updater_cfg = getattr(workflow_cfg, "updater", None)

    temperature = os.getenv("ERL_HOTPOT_UPDATER_TEMPERATURE", os.getenv("ERL_UPDATER_TEMPERATURE", "0.5"))
    top_p = os.getenv("ERL_HOTPOT_UPDATER_TOP_P", os.getenv("ERL_UPDATER_TOP_P", "0.9"))

    if updater_cfg is not None:
        if updater_cfg.get("temperature") is not None:
            temperature = updater_cfg.get("temperature")
        if updater_cfg.get("top_p") is not None:
            top_p = updater_cfg.get("top_p")

    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
    }


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig):
    dataset_name = os.getenv("ERL_HOTPOT_DATASET", "hotpotqa")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa-small", "test")

    assert train_dataset is not None, f"Train dataset '{dataset_name}' not found."
    assert val_dataset is not None, "Validation dataset 'hotpotqa-small' not found."

    workflow_cfg = getattr(config, "workflow", None)
    updater_engine_name = os.getenv("ERL_HOTPOT_UPDATER_ENGINE", "openai").lower()
    no_memory = _get_bool(_maybe_get(workflow_cfg, "no_memory", os.getenv("ERL_HOTPOT_NO_MEMORY", "false")), default=False)
    no_reflection = _get_bool(_maybe_get(workflow_cfg, "no_reflection", os.getenv("ERL_HOTPOT_NO_REFLECTION", "false")), default=False)

    initial_prompt = build_initial_prompt(config)

    solver_model = os.getenv("ERL_HOTPOT_SOLVER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    solver_engine_name = os.getenv("ERL_HOTPOT_SOLVER_ENGINE", "sdk").lower()
    solver_base_url = os.getenv("ERL_HOTPOT_SOLVER_BASE_URL", "http://localhost:4000/v1")
    solver_api_key = os.getenv("ERL_HOTPOT_SOLVER_API_KEY", "")
    solver_use_proxy = _get_bool(os.getenv("ERL_HOTPOT_SOLVER_USE_PROXY", "true"), default=True)
    solver_temperature = float(os.getenv("ERL_HOTPOT_SOLVER_TEMPERATURE", "0.7"))
    solver_max_tokens = int(os.getenv("ERL_HOTPOT_SOLVER_MAX_TOKENS", "2048"))
    max_turns = int(os.getenv("ERL_HOTPOT_MAX_TURNS", "5"))

    retriever_server_url = os.getenv("ERL_HOTPOT_RETRIEVER_URL", "http://127.0.0.1:9002")
    retriever_max_results = int(os.getenv("ERL_HOTPOT_RETRIEVER_MAX_RESULTS", "5"))
    retriever_timeout = float(os.getenv("ERL_HOTPOT_RETRIEVER_TIMEOUT", "30"))

    workflow_args = {
        "initial_system_prompt": initial_prompt,
        "updater_engine_name": updater_engine_name,
        "train_dataset": list(train_dataset.get_data()),
        "batch_size": int(os.getenv("ERL_HOTPOT_BATCH_SIZE", "1")),
        "max_concurrency": int(os.getenv("ERL_HOTPOT_MAX_CONCURRENCY", "32")),
        "proxy_mode": getattr(config.rllm.sdk.proxy, "mode", None),
        "proxy_host": getattr(config.rllm.sdk.proxy, "host", None),
        "proxy_port": getattr(config.rllm.sdk.proxy, "port", None),
        "proxy_admin_token": getattr(config.rllm.sdk.proxy, "admin_token", None),
        "proxy_db_path": getattr(config.rllm.sdk.store, "path", None),
        "proxy_project": getattr(config.trainer, "project_name", "erl-hotpot"),
        "proxy_snapshot_dir": os.getenv("LITELLM_PROXY_STATE_DIR"),
        "solver_engine_name": solver_engine_name,
        "solver_model": solver_model,
        "solver_base_url": solver_base_url,
        "solver_api_key": solver_api_key,
        "solver_use_proxy": solver_use_proxy,
        "solver_temperature": solver_temperature,
        "solver_max_tokens": solver_max_tokens,
        "max_turns": max_turns,
        "retriever_server_url": retriever_server_url,
        "retriever_max_results": retriever_max_results,
        "retriever_timeout": retriever_timeout,
        "max_prompt_length": int(os.getenv("ERL_HOTPOT_MAX_PROMPT_LENGTH", "4096")),
        "max_response_length": int(os.getenv("ERL_HOTPOT_MAX_RESPONSE_LENGTH", "16384")),
        "success_reward_threshold": float(os.getenv("ERL_HOTPOT_SUCCESS_REWARD_THRESHOLD", "1.0")),
        "train_first_attempt": _get_bool(os.getenv("ERL_HOTPOT_TRAIN_FIRST_ATTEMPT", "true"), default=True),
        "train_second_attempt_raw": _get_bool(os.getenv("ERL_HOTPOT_TRAIN_SECOND_ATTEMPT_RAW", "false")),
        "train_second_attempt_distilled": _get_bool(os.getenv("ERL_HOTPOT_TRAIN_SECOND_ATTEMPT_DISTILLED", "true"), default=True),
        "train_updater": _get_bool(os.getenv("ERL_HOTPOT_TRAIN_UPDATER", "false")),
        "strict_correctness": _get_bool(os.getenv("ERL_HOTPOT_STRICT_CORRECTNESS", "true"), default=True),
        "no_memory": no_memory,
        "no_reflection": no_reflection,
    }

    if updater_engine_name != "verl":
        workflow_args["updater_rollout_engine_config"] = build_updater_engine_config(config)
        workflow_args["updater_sampling_params"] = build_updater_sampling_params(config)

    trainer = AgentTrainer(
        workflow_class=ErlHotpotWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
