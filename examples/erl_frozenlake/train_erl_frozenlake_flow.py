import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from examples.erl_frozenlake.erl_frozenlake_agent import DEFAULT_SYMBOLS, ErlFrozenLakeAgent, build_system_prompt
from examples.erl_frozenlake.erl_frozenlake_env import ErlFrozenLakeEnv
from examples.erl_frozenlake.erl_frozenlake_flow import ErlFrozenLakeWorkflow
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


def build_symbol_map(config: DictConfig) -> dict[str, str]:
    resolved = DEFAULT_SYMBOLS.copy()

    workflow_cfg = getattr(config, "workflow", None)
    cfg_map: dict[str, Any] = {}
    if workflow_cfg is not None and hasattr(workflow_cfg, "symbol_map"):
        cfg_map = OmegaConf.to_container(workflow_cfg.symbol_map, resolve=True) or {}

    env_overrides = {
        "player": os.getenv("ERL_FROZENLAKE_SYMBOL_PLAYER"),
        "goal": os.getenv("ERL_FROZENLAKE_SYMBOL_GOAL"),
        "hole": os.getenv("ERL_FROZENLAKE_SYMBOL_HOLE"),
        "frozen": os.getenv("ERL_FROZENLAKE_SYMBOL_FROZEN"),
        "player_hole": os.getenv("ERL_FROZENLAKE_SYMBOL_PLAYER_HOLE"),
        "player_goal": os.getenv("ERL_FROZENLAKE_SYMBOL_PLAYER_GOAL"),
    }

    for source in (cfg_map, env_overrides):
        for key, value in (source or {}).items():
            if value is not None:
                resolved[key] = str(value)
    return resolved


def build_initial_prompt(config: DictConfig, symbol_map: dict[str, str]) -> str:
    workflow_cfg = getattr(config, "workflow", None)
    literal_cfg = _maybe_get(workflow_cfg, "system_prompt", None)
    file_cfg = _maybe_get(workflow_cfg, "system_prompt_file", None)

    literal_env = os.getenv("ERL_FROZENLAKE_SYSTEM_PROMPT")
    file_env = os.getenv("ERL_FROZENLAKE_SYSTEM_PROMPT_FILE")

    if literal_cfg:
        return literal_cfg
    if literal_env:
        return literal_env

    prompt_file = file_cfg or file_env
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()

    return build_system_prompt(symbol_map=symbol_map)


def build_updater_engine_config(config: DictConfig) -> dict[str, dict[str, object]]:
    workflow_cfg = getattr(config, "workflow", None)
    updater_cfg = getattr(workflow_cfg, "updater", None)

    engine_name = _maybe_get(updater_cfg, "engine_name", os.getenv("ERL_FROZENLAKE_UPDATER_ENGINE", os.getenv("ERL_UPDATER_ENGINE", "openai")))
    if str(engine_name).lower() == "verl":
        return {}

    default_model = os.getenv("ERL_FROZENLAKE_UPDATER_MODEL", os.getenv("ERL_UPDATER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    default_base_url = os.getenv("ERL_FROZENLAKE_UPDATER_BASE_URL", os.getenv("ERL_UPDATER_BASE_URL", "http://localhost:30000/v1"))
    default_api_key = os.getenv("ERL_FROZENLAKE_UPDATER_API_KEY", os.getenv("ERL_UPDATER_API_KEY", "None"))
    default_max_prompt = int(os.getenv("ERL_FROZENLAKE_UPDATER_MAX_PROMPT_LENGTH", os.getenv("ERL_UPDATER_MAX_PROMPT_LENGTH", "4096")))
    default_max_response = int(os.getenv("ERL_FROZENLAKE_UPDATER_MAX_RESPONSE_LENGTH", os.getenv("ERL_UPDATER_MAX_RESPONSE_LENGTH", "1024")))

    model_name = _maybe_get(updater_cfg, "model", default_model)
    base_url = _maybe_get(updater_cfg, "base_url", default_base_url)
    api_key = _maybe_get(updater_cfg, "api_key", default_api_key)
    max_prompt_length = _maybe_get(updater_cfg, "max_prompt_length", default_max_prompt)
    max_response_length = _maybe_get(updater_cfg, "max_response_length", default_max_response)

    temperature_cfg = _maybe_get(updater_cfg, "temperature", None)
    top_p_cfg = _maybe_get(updater_cfg, "top_p", None)
    temperature_env = os.getenv("ERL_FROZENLAKE_UPDATER_TEMPERATURE", os.getenv("ERL_UPDATER_TEMPERATURE"))
    top_p_env = os.getenv("ERL_FROZENLAKE_UPDATER_TOP_P", os.getenv("ERL_UPDATER_TOP_P"))

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

    temperature = os.getenv("ERL_FROZENLAKE_UPDATER_TEMPERATURE", os.getenv("ERL_UPDATER_TEMPERATURE", "0.5"))
    top_p = os.getenv("ERL_FROZENLAKE_UPDATER_TOP_P", os.getenv("ERL_UPDATER_TOP_P", "0.9"))

    return {
        "temperature": float(_maybe_get(updater_cfg, "temperature", temperature)),
        "top_p": float(_maybe_get(updater_cfg, "top_p", top_p)),
    }


def build_solver_settings(config: DictConfig):
    workflow_cfg = getattr(config, "workflow", None)
    solver_cfg = getattr(workflow_cfg, "solver", None)

    default_model = os.getenv("ERL_FROZENLAKE_SOLVER_MODEL", os.getenv("ERL_SOLVER_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    default_base_url = os.getenv("ERL_FROZENLAKE_SOLVER_BASE_URL", os.getenv("ERL_SOLVER_BASE_URL", "http://localhost:30000/v1"))
    default_api_key = os.getenv("ERL_FROZENLAKE_SOLVER_API_KEY", os.getenv("ERL_SOLVER_API_KEY", "None"))
    default_temperature = os.getenv("ERL_FROZENLAKE_SOLVER_TEMPERATURE", os.getenv("ERL_SOLVER_TEMPERATURE", "0.6"))
    default_top_p = os.getenv("ERL_FROZENLAKE_SOLVER_TOP_P", os.getenv("ERL_SOLVER_TOP_P", "0.95"))

    model_name = _maybe_get(solver_cfg, "model", default_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {
        "model": model_name,
        "temperature": float(_maybe_get(solver_cfg, "temperature", default_temperature)),
        "top_p": float(_maybe_get(solver_cfg, "top_p", default_top_p)),
    }

    rollout_args = {
        "base_url": _maybe_get(solver_cfg, "base_url", default_base_url),
        "api_key": _maybe_get(solver_cfg, "api_key", default_api_key),
    }

    default_max_prompt = int(os.getenv("ERL_FROZENLAKE_SOLVER_MAX_PROMPT_LENGTH", "4096"))
    default_max_response = int(os.getenv("ERL_FROZENLAKE_SOLVER_MAX_RESPONSE_LENGTH", "16384"))

    max_prompt_length = _maybe_get(solver_cfg, "max_prompt_length", default_max_prompt)
    max_response_length = _maybe_get(solver_cfg, "max_response_length", default_max_response)

    return tokenizer, sampling_params, rollout_args, max_prompt_length, max_response_length


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    val_dataset = DatasetRegistry.load_dataset("frozenlake", "test")

    symbol_map = build_symbol_map(config)
    initial_prompt = build_initial_prompt(config, symbol_map)

    workflow_cfg = getattr(config, "workflow", None)
    default_agent_steps = int(os.getenv("ERL_FROZENLAKE_AGENT_MAX_STEPS", "8"))
    default_env_steps = int(os.getenv("ERL_FROZENLAKE_ENV_MAX_STEPS", str(default_agent_steps)))
    default_is_slippery = os.getenv("ERL_FROZENLAKE_IS_SLIPPERY", "false").lower() in {"1", "true", "yes"}
    default_success_threshold = float(os.getenv("ERL_FROZENLAKE_SUCCESS_REWARD_THRESHOLD", "1.0"))
    solver_engine_name = _maybe_get(workflow_cfg, "solver_engine_name", os.getenv("ERL_FROZENLAKE_SOLVER_ENGINE", "workflow"))
    updater_engine_name = _maybe_get(workflow_cfg, "updater_engine_name", os.getenv("ERL_FROZENLAKE_UPDATER_ENGINE", os.getenv("ERL_UPDATER_ENGINE", "openai")))

    solver_tokenizer = None
    solver_sampling_params: dict[str, Any] = {}
    solver_rollout_args: dict[str, Any] = {}
    solver_max_prompt = int(os.getenv("ERL_FROZENLAKE_SOLVER_MAX_PROMPT_LENGTH", "4096"))
    solver_max_response = int(os.getenv("ERL_FROZENLAKE_SOLVER_MAX_RESPONSE_LENGTH", "16384"))

    if solver_engine_name != "verl":
        solver_tokenizer, solver_sampling_params, solver_rollout_args, solver_max_prompt, solver_max_response = build_solver_settings(config)

    updater_engine_config = None if str(updater_engine_name).lower() == "verl" else build_updater_engine_config(config)

    workflow_args = {
        "initial_system_prompt": initial_prompt,
        "updater_engine_name": updater_engine_name,
        "updater_rollout_engine_config": updater_engine_config,
        "updater_sampling_params": build_updater_sampling_params(config),
        "train_dataset": list(train_dataset.get_data()),
        "batch_size": int(os.getenv("ERL_FROZENLAKE_BATCH_SIZE", os.getenv("ERL_BATCH_SIZE", "1"))),
        "max_concurrency": int(os.getenv("ERL_FROZENLAKE_MAX_CONCURRENCY", os.getenv("ERL_MAX_CONCURRENCY", "4"))),
        "solver_tokenizer": solver_tokenizer,
        "solver_sampling_params": solver_sampling_params,
        "solver_rollout_engine_args": solver_rollout_args,
        "max_prompt_length": solver_max_prompt,
        "max_response_length": solver_max_response,
        "log_reflection_trajectory": bool(_maybe_get(workflow_cfg, "log_reflection_trajectory", False)),
        "train_first_attempt": bool(_maybe_get(workflow_cfg, "train_first_attempt", False)),
        "train_first_attempt_adv_estimator": _maybe_get(workflow_cfg, "train_first_attempt_adv_estimator", None),
        "train_second_attempt_raw": bool(_maybe_get(workflow_cfg, "train_second_attempt_raw", False)),
        "train_second_attempt_raw_adv_estimator": _maybe_get(workflow_cfg, "train_second_attempt_raw_adv_estimator", None),
        "train_second_attempt_distilled": bool(_maybe_get(workflow_cfg, "train_second_attempt_distilled", True)),
        "train_second_attempt_distilled_adv_estimator": _maybe_get(workflow_cfg, "train_second_attempt_distilled_adv_estimator", None),
        "train_updater": bool(_maybe_get(workflow_cfg, "train_updater", False)),
        "train_updater_adv_estimator": _maybe_get(workflow_cfg, "train_updater_adv_estimator", None),
        "no_memory": _get_bool(_maybe_get(workflow_cfg, "no_memory", os.getenv("ERL_FROZENLAKE_NO_MEMORY", "false"))),
        "no_reflection": _get_bool(_maybe_get(workflow_cfg, "no_reflection", os.getenv("ERL_FROZENLAKE_NO_REFLECTION", "false"))),
        "agent_args": {
            "max_steps": _maybe_get(workflow_cfg, "agent_max_steps", default_agent_steps),
            "use_accumulate_history": True,
            "symbol_map": symbol_map,
            "system_prompt": initial_prompt,
        },
        "env_args": {
            "max_steps": _maybe_get(workflow_cfg, "env_max_steps", default_env_steps),
            "is_slippery": _maybe_get(workflow_cfg, "env_is_slippery", default_is_slippery),
            "symbol_map": symbol_map,
        },
        "success_reward_threshold": _maybe_get(workflow_cfg, "success_reward_threshold", default_success_threshold),
        "solver_engine_name": solver_engine_name,
        "solver_tokenizer": solver_tokenizer,
        "solver_sampling_params": solver_sampling_params,
        "solver_rollout_engine_args": solver_rollout_args,
        "workflow_config": config,
    }

    trainer = AgentTrainer(
        workflow_class=ErlFrozenLakeWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
