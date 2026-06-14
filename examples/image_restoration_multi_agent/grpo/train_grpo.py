#!/usr/bin/env python3
"""Train one restoration expert LoRA with trajectory-level Agent Lightning GRPO.

Example:

    python examples/image_restoration_multi_agent/grpo/train_grpo.py \
      --config examples/image_restoration_multi_agent/grpo/configs/fog.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml
from transformers import AutoTokenizer

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
NO_THINKING_CHAT_TEMPLATE = EXAMPLE_DIR / "grpo/templates/glm4v_no_thinking.jinja"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from config import load_stage_g_example_config  # noqa: E402
from factory import RealControllerFactory  # noqa: E402
from grpo.agent import GRPOImageRestorationAgent  # noqa: E402
from grpo.smoke_runtime import SmokeControllerFactory  # noqa: E402
from schemas import ExpertName, GRPORestorationTask  # noqa: E402
from tool_registry import ToolRegistry  # noqa: E402

import agentlightning as agl  # noqa: E402


def _resolve(config_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def _load_run_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"GRPO run config must be a mapping: {path}")
    for key in ("stage_config", "model_path", "adapter_path", "train_data", "val_data", "output_dir"):
        payload[key] = _resolve(path, str(payload[key]))
    swanlab = payload.get("swanlab")
    if not isinstance(swanlab, dict):
        raise ValueError(f"GRPO run config must contain a swanlab mapping: {path}")
    required_swanlab_keys = {
        "enabled",
        "project_name",
        "experiment_name",
        "mode",
        "smoke_mode",
        "log_dir",
    }
    missing_swanlab_keys = required_swanlab_keys.difference(swanlab)
    if missing_swanlab_keys:
        missing = ", ".join(sorted(missing_swanlab_keys))
        raise ValueError(f"GRPO swanlab config is missing keys: {missing}")
    valid_modes = {"online", "local", "offline", "disabled"}
    for key in ("mode", "smoke_mode"):
        if swanlab[key] not in valid_modes:
            choices = ", ".join(sorted(valid_modes))
            raise ValueError(f"swanlab.{key} must be one of {choices}, got {swanlab[key]!r}")
    swanlab["log_dir"] = _resolve(path, str(swanlab["log_dir"]))
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"GRPO run config must contain a training mapping: {path}")
    rollout_concurrency = int(training["train_batch_size"]) * int(training["rollout_n"])
    if int(training["n_runners"]) != rollout_concurrency:
        raise ValueError(
            "training.n_runners must equal training.train_batch_size * training.rollout_n "
            f"to keep the complete GRPO rollout batch in flight; expected {rollout_concurrency}, "
            f"got {training['n_runners']}"
        )
    return cast(dict[str, Any], payload)


def _load_dataset(path: str, expert_name: ExpertName, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            task = GRPORestorationTask.model_validate(json.loads(line))
            if task.degradation_type.value != expert_name.value.removesuffix("_expert"):
                raise ValueError(f"{path} contains {task.degradation_type.value} data for {expert_name.value}")
            records.append(task.model_dump(mode="json"))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"empty GRPO dataset: {path}")
    return records


def _verl_config(run: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    training = cast(dict[str, Any], run["training"])
    swanlab = cast(dict[str, Any], run["swanlab"])
    rollout_n = 2 if smoke else int(training["rollout_n"])
    train_batch_size = 1 if smoke else int(training["train_batch_size"])
    max_response_length = 128 if smoke else int(training["max_response_length"])
    ppo_mini_batch_size = 2 if smoke else int(training["ppo_mini_batch_size"])
    output_dir = str(Path(run["output_dir"]) / ("smoke" if smoke else "full"))
    experiment_name = str(swanlab["experiment_name"]) + ("-smoke" if smoke else "")
    loggers = ["console"]
    if bool(swanlab["enabled"]):
        loggers.append("swanlab")
    chat_template = NO_THINKING_CHAT_TEMPLATE.read_text(encoding="utf-8")
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": bool(training["use_kl_in_reward"]),
        },
        "agentlightning": {
            "trace_aggregator": {
                # Keep each turn as an independent visual observation so the
                # actor encodes only the latest image. The trainer still
                # deduplicates rewards by rollout and computes GRPO at the
                # complete-trajectory level.
                "level": "trajectory_transition",
                "trajectory_max_prompt_length": int(training["trajectory_max_prompt_length"]),
                "trajectory_max_response_length": int(training["trajectory_max_response_length"]),
                "debug": False,
            }
        },
        "data": {
            "train_batch_size": train_batch_size,
            "max_prompt_length": int(training["max_prompt_length"]),
            "max_response_length": max_response_length,
            "filter_overlong_prompts": False,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "model": {
                "path": run["model_path"],
                "lora_rank": int(training["lora_rank"]),
                "lora_alpha": int(training["lora_alpha"]),
                "target_modules": str(training["lora_target_modules"]),
                "lora_adapter_path": run["adapter_path"],
                # Expert SFT and its evaluation use LlamaFactory's
                # enable_thinking=false prefix. Apply the same template to both
                # rollout generation and actor/ref retokenization.
                "custom_chat_template": chat_template,
                # GLM-4.1V uses 3-axis M-RoPE. Keeping the 2D attention mask
                # avoids Transformers treating those axes as packed batches.
                "use_remove_padding": bool(training["use_remove_padding"]),
                "enable_gradient_checkpointing": bool(training["enable_gradient_checkpointing"]),
                "enable_activation_offload": bool(training["enable_activation_offload"]),
            },
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": int(training["tensor_model_parallel_size"]),
                "n": rollout_n,
                # GRPO requires stochastic candidates even in the one-step
                # smoke run. Greedy decoding can make all group members
                # identical and produced non-finite policy statistics on GLM.
                "temperature": float(training["temperature"]),
                "top_k": int(training["top_k"]),
                "top_p": float(training["top_p"]),
                "do_sample": bool(training["do_sample"]),
                "ignore_eos": bool(training["ignore_eos"]),
                "log_prob_micro_batch_size_per_gpu": int(training["rollout_log_prob_micro_batch_size_per_gpu"]),
                "log_prob_use_dynamic_bsz": bool(training["rollout_log_prob_use_dynamic_bsz"]),
                "log_prob_max_token_len_per_gpu": int(training["rollout_log_prob_max_token_len_per_gpu"]),
                "gpu_memory_utilization": float(training["gpu_memory_utilization"]),
                "max_num_batched_tokens": int(training["max_num_batched_tokens"]),
                "max_num_seqs": int(training["max_num_seqs"]),
                # Preload the immutable base model in vLLM, then synchronize only
                # LoRA tensors. GLM-4.1V's vLLM loader does not accept PEFT
                # ``base_layer`` names from full-model hybrid weight sync.
                "load_format": "safetensors",
                # Keep the preloaded base model resident during trainer/rollout
                # switches and synchronize only the trainable LoRA tensors.
                "layered_summon": bool(training["layered_summon"]),
                "enable_chunked_prefill": bool(training["enable_chunked_prefill"]),
                "enable_prefix_caching": bool(training["enable_prefix_caching"]),
                "enforce_eager": True if smoke else bool(training["enforce_eager"]),
                "free_cache_engine": bool(training["free_cache_engine"]),
                "multi_turn": {"format": "hermes"},
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                        "chat_template": str(NO_THINKING_CHAT_TEMPLATE),
                        "max_model_len": int(training["max_model_len"]),
                        "mm_processor_cache_gb": 0,
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": int(training["ppo_micro_batch_size_per_gpu"]),
                "ppo_epochs": int(training["ppo_epochs"]),
                "use_dynamic_bsz": bool(training["use_dynamic_bsz"]),
                "ppo_max_token_len_per_gpu": int(training["ppo_max_token_len_per_gpu"]),
                # First average tokens within each action, then average action
                # rows. Combined with the 1/T advantage scaling this gives each
                # trajectory equal total policy weight regardless of turn count.
                "loss_agg_mode": str(training["loss_agg_mode"]),
                "shuffle": bool(training["shuffle"]),
                "grad_clip": float(training["grad_clip"]),
                "use_torch_compile": bool(training["actor_use_torch_compile"]),
                "optim": {
                    "lr": float(training["learning_rate"]),
                    "weight_decay": float(training["weight_decay"]),
                    "lr_scheduler_type": str(training["lr_scheduler_type"]),
                    "lr_warmup_steps_ratio": float(training["lr_warmup_steps_ratio"]),
                    "clip_grad": float(training["grad_clip"]),
                },
                # A smoke run only needs the consolidated LoRA adapter written
                # by the VERL worker. Formal runs retain full optimizer-resume
                # checkpoints.
                "checkpoint": {
                    "save_contents": [] if smoke else ["model", "optimizer", "extra"],
                    "load_contents": [] if smoke else ["model", "optimizer", "extra"],
                },
                "freeze_vision_tower": bool(training["freeze_vision_tower"]),
                "use_kl_loss": bool(training["use_kl_loss"]),
                "kl_loss_coef": float(training["kl_loss_coef"]),
                "kl_loss_type": str(training["kl_loss_type"]),
                "entropy_coeff": float(training["entropy_coeff"]),
                "clip_ratio": float(training["clip_ratio"]),
                "clip_ratio_low": float(training["clip_ratio_low"]),
                "clip_ratio_high": float(training["clip_ratio_high"]),
                "fsdp_config": {
                    "model_dtype": "bfloat16",
                    "param_offload": bool(training["fsdp_param_offload"]),
                    "optimizer_offload": bool(training["fsdp_optimizer_offload"]),
                    "use_torch_compile": bool(training["fsdp_use_torch_compile"]),
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": int(training["ref_log_prob_micro_batch_size_per_gpu"]),
                "log_prob_use_dynamic_bsz": bool(training["ref_log_prob_use_dynamic_bsz"]),
                "log_prob_max_token_len_per_gpu": int(training["ref_log_prob_max_token_len_per_gpu"]),
                "fsdp_config": {"param_offload": bool(training["ref_param_offload"])},
            },
        },
        "trainer": {
            "n_gpus_per_node": int(training["n_gpus_per_node"]),
            "nnodes": 1,
            "critic_warmup": 0,
            "logger": loggers,
            "project_name": str(swanlab["project_name"]),
            "experiment_name": experiment_name,
            "default_local_dir": output_dir,
            "resume_mode": "disable" if smoke else str(training["resume_mode"]),
            "max_actor_ckpt_to_keep": int(training["max_actor_ckpt_to_keep"]),
            "val_before_train": bool(training["val_before_train"]),
            "save_freq": 1 if smoke else int(training["save_freq"]),
            "test_freq": -1 if smoke else int(training["test_freq"]),
            "total_training_steps": 1 if smoke else -1,
            "total_epochs": 1 if smoke else int(training["total_epochs"]),
        },
    }


def _configure_swanlab_environment(run: dict[str, Any], *, smoke: bool) -> None:
    swanlab = cast(dict[str, Any], run["swanlab"])
    if not bool(swanlab["enabled"]):
        return
    os.environ["SWANLAB_LOG_DIR"] = str(swanlab["log_dir"])
    os.environ["SWANLAB_MODE"] = str(swanlab["smoke_mode"] if smoke else swanlab["mode"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    run = _load_run_config(config_path)
    expert_name = ExpertName(run["expert_name"])
    stage_config = load_stage_g_example_config(run["stage_config"])
    if args.smoke:
        stage_config.workflow.max_steps = 2
        stage_config.workflow.no_improvement_limit = 2
    registry = ToolRegistry.from_yaml(stage_config.tools_config)
    factory_class = SmokeControllerFactory if args.smoke else RealControllerFactory
    factory = factory_class(stage_config, registry)
    tokenizer = AutoTokenizer.from_pretrained(run["model_path"], trust_remote_code=True)
    agent = GRPOImageRestorationAgent(
        config=stage_config,
        factory=factory,
        expert_name=expert_name,
        tokenizer=tokenizer,
        smoke_override_actions=["scunet", "stop"] if args.smoke else None,
    )
    train_dataset = _load_dataset(run["train_data"], expert_name, limit=1 if args.smoke else None)
    val_dataset = _load_dataset(run["val_data"], expert_name, limit=1 if args.smoke else None)

    _configure_swanlab_environment(run, smoke=args.smoke)
    algorithm = agl.VERL(_verl_config(run, smoke=args.smoke))
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=1 if args.smoke else int(run["training"]["n_runners"]),
        tracer=agl.OtelTracer(),
        adapter=agl.LlmProxyTraceToTriplet(),
    )
    trainer.fit(
        agent,
        train_dataset=cast(agl.Dataset[dict[str, Any]], train_dataset),
        val_dataset=cast(agl.Dataset[dict[str, Any]], val_dataset),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
