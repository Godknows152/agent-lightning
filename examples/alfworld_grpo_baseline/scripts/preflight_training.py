"""Validate the canonical versioned ALFWorld training composition without GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # Configuration validation alone does not import the executable entrypoint.
    from alfworld_baseline.backend import assert_native_verl
    from alfworld_baseline.main_ppo import ALFWorldTaskRunner

    from verl.trainer.ppo.v1 import get_trainer_cls, AgentLoopManagerTQ
    from alfworld_baseline.budget import configure_environment_driven_rollout, validate_native_step_config

    print(f"native_verl={assert_native_verl()} runner={ALFWorldTaskRunner.__name__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--kind", choices=("full", "smoke", "pilot"), default="full")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--swanlab-log-dir", type=Path, default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config" / "alfworld" / "qwen35_2b" / "v1")
    parser.add_argument("--config-name", default="alfworld_config_2gpu")
    parser.add_argument("--model-profile", default="qwen35_2b")
    parser.add_argument("--training-backend", choices=("trajectory", "gigpo_grpo"), default="trajectory")
    parser.add_argument("--resume-config", choices=("qwen35_2b_gigpo",), default=None)
    args = parser.parse_args()
    if args.resume_config and (args.kind != "full" or args.model_profile != "qwen35_2b" or args.training_backend != "gigpo_grpo"):
        parser.error("The resume config requires a full Qwen3.5-2B GiGPO run")
    directory_name = f"seed{args.seed}" if args.kind == "full" else f"{args.kind}_seed{args.seed}"
    backend_suffix = "_gigpo_grpo" if args.training_backend == "gigpo_grpo" else ""
    prefix = f"alfworld_{args.model_profile}{backend_suffix}_v1"
    experiment_name = f"{prefix}_seed{args.seed}" if args.kind == "full" else f"{prefix}_{args.kind}_seed{args.seed}"
    if args.training_backend == "gigpo_grpo" and args.model_profile == "qwen35_2b":
        experiment_name = "qwen3.5_2B_GiGPO后端_full_sft"
        if args.kind != "full":
            experiment_name += f"_{args.kind}_seed{args.seed}"
    layout = Path(args.model_profile) / "gigpo_grpo" / "full_sft" if backend_suffix else Path(args.model_profile) / "full_sft"
    output = (args.output_dir or ROOT / "outputs" / "alfworld" / layout / "v1" / "2gpu" / directory_name).resolve()
    if args.training_backend == "gigpo_grpo" and args.model_profile == "qwen35_2b" and args.output_dir is None:
        output = ROOT / "log" / "alfworld" / "qwen35_2b" / "gigpo_grpo" / "full_sft"
        if args.kind != "full":
            output /= directory_name
    log_dir = (args.log_dir or ROOT / "log" / "alfworld" / layout / "v1" / "2gpu" / directory_name).resolve()
    swanlab_dir = (args.swanlab_log_dir or output / "swanlab").resolve()
    swanlab_mode = args.swanlab_mode or ("cloud" if args.kind == "full" else "offline")
    overrides = [
        f"trainer.default_local_dir={json.dumps(str(output), ensure_ascii=False)}",
        f"trainer.experiment_name={json.dumps(experiment_name, ensure_ascii=False)}",
        f"ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR={json.dumps(str(swanlab_dir), ensure_ascii=False)}",
        f"ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE={swanlab_mode}",
        f"ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR={json.dumps(str(log_dir), ensure_ascii=False)}",
        f"variables.SEED={args.seed}",
    ]
    if args.training_backend == "gigpo_grpo":
        overrides.append("+training_backend=gigpo_grpo")
    if args.resume_config:
        overrides.append(f"+resume_run={args.resume_config}")
    if args.kind in {"smoke", "pilot"}:
        overrides += [
            "variables.NUM_ROLLOUTS=2",
            "data.train_batch_size=8",
            "data.val_batch_size=8",
            "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
            f"trainer.total_training_steps={1 if args.kind == 'smoke' else 5}",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
        ]
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    validate_native_step_config(cfg)
    from alfworld_baseline.full_finetuning import validate_full_finetuning
    validate_full_finetuning(cfg)
    if args.resume_config:
        from alfworld_baseline.resume import validate_native_resume

        validate_native_resume(cfg)
        output_root = Path(cfg.trainer.default_local_dir)
        identity_path = output_root / ".swanlab_experiment.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            checkpoint = cfg.trainer.resume_from_path
            if cfg.trainer.resume_mode == "auto":
                step = int((output_root / "latest_checkpointed_iteration.txt").read_text().strip())
                checkpoint = output_root / f"global_step_{step}"
            print(f"resume_checkpoint={checkpoint} swanlab_run_id={identity['run_id']} resume=must")
        else:
            print(f"resume_mode={cfg.trainer.resume_mode} checkpoint=none start=new_run")
        print(f"ray_node_ip={cfg.ray_kwargs.ray_init._node_ip_address}")
    budget = configure_environment_driven_rollout(cfg)
    assert budget is not None
    # Match the worker's recursive schema validation before allocating GPUs.
    omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    from verl.utils.import_utils import load_class_from_fqn
    manager_fqn = cfg.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
    manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
    print(f"native_trainer={get_trainer_cls(cfg.trainer.v1.trainer_mode).__name__} manager={manager_cls.__name__}")
    assert cfg.trainer.logger == ["console", "swanlab"]
    assert cfg.trainer.enable_penalty_logging is False
    assert cfg.actor_rollout_ref.rollout.name == "sglang"
    assert str(cfg.variables.MODEL_PROFILE) == args.model_profile
    if args.training_backend == "gigpo_grpo":
        from alfworld_baseline.step_advantage import ADV_ESTIMATOR, compute_step_grpo_advantage
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn

        assert get_adv_estimator_fn(ADV_ESTIMATOR) is compute_step_grpo_advantage
    assert Path(cfg.data.train_files).is_file()
    assert Path(cfg.data.val_files).is_file()
    assert int(cfg.trainer.n_gpus_per_node) == 2
    assert int(cfg.data.seed) == args.seed
    assert int(cfg.actor_rollout_ref.actor.data_loader_seed) == args.seed
    assert int(cfg.actor_rollout_ref.actor.fsdp_config.seed) == args.seed
    if args.model_profile.startswith("qwen35") or args.model_profile.startswith("qwen25"):
        from alfworld_baseline.prompts_gigpo import NONTHINKING_PROMPT_VERSION, PROMPT_VERSION
        thinking = cfg.data.apply_chat_template_kwargs.enable_thinking
        assert isinstance(thinking, bool)
        prompt_version = PROMPT_VERSION if thinking else NONTHINKING_PROMPT_VERSION
        assert cfg.variables.PROMPT_VERSION == prompt_version
        assert cfg.variables.PROMPT_PROFILE == "gigpo"
        assert cfg.actor_rollout_ref.rollout.multi_turn.format == "qwen3_coder"
        assert cfg.actor_rollout_ref.rollout.multi_turn.max_assistant_turns is None
        assert cfg.actor_rollout_ref.rollout.response_length == budget.max_new_tokens_per_turn
        print(f"prompt_version={prompt_version} output=qwen3_xml_tool_call thinking={str(thinking).lower()}")
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))
    print(OmegaConf.to_yaml(cfg.trainer))
    assert cfg.trainer.experiment_name == experiment_name
    assert str(cfg.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR) == str(swanlab_dir)
    assert str(cfg.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE) == swanlab_mode
    print(f"training_preflight_ok seed={args.seed} kind={args.kind} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
