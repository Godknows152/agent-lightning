"""Validate the canonical versioned ALFWorld training composition without GPUs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
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
    args = parser.parse_args()
    directory_name = f"seed{args.seed}" if args.kind == "full" else f"{args.kind}_seed{args.seed}"
    prefix = f"alfworld_{args.model_profile}_v1"
    experiment_name = f"{prefix}_seed{args.seed}" if args.kind == "full" else f"{prefix}_{args.kind}_seed{args.seed}"
    output = (args.output_dir or ROOT / "outputs" / "alfworld" / "v1" / "2gpu" / directory_name).resolve()
    log_dir = (args.log_dir or ROOT / "log" / "alfworld" / "v1" / "2gpu" / directory_name).resolve()
    swanlab_dir = (args.swanlab_log_dir or output / "swanlab").resolve()
    swanlab_mode = args.swanlab_mode or ("cloud" if args.kind == "full" else "offline")
    overrides = [
        f"trainer.default_local_dir={output}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR={swanlab_dir}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE={swanlab_mode}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR={log_dir}",
        f"variables.SEED={args.seed}",
    ]
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
    assert cfg.trainer.logger == ["console", "swanlab"]
    assert cfg.actor_rollout_ref.rollout.name == "sglang"
    assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "alfworld_tool_agent"
    assert str(cfg.variables.MODEL_PROFILE) == args.model_profile
    assert cfg.reward.custom_reward_function.path.endswith("alfworld_baseline/reward.py")
    assert Path(cfg.data.train_files).is_file()
    assert Path(cfg.data.val_files).is_file()
    assert int(cfg.trainer.n_gpus_per_node) == 2
    assert int(cfg.data.seed) == args.seed
    assert int(cfg.actor_rollout_ref.actor.data_loader_seed) == args.seed
    assert int(cfg.actor_rollout_ref.actor.fsdp_config.seed) == args.seed
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))
    print(OmegaConf.to_yaml(cfg.trainer))
    assert cfg.trainer.experiment_name == experiment_name
    assert str(cfg.trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR) == str(swanlab_dir)
    assert str(cfg.trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE) == swanlab_mode
    print(f"training_preflight_ok seed={args.seed} kind={args.kind} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
