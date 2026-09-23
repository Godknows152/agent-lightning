"""ALFWorld decision limits and native per-step response capacity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict


@dataclass(frozen=True)
class ALFWorldDecisionBudget:
    """Every decision attempt consumes one step, including invalid outputs."""

    max_steps: int
    max_new_tokens_per_turn: int

    @property
    def response_capacity(self) -> int:
        """Native V1 stores each decision in a separate response row."""
        return self.max_new_tokens_per_turn

    @classmethod
    def from_tool_config(cls, config: Any) -> ALFWorldDecisionBudget | None:
        if not config.get("environment_driven", False):
            return None
        values = {}
        for key, default in (("max_steps", 50), ("max_new_tokens_per_turn", 768)):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ALFWorld {key} must be a positive integer, got {value!r}")
            values[key] = value
        return cls(**values)


def configure_environment_driven_rollout(config: DictConfig) -> ALFWorldDecisionBudget | None:
    """Configure native step-sized tensors before workers initialize."""
    rollout = config.actor_rollout_ref.rollout
    tool_configs = OmegaConf.load(rollout.multi_turn.tool_config_path)
    entry = next(t for t in tool_configs.tools if t.class_name == "alfworld_baseline.alfworld_tool.ALFWorldTool")
    budget = ALFWorldDecisionBudget.from_tool_config(entry.config)
    if budget is None:
        raise ValueError("Native ALFWorld requires environment_driven=true")
    with open_dict(config):
        config.data.max_response_length = budget.response_capacity
        rollout.response_length = budget.response_capacity
        rollout.multi_turn.max_assistant_turns = None
        rollout.multi_turn.max_user_turns = None
    return budget


def validate_native_step_config(config: DictConfig) -> None:
    """Reject mismatched reward/loop/advantage backends before starting workers."""
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("ALFWorld step samples currently require native V1 sync training")
    backend = OmegaConf.select(config, "variables.TRAINING_BACKEND", default="trajectory")
    contracts = {
        "trajectory": ("grpo", "alfworld_tool_agent", "reward.py"),
        "gigpo_grpo": ("alfworld_step_grpo", "alfworld_step_grpo_agent", "step_reward.py"),
    }
    if backend not in contracts:
        raise ValueError(f"Unknown ALFWorld training backend: {backend!r}")
    estimator, loop, reward_file = contracts[backend]
    if config.algorithm.adv_estimator != estimator or config.algorithm.use_kl_in_reward:
        raise ValueError(f"ALFWorld {backend} requires adv_estimator={estimator} and use_kl_in_reward=false")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != loop:
        raise ValueError(f"ALFWorld {backend} requires agent loop {loop}")
    if not config.reward.custom_reward_function.path.endswith(f"alfworld_baseline/{reward_file}"):
        raise ValueError(f"ALFWorld {backend} requires reward function {reward_file}")
    manager = config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
    step_manager = "alfworld_baseline.step_workers.StepGRPOAgentLoopManager"
    if (backend == "gigpo_grpo" and manager != step_manager) or (backend == "trajectory" and manager == step_manager):
        raise ValueError(f"ALFWorld {backend} has a mismatched agent loop manager")
    loop_configs = OmegaConf.load(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    target = ("step_agent_loop.ALFWorldStepGRPOAgentLoop" if backend == "gigpo_grpo"
              else "agent_loop.ALFWorldToolAgentLoop")
    if not any(entry.name == loop and entry.get("_target_") == f"alfworld_baseline.{target}" for entry in loop_configs):
        raise ValueError(f"ALFWorld {backend} requires loop config for {target}")
    if OmegaConf.select(config, "distillation.enabled", default=False):
        raise ValueError("Native teacher scoring does not yet support every ALFWorld step")
    if config.reward.reward_model.enable:
        raise ValueError("ALFWorld uses environment rewards, not a colocated reward model")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("This ALFWorld migration supports native FSDP/FSDP2")
