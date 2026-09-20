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
        rollout.multi_turn.max_generated_response_length = None
        rollout.multi_turn.max_assistant_turns = None
        rollout.multi_turn.max_user_turns = None
    return budget


def validate_native_step_config(config: DictConfig) -> None:
    """Reject configurations that violate the episode-to-step GRPO contract."""
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("ALFWorld step samples currently require native V1 sync training")
    if config.algorithm.adv_estimator != "grpo" or config.algorithm.use_kl_in_reward:
        raise ValueError("ALFWorld requires trajectory-level GRPO and use_kl_in_reward=false")
    if OmegaConf.select(config, "distillation.enabled", default=False):
        raise ValueError("Native teacher scoring does not yet support every ALFWorld step")
    if config.reward.reward_model.enable:
        raise ValueError("ALFWorld uses environment rewards, not a colocated reward model")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("This ALFWorld migration supports native FSDP/FSDP2")
