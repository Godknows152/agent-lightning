"""ALFWorld decision budgets and loss-tensor capacity (not rollout cutoffs)."""
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
        """Enough slots for every possible generated token; excludes observations."""
        return self.max_steps * self.max_new_tokens_per_turn

    @classmethod
    def from_tool_config(cls, config: Any) -> ALFWorldDecisionBudget | None:
        if not config.get("environment_driven", False):
            return None
        values = {}
        for key, default in (("max_steps", 50), ("max_new_tokens_per_turn", 256)):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ALFWorld {key} must be a positive integer, got {value!r}")
            values[key] = value
        return cls(**values)


def configure_environment_driven_rollout(config: DictConfig) -> ALFWorldDecisionBudget | None:
    """Size fixed-shape VERL tensors before workers initialize from this config.

    The ALFWorld-only loop packs model outputs and replays actual per-turn
    prompts separately. There is no cumulative generation-token stop condition.
    Other profiles keep their existing trajectory layout and limits.
    """
    rollout = config.actor_rollout_ref.rollout
    tool_configs = OmegaConf.load(rollout.multi_turn.tool_config_path)
    entry = next(t for t in tool_configs.tools if t.class_name == "alfworld_baseline.alfworld_tool.ALFWorldTool")
    budget = ALFWorldDecisionBudget.from_tool_config(entry.config)
    if budget is None:
        return None
    if not config.actor_rollout_ref.model.use_remove_padding:
        raise ValueError("Environment-driven ALFWorld requires use_remove_padding=True for turn-context replay")
    with open_dict(config):
        config.data.max_response_length = budget.response_capacity
        rollout.response_length = budget.response_capacity
        rollout.multi_turn.max_generated_response_length = None
        rollout.multi_turn.max_assistant_turns = None
        rollout.multi_turn.max_user_turns = None
    return budget
