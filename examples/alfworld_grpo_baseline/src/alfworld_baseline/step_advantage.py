"""Task-group normalization over decision rows, as in GiGPO's GRPO baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from verl.trainer.ppo.core_algos import register_adv_est

ADV_ESTIMATOR = "alfworld_step_grpo"


@register_adv_est(ADV_ESTIMATOR)
@torch.no_grad()
def compute_step_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    config: Any = None,
    epsilon: float = 1e-6,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each row's outcome-plus-local-cost within its initial task.

    Match GiGPO's ``compute_mean_std_cross_steps=True`` and sample standard
    deviation. Do not deduplicate sessions, match step numbers, or group states.
    Masked padding rows contribute neither statistics nor advantages. The
    registered non-``grpo`` name bypasses native V1's final-row broadcast path.
    """
    if (
        token_level_rewards.shape != response_mask.shape
        or token_level_rewards.ndim != 2
    ):
        raise ValueError(
            "Step rewards and response masks must be matching two-dimensional tensors"
        )
    if len(index) != len(response_mask):
        raise ValueError("Each step must have an initial-task group uid")
    normalize = True if config is None else config.get("norm_adv_by_std_in_grpo", True)
    active = response_mask.bool().any(dim=-1)
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    groups: dict[Any, list[int]] = defaultdict(list)
    for row in active.nonzero(as_tuple=True)[0].tolist():
        groups[index[row]].append(row)
    advantages = torch.zeros_like(scores)
    for rows in groups.values():
        values = scores[rows]
        if len(rows) == 1:
            # GiGPO's GRPO baseline uses mean=0, std=1 for a singleton.
            mean, std = values.new_tensor(0.0), values.new_tensor(1.0)
        elif torch.all(values == values[0]):
            # Identical nonzero float32 scores can acquire spurious advantages
            # from reduction roundoff. The exact constant-group result is zero.
            continue
        else:
            mean, std = values.mean(), values.std()
        advantages[rows] = (
            (values - mean) / (std + epsilon) if normalize else values - mean
        )
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
