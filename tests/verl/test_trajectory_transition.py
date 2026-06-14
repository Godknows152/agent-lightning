"""Tests for trajectory-level rewards over independent visual transitions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agentlightning.verl.trainer import compute_trajectory_transition_grpo_advantage


def test_trajectory_transition_advantage_deduplicates_rollout_rewards_and_scales_by_turns() -> None:
    token_level_rewards = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )

    advantages, returns = compute_trajectory_transition_grpo_advantage(
        token_level_rewards,
        response_mask,
        data_ids=np.array(["image-1"] * 5),
        rollout_ids=np.array(["rollout-low", "rollout-low", "rollout-high", "rollout-high", "rollout-high"]),
    )

    normalized = 1.0 / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(advantages[0], torch.tensor([-normalized / 2, -normalized / 2]), atol=1e-6)
    assert torch.allclose(advantages[1], torch.tensor([-normalized / 2, 0.0]), atol=1e-6)
    assert torch.allclose(advantages[2], torch.tensor([normalized / 3, normalized / 3]), atol=1e-6)
    assert torch.allclose(advantages[4], torch.tensor([normalized / 3, 0.0]), atol=1e-6)
    assert torch.equal(returns, advantages)


def test_trajectory_transition_advantage_rejects_inconsistent_rewards_within_rollout() -> None:
    with pytest.raises(ValueError, match="inconsistent trajectory rewards"):
        compute_trajectory_transition_grpo_advantage(
            torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
            torch.ones((2, 2)),
            data_ids=np.array(["image-1", "image-1"]),
            rollout_ids=np.array(["rollout-1", "rollout-1"]),
        )
