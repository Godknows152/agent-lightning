# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, OptimizerConfig
from verl.workers.utils.losses import ppo_loss


def _actor_config(tool_action_entropy_coeff: float) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        use_dynamic_bsz=True,
        optim=OptimizerConfig(lr=0.1),
        rollout_n=1,
        entropy_coeff=0.0,
        tool_action_entropy_coeff=tool_action_entropy_coeff,
        calculate_entropy=True,
        use_kl_loss=False,
    )


def _batch(
    tool_action_token_mask: torch.Tensor | None,
    response_mask: torch.Tensor | None = None,
) -> TensorDict:
    if response_mask is None:
        response_mask = torch.ones(2, 3, dtype=torch.long)
    tensors = {
        "prompts": torch.tensor([[1], [1]]),
        "responses": torch.tensor([[2, 3, 4], [2, 3, 4]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "response_mask": response_mask,
        "old_log_probs": torch.zeros(2, 3),
        "advantages": torch.zeros(2, 3),
    }
    if tool_action_token_mask is not None:
        tensors["tool_action_token_mask"] = tool_action_token_mask
    data = TensorDict(tensors, batch_size=2)
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum()), global_batch_size=2)
    return data


def _model_output(response_entropies: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Place two length-three response rows in the flattened no-padding output."""

    entropy = torch.stack(
        (
            response_entropies[0, 0],
            response_entropies[0, 1],
            response_entropies[0, 2],
            torch.tensor(0.0),
            response_entropies[1, 0],
            response_entropies[1, 1],
            response_entropies[1, 2],
            torch.tensor(0.0),
        )
    ).requires_grad_()
    return {"log_probs": torch.zeros(8, requires_grad=True), "entropy": entropy}, entropy


def test_zero_coefficient_preserves_baseline_loss_without_requiring_mask() -> None:
    entropies = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    output_without_mask, _ = _model_output(entropies)
    output_with_mask, _ = _model_output(entropies)

    baseline_loss, baseline_metrics = ppo_loss(
        _actor_config(0.0),
        output_without_mask,
        _batch(None),
    )
    masked_loss, masked_metrics = ppo_loss(
        _actor_config(0.0),
        output_with_mask,
        _batch(torch.ones(2, 3, dtype=torch.long)),
    )

    torch.testing.assert_close(masked_loss, baseline_loss, rtol=0, atol=0)
    assert "actor/tool_action_token_entropy" not in baseline_metrics
    assert "actor/tool_action_token_entropy" not in masked_metrics


def test_positive_coefficient_adds_expected_trajectory_normalized_bonus() -> None:
    output, entropy = _model_output(torch.tensor([[2.0, 99.0, 4.0], [5.0, 99.0, 7.0]]))
    mask = torch.tensor([[1, 0, 1], [0, 0, 0]])

    loss, metrics = ppo_loss(_actor_config(0.001), output, _batch(mask))

    # First trajectory mean: (2 + 4) / 2 = 3. The second contributes zero;
    # the global batch mean is therefore 3 / 2 = 1.5.
    assert metrics["actor/tool_action_token_entropy"].aggregate() == pytest.approx(1.5)
    assert metrics["actor/tool_action_entropy_bonus"].aggregate() == pytest.approx(0.0015)
    assert metrics["actor/tool_action_entropy_coeff"].aggregate() == pytest.approx(0.001)
    assert loss.item() == pytest.approx(-0.0015)

    loss.backward()
    assert entropy.grad is not None
    assert torch.isfinite(entropy.grad).all()


def test_mask_is_intersected_with_response_mask_and_all_zero_is_finite() -> None:
    output, entropy = _model_output(torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    response_mask = torch.tensor([[1, 0, 1], [1, 1, 1]])
    action_mask = torch.tensor([[0, 1, 0], [0, 0, 0]])

    loss, metrics = ppo_loss(
        _actor_config(0.001),
        output,
        _batch(action_mask, response_mask=response_mask),
    )

    assert metrics["actor/tool_action_token_entropy"].aggregate() == pytest.approx(0.0)
    assert metrics["actor/tool_action_entropy_bonus"].aggregate() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert entropy.grad is not None
    assert torch.isfinite(entropy.grad).all()


def test_more_masked_tokens_do_not_linearly_increase_bonus() -> None:
    entropies = torch.full((2, 3), 3.0)
    one_token_output, _ = _model_output(entropies)
    three_token_output, _ = _model_output(entropies)

    _, one_token_metrics = ppo_loss(
        _actor_config(0.001),
        one_token_output,
        _batch(torch.tensor([[1, 0, 0], [0, 0, 0]])),
    )
    _, three_token_metrics = ppo_loss(
        _actor_config(0.001),
        three_token_output,
        _batch(torch.tensor([[1, 1, 1], [0, 0, 0]])),
    )

    assert one_token_metrics["actor/tool_action_token_entropy"].aggregate() == pytest.approx(1.5)
    assert three_token_metrics["actor/tool_action_token_entropy"].aggregate() == pytest.approx(1.5)


def test_positive_coefficient_requires_action_mask() -> None:
    output, _ = _model_output(torch.ones(2, 3))

    with pytest.raises(ValueError, match="tool_action_token_mask is required"):
        ppo_loss(_actor_config(0.001), output, _batch(None))
