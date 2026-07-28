# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

import math

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, OptimizerConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding
from verl.workers.utils.tool_choice_entropy import (
    gather_tool_choice_candidate_logits,
    restricted_tool_choice_entropy_from_candidates,
)


def _actor_config(coefficient: float) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        use_dynamic_bsz=True,
        optim=OptimizerConfig(lr=0.1),
        rollout_n=1,
        entropy_coeff=0.0,
        tool_choice_entropy_coeff=coefficient,
        calculate_entropy=False,
        use_kl_loss=False,
    )


def _batch(
    decision_mask: torch.Tensor | None,
    call_start_mask: torch.Tensor | None,
    response_mask: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
) -> TensorDict:
    if response_mask is None:
        response_mask = torch.ones(2, 3, dtype=torch.long)
    if advantages is None:
        advantages = torch.zeros(2, 3)
    tensors = {
        "prompts": torch.tensor([[1], [1]]),
        "responses": torch.tensor([[2, 3, 4], [2, 3, 4]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "response_mask": response_mask,
        "old_log_probs": torch.zeros(2, 3),
        "advantages": advantages,
    }
    if decision_mask is not None:
        tensors["tool_decision_position_mask"] = decision_mask
    if call_start_mask is not None:
        tensors["tool_choice_call_start_mask"] = call_start_mask
    data = TensorDict(tensors, batch_size=2)
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum()), global_batch_size=2)
    return data


def _model_output(tool_entropies: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    output = {"log_probs": torch.zeros(8, requires_grad=True)}
    if tool_entropies is not None:
        output["tool_choice_restricted_entropy"] = tool_entropies
    return output


def test_uniform_16_repair_action_distribution_has_log_16_entropy_and_backward() -> None:
    logits = torch.zeros(1, 1, 17, requires_grad=True)
    leaf_counts = torch.tensor([[[1] * 16 + [0]]])

    entropy = restricted_tool_choice_entropy_from_candidates(logits, leaf_counts)

    assert entropy.item() == pytest.approx(math.log(16))
    entropy.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dominant_tool_entropy_is_near_zero_and_empty_rows_are_finite() -> None:
    logits = torch.tensor([[[30.0, -30.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    leaf_counts = torch.tensor([[[1, 1, 0], [0, 0, 0]]])

    entropy = restricted_tool_choice_entropy_from_candidates(logits, leaf_counts)

    assert entropy[0, 0].item() < 1e-20
    assert entropy[0, 1].item() == 0.0
    assert torch.isfinite(entropy).all()
    entropy.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_candidate_gather_tracks_predictors_after_real_no_padding_micro_batch_conversion() -> None:
    prompts = torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23]])
    responses = torch.tensor([[31, 32, 0], [41, 42, 43]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]])
    padded_batch = TensorDict(
        {
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": attention_mask[:, prompts.shape[1] :],
            "position_ids": attention_mask.cumsum(dim=-1) - 1,
        },
        batch_size=2,
    )
    no_padding_batch = left_right_2_no_padding(padded_batch)
    local_sequence_width = int(no_padding_batch["input_ids"].offsets().diff().max())
    assert local_sequence_width == 6

    logits = torch.arange(2 * local_sequence_width * 5, dtype=torch.float32).reshape(2, local_sequence_width, 5)
    logits.requires_grad_()
    candidate_ids = torch.tensor(
        [
            [[1, 3], [0, 4], [0, 0]],
            [[2, 4], [1, 3], [0, 2]],
        ]
    )
    leaf_counts = torch.tensor(
        [
            [[1, 1], [1, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1]],
        ]
    )

    gathered = gather_tool_choice_candidate_logits(
        logits,
        candidate_ids,
        leaf_counts,
        no_padding_batch["attention_mask"],
        prompt_width=prompts.shape[1],
    )

    # Real prompt lengths are 2 and 3, so their first response predictors are
    # positions 1 and 2 in the locally right-padded actor input.
    expected = torch.stack(
        (
            torch.stack((logits[0, 1, [1, 3]], logits[0, 2, [0, 4]], logits[0, 0, [0, 0]])),
            torch.stack((logits[1, 2, [2, 4]], logits[1, 3, [1, 3]], logits[1, 4, [0, 2]])),
        )
    )
    torch.testing.assert_close(gathered, expected)
    gathered.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_coefficient_preserves_baseline_without_tool_fields() -> None:
    loss, metrics = ppo_loss(_actor_config(0.0), _model_output(), _batch(None, None))

    assert loss.item() == pytest.approx(0.0)
    assert "actor/tool_choice_restricted_entropy" not in metrics


def test_coefficient_001_adds_expected_call_normalized_bonus() -> None:
    entropy = torch.tensor([[2.0, 1.0, 99.0], [4.0, 99.0, 99.0]], requires_grad=True)
    decision_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    call_start_mask = torch.tensor([[1, 0, 0], [1, 0, 0]])
    positive_advantages = torch.ones(2, 3)

    loss, metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(entropy),
        _batch(decision_mask, call_start_mask, advantages=positive_advantages),
    )
    baseline_loss, _ = ppo_loss(
        _actor_config(0.0),
        _model_output(),
        _batch(None, None, advantages=positive_advantages),
    )

    # Per-trajectory values are 3 and 4, so the global mean is 3.5.
    assert metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(3.5)
    assert metrics["actor/tool_choice_entropy_bonus"].aggregate() == pytest.approx(0.035)
    assert metrics["actor/tool_choice_entropy_coeff"].aggregate() == pytest.approx(0.01)
    assert metrics["actor/tool_choice_entropy_gate_ratio"].aggregate() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(baseline_loss.item() - 0.035)
    loss.backward()
    assert entropy.grad is not None
    assert torch.isfinite(entropy.grad).all()


def test_two_calls_and_one_call_have_the_same_entropy_scale() -> None:
    one_call_entropy = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    two_call_entropy = torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    positive_advantages = torch.ones(2, 3)

    _, one_call_metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(one_call_entropy),
        _batch(
            torch.tensor([[1, 0, 0], [0, 0, 0]]),
            torch.tensor([[1, 0, 0], [0, 0, 0]]),
            advantages=positive_advantages,
        ),
    )
    _, two_call_metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(two_call_entropy),
        _batch(
            torch.tensor([[1, 1, 0], [0, 0, 0]]),
            torch.tensor([[1, 1, 0], [0, 0, 0]]),
            advantages=positive_advantages,
        ),
    )

    assert one_call_metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(1.0)
    assert two_call_metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(1.0)


def test_two_tool_schema_and_zero_decision_batch_are_finite() -> None:
    entropy = torch.tensor([[math.log(2), 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    decision_mask = torch.tensor([[1, 0, 0], [0, 0, 0]])
    call_start_mask = torch.tensor([[1, 0, 0], [0, 0, 0]])
    loss, metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(entropy),
        _batch(decision_mask, call_start_mask, advantages=torch.ones(2, 3)),
    )
    assert metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(math.log(2) / 2)
    assert torch.isfinite(loss)

    zero_entropy = torch.zeros(2, 3, requires_grad=True)
    zero_loss, zero_metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(zero_entropy),
        _batch(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long)),
    )
    assert zero_metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(0.0)
    assert torch.isfinite(zero_loss)
    zero_loss.backward()
    assert zero_entropy.grad is not None
    assert torch.isfinite(zero_entropy.grad).all()


def test_non_positive_advantage_disables_tool_choice_entropy_bonus() -> None:
    entropy = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], requires_grad=True)
    masks = torch.tensor([[1, 0, 0], [1, 0, 0]])

    loss, metrics = ppo_loss(
        _actor_config(0.01),
        _model_output(entropy),
        _batch(masks, masks, advantages=-torch.ones(2, 3)),
    )
    baseline_loss, _ = ppo_loss(
        _actor_config(0.0),
        _model_output(),
        _batch(None, None, advantages=-torch.ones(2, 3)),
    )

    assert metrics["actor/tool_choice_restricted_entropy"].aggregate() == pytest.approx(0.0)
    assert metrics["actor/tool_choice_entropy_bonus"].aggregate() == pytest.approx(0.0)
    assert metrics["actor/tool_choice_entropy_gate_ratio"].aggregate() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(baseline_loss.item())


def test_positive_coefficient_requires_tool_choice_fields_and_model_entropy() -> None:
    with pytest.raises(ValueError, match="tool_choice_call_start_mask"):
        ppo_loss(_actor_config(0.01), _model_output(), _batch(None, None))

    masks = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="tool_choice_restricted_entropy"):
        ppo_loss(_actor_config(0.01), _model_output(), _batch(masks, masks))
