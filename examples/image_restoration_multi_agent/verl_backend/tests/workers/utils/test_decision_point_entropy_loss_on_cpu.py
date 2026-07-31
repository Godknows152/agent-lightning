# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.workers.utils import losses


def _make_config(*, coeff: float = 0.002):
    return SimpleNamespace(
        decision_point_entropy_coeff=coeff,
        entropy_coeff=0.0,
        global_batch_info={},
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        loss_agg_mode="token-mean",
        loss_scale_factor=None,
        policy_loss={"loss_mode": "vanilla"},
        use_kl_loss=False,
    )


def _singleton_nested(values: list[float]) -> torch.Tensor:
    value_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    offsets = torch.arange(len(values) + 1, dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(value_tensor, offsets)


def _run_ppo_loss(
    monkeypatch,
    *,
    normalized_action_entropies,
    raw_action_entropies,
    found,
    global_decision_count,
    dp_size=1,
    coeff=0.002,
):
    batch_size = len(found)
    response_length = 3
    token_entropy = torch.full((batch_size, response_length), 99.0)
    decision_mask = torch.zeros((batch_size, response_length), dtype=torch.float32)
    for sample_index, is_found in enumerate(found):
        if is_found:
            decision_mask[sample_index, 1] = 1.0
    response_mask = torch.ones_like(decision_mask)
    data = TensorDict(
        {
            "advantages": torch.zeros_like(response_mask),
            "decision_point_mask": decision_mask,
            "old_log_probs": torch.zeros_like(response_mask),
            "response_mask": response_mask,
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor(
        data,
        batch_num_decision_points=global_decision_count,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=dp_size,
        global_batch_size=dp_size,
    )

    def fake_policy_loss(**kwargs):
        return kwargs["log_prob"].sum() * 0.0, {}

    monkeypatch.setattr(losses, "get_policy_loss_fn", lambda _: fake_policy_loss)
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _: tensor)

    return losses.ppo_loss(
        config=_make_config(coeff=coeff),
        model_output={
            "log_probs": torch.zeros_like(response_mask, requires_grad=True),
            "entropy": token_entropy,
            "decision_action_sequence_entropy": _singleton_nested(raw_action_entropies),
            "decision_action_normalized_entropy": _singleton_nested(normalized_action_entropies),
        },
        data=data,
    )


def test_decision_point_loss_uses_normalized_action_entropy_not_token_entropy(monkeypatch):
    policy_loss, metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.25, 0.75],
        raw_action_entropies=[0.5, 1.5],
        found=[True, True],
        global_decision_count=2,
    )

    assert metrics["actor/decision_point_entropy"].aggregate() == pytest.approx(0.5)
    assert metrics["actor/decision_point_action_sequence_entropy"].aggregate() == pytest.approx(1.0)
    assert policy_loss.item() == pytest.approx(-0.001)


def test_decision_point_entropy_distributed_sum_reconstructs_global_mean(monkeypatch):
    _, rank0_metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.1],
        raw_action_entropies=[0.2],
        found=[True],
        global_decision_count=3,
        dp_size=2,
    )
    _, rank1_metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.9, 0.5],
        raw_action_entropies=[1.8, 1.0],
        found=[True, True],
        global_decision_count=3,
        dp_size=2,
    )

    decision_entropy = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_entropy"],
            rank1_metrics["actor/decision_point_entropy"],
        ]
    )
    assert decision_entropy == pytest.approx(0.5)


def test_decision_point_entropy_handles_a_rank_without_local_points(monkeypatch):
    _, rank0_metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.0, 0.0],
        raw_action_entropies=[0.0, 0.0],
        found=[False, False],
        global_decision_count=2,
        dp_size=2,
    )
    _, rank1_metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.2, 0.6],
        raw_action_entropies=[0.4, 1.2],
        found=[True, True],
        global_decision_count=2,
        dp_size=2,
    )

    decision_entropy = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_entropy"],
            rank1_metrics["actor/decision_point_entropy"],
        ]
    )
    assert decision_entropy == pytest.approx(0.4)


def test_decision_point_entropy_is_zero_without_a_found_point(monkeypatch):
    policy_loss, metrics = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.0, 0.0],
        raw_action_entropies=[0.0, 0.0],
        found=[False, False],
        global_decision_count=0,
    )

    assert metrics["actor/decision_point_entropy"].aggregate() == pytest.approx(0.0)
    assert policy_loss.item() == pytest.approx(0.0)


def test_decision_point_entropy_backpropagates_through_action_distribution(monkeypatch):
    policy_loss, _ = _run_ppo_loss(
        monkeypatch,
        normalized_action_entropies=[0.3],
        raw_action_entropies=[0.7],
        found=[True],
        global_decision_count=1,
    )
    policy_loss.backward()

    assert policy_loss.grad_fn is not None
