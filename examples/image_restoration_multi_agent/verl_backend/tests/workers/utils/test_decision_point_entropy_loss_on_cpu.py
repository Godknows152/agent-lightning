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


def _make_config(*, coeff: float = 0.05):
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


def _run_ppo_loss(monkeypatch, *, entropy, decision_mask, global_decision_count, dp_size=1, coeff=0.05):
    entropy = torch.tensor([entropy], dtype=torch.float32)
    decision_mask = torch.tensor([decision_mask], dtype=torch.float32)
    response_mask = torch.ones_like(decision_mask)
    data = TensorDict(
        {
            "advantages": torch.zeros_like(entropy),
            "decision_point_mask": decision_mask,
            "old_log_probs": torch.zeros_like(entropy),
            "response_mask": response_mask,
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        batch_num_decision_points=global_decision_count,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=dp_size,
        global_batch_size=dp_size,
    )

    def fake_policy_loss(**kwargs):
        del kwargs
        return torch.tensor(0.0), {}

    monkeypatch.setattr(losses, "get_policy_loss_fn", lambda _: fake_policy_loss)
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _: tensor)

    return losses.ppo_loss(
        config=_make_config(coeff=coeff),
        model_output={"log_probs": torch.zeros_like(entropy), "entropy": entropy},
        data=data,
    )


def test_decision_point_entropy_uses_decision_count_not_response_token_count(monkeypatch):
    policy_loss, metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[0.09, 8.0, 1.91, 6.0, 5.0, 4.0],
        decision_mask=[1, 0, 1, 0, 0, 0],
        global_decision_count=2,
        coeff=0.05,
    )

    decision_entropy = metrics["actor/decision_point_entropy"].aggregate()
    assert decision_entropy == pytest.approx(1.0)
    assert policy_loss.item() == pytest.approx(-0.05)


def test_decision_point_entropy_distributed_sum_reconstructs_global_mean(monkeypatch):
    _, rank0_metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[0.09, 8.0],
        decision_mask=[1, 0],
        global_decision_count=3,
        dp_size=2,
    )
    _, rank1_metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[0.91, 2.0],
        decision_mask=[1, 1],
        global_decision_count=3,
        dp_size=2,
    )

    decision_entropy = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_entropy"],
            rank1_metrics["actor/decision_point_entropy"],
        ]
    )
    assert decision_entropy == pytest.approx(1.0)


def test_decision_point_entropy_handles_a_rank_without_local_points(monkeypatch):
    _, rank0_metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[8.0, 9.0],
        decision_mask=[0, 0],
        global_decision_count=2,
        dp_size=2,
    )
    _, rank1_metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[1.0, 3.0],
        decision_mask=[1, 1],
        global_decision_count=2,
        dp_size=2,
    )

    decision_entropy = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_entropy"],
            rank1_metrics["actor/decision_point_entropy"],
        ]
    )
    assert decision_entropy == pytest.approx(2.0)


def test_decision_point_entropy_is_zero_without_a_found_point(monkeypatch):
    policy_loss, metrics = _run_ppo_loss(
        monkeypatch,
        entropy=[0.09, 1.91],
        decision_mask=[0, 0],
        global_decision_count=0,
    )

    assert metrics["actor/decision_point_entropy"].aggregate() == pytest.approx(0.0)
    assert policy_loss.item() == pytest.approx(0.0)
