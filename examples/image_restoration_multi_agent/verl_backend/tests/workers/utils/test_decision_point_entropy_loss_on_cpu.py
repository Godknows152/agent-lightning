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


def _make_config(*, coeff: float = 0.002, first_token_coeff: float = 0.0, first_token_gate: str = "none"):
    return SimpleNamespace(
        decision_point_entropy_coeff=coeff,
        decision_point_first_token_entropy_coeff=first_token_coeff,
        decision_point_first_token_entropy_gate=first_token_gate,
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


def _run_first_token_ppo_loss(
    monkeypatch,
    *,
    normalized_entropies,
    raw_entropies,
    effective_counts,
    legal_masses,
    found,
    global_trajectory_count,
    dp_size=1,
    coeff=0.002,
    scheduled_coeff=None,
    entropy_gate=None,
    gate_mode="none",
):
    batch_size = len(found)
    response_mask = torch.ones((batch_size, 3), dtype=torch.float32)
    found_tensor = torch.tensor(found, dtype=torch.bool)
    data = TensorDict(
        {
            "advantages": torch.zeros_like(response_mask),
            "decision_first_token_found": found_tensor,
            "old_log_probs": torch.zeros_like(response_mask),
            "response_mask": response_mask,
        },
        batch_size=[batch_size],
    )
    if entropy_gate is not None:
        data["decision_first_token_entropy_gate"] = torch.tensor(entropy_gate, dtype=torch.bool)
    tu.assign_non_tensor(
        data,
        batch_num_decision_trajectories=global_trajectory_count,
        batch_num_tokens=int(response_mask.sum().item()),
        dp_size=dp_size,
        global_batch_size=dp_size,
    )
    if scheduled_coeff is not None:
        tu.assign_non_tensor(data, decision_point_first_token_entropy_coeff=scheduled_coeff)

    def fake_policy_loss(**kwargs):
        return kwargs["log_prob"].sum() * 0.0, {}

    monkeypatch.setattr(losses, "get_policy_loss_fn", lambda _: fake_policy_loss)
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _: tensor)

    return losses.ppo_loss(
        config=_make_config(coeff=0.0, first_token_coeff=coeff, first_token_gate=gate_mode),
        model_output={
            "log_probs": torch.zeros_like(response_mask, requires_grad=True),
            "decision_first_token_action_entropy": torch.tensor(raw_entropies, requires_grad=True),
            "decision_first_token_action_entropy_normalized": torch.tensor(normalized_entropies, requires_grad=True),
            "decision_first_token_effective_action_count": torch.tensor(effective_counts),
            "decision_first_token_legal_mass": torch.tensor(legal_masses),
        },
        data=data,
    )


def test_first_token_entropy_averages_decisions_within_each_trajectory(monkeypatch):
    policy_loss, metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.2, 0.0, 0.0], [0.6, 0.8, 1.0]],
        raw_entropies=[[0.4, 0.0, 0.0], [1.2, 1.6, 2.0]],
        effective_counts=[[2.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
        legal_masses=[[0.8, 0.0, 0.0], [0.6, 0.5, 0.4]],
        found=[[True, False, False], [True, True, True]],
        global_trajectory_count=2,
    )

    # trajectory means are 0.2 and 0.8, so the two trajectories contribute equally.
    assert metrics["actor/decision_point_first_token_action_entropy_normalized"].aggregate() == pytest.approx(0.5)
    first_token_metric_keys = {key for key in metrics if "decision_point_first_token" in key}
    assert first_token_metric_keys == {"actor/decision_point_first_token_action_entropy_normalized"}
    assert policy_loss.item() == pytest.approx(-0.001)


def test_first_token_entropy_uses_batch_schedule_coefficient(monkeypatch):
    policy_loss, _ = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.5]],
        raw_entropies=[[1.0]],
        effective_counts=[[4.0]],
        legal_masses=[[0.9]],
        found=[[True]],
        global_trajectory_count=1,
        coeff=0.002,
        scheduled_coeff=0.0005,
    )

    assert policy_loss.item() == pytest.approx(-0.00025)


def test_first_token_entropy_positive_advantage_gate_weights_only_selected_trajectories(monkeypatch):
    policy_loss, metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.2], [0.8]],
        raw_entropies=[[0.4], [1.6]],
        effective_counts=[[2.0], [5.0]],
        legal_masses=[[0.8], [0.5]],
        found=[[True], [True]],
        global_trajectory_count=2,
        entropy_gate=[True, False],
        gate_mode="positive_advantage",
    )

    assert metrics["actor/decision_point_first_token_action_entropy_normalized"].aggregate() == pytest.approx(0.5)
    assert metrics[
        "actor/decision_point_first_token_gated_action_entropy_normalized"
    ].aggregate() == pytest.approx(0.1)
    assert metrics["actor/decision_point_first_token_entropy_gate_ratio"].aggregate() == pytest.approx(0.5)
    assert policy_loss.item() == pytest.approx(-0.0002)


def test_first_token_entropy_quality_validity_gate_weights_only_selected_trajectories(monkeypatch):
    policy_loss, metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.2], [0.8]],
        raw_entropies=[[0.4], [1.6]],
        effective_counts=[[2.0], [5.0]],
        legal_masses=[[0.8], [0.5]],
        found=[[True], [True]],
        global_trajectory_count=2,
        entropy_gate=[True, False],
        gate_mode="quality_validity",
    )

    assert metrics["actor/decision_point_first_token_gated_action_entropy_normalized"].aggregate() == pytest.approx(
        0.1
    )
    assert metrics["actor/decision_point_first_token_entropy_gate_ratio"].aggregate() == pytest.approx(0.5)
    assert policy_loss.item() == pytest.approx(-0.0002)


def test_first_token_entropy_nonpositive_advantage_gate_weights_only_selected_trajectories(monkeypatch):
    policy_loss, metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.2], [0.8]],
        raw_entropies=[[0.4], [1.6]],
        effective_counts=[[2.0], [5.0]],
        legal_masses=[[0.8], [0.5]],
        found=[[True], [True]],
        global_trajectory_count=2,
        entropy_gate=[True, False],
        gate_mode="nonpositive_advantage",
    )

    assert metrics["actor/decision_point_first_token_gated_action_entropy_normalized"].aggregate() == pytest.approx(
        0.1
    )
    assert metrics["actor/decision_point_first_token_entropy_gate_ratio"].aggregate() == pytest.approx(0.5)
    assert policy_loss.item() == pytest.approx(-0.0002)


def test_first_token_entropy_positive_advantage_gate_requires_batch_metadata(monkeypatch):
    with pytest.raises(ValueError, match="decision_first_token_entropy_gate is required"):
        _run_first_token_ppo_loss(
            monkeypatch,
            normalized_entropies=[[0.5]],
            raw_entropies=[[1.0]],
            effective_counts=[[4.0]],
            legal_masses=[[0.9]],
            found=[[True]],
            global_trajectory_count=1,
            gate_mode="positive_advantage",
        )


def test_first_token_entropy_distributed_mean_uses_valid_trajectory_count(monkeypatch):
    _, rank0_metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.2, 0.0]],
        raw_entropies=[[0.4, 0.0]],
        effective_counts=[[2.0, 0.0]],
        legal_masses=[[0.8, 0.0]],
        found=[[True, False]],
        global_trajectory_count=2,
        dp_size=2,
    )
    _, rank1_metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.6, 1.0]],
        raw_entropies=[[1.2, 2.0]],
        effective_counts=[[4.0, 6.0]],
        legal_masses=[[0.6, 0.4]],
        found=[[True, True]],
        global_trajectory_count=2,
        dp_size=2,
    )

    normalized = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_first_token_action_entropy_normalized"],
            rank1_metrics["actor/decision_point_first_token_action_entropy_normalized"],
        ]
    )
    assert normalized == pytest.approx(0.5)


def test_first_token_entropy_handles_rank_without_valid_trajectory(monkeypatch):
    _, rank0_metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.0]],
        raw_entropies=[[0.0]],
        effective_counts=[[0.0]],
        legal_masses=[[0.0]],
        found=[[False]],
        global_trajectory_count=1,
        dp_size=2,
    )
    _, rank1_metrics = _run_first_token_ppo_loss(
        monkeypatch,
        normalized_entropies=[[0.7]],
        raw_entropies=[[1.4]],
        effective_counts=[[4.0]],
        legal_masses=[[0.6]],
        found=[[True]],
        global_trajectory_count=1,
        dp_size=2,
    )

    normalized = Metric.aggregate_dp(
        [
            rank0_metrics["actor/decision_point_first_token_action_entropy_normalized"],
            rank1_metrics["actor/decision_point_first_token_action_entropy_normalized"],
        ]
    )
    assert normalized == pytest.approx(0.7)
