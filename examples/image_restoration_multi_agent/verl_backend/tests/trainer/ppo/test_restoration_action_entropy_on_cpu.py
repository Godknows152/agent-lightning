# Copyright 2026 Microsoft Corporation
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

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.restoration_action_entropy import (
    FIRST_TURN_RESTORATION_ACTIONS,
    action_sequence_entropies,
    find_first_restoration_action,
)
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    _build_decision_action_candidate_chunk,
)


def test_complete_action_entropy_is_normalized_and_differentiable():
    sequence_scores = torch.tensor(
        [[math.log(0.7), math.log(0.2), math.log(0.1)]],
        dtype=torch.float32,
        requires_grad=True,
    )

    raw_entropy, normalized_entropy = action_sequence_entropies(sequence_scores)
    (-normalized_entropy.mean()).backward()

    assert raw_entropy.item() == pytest.approx(0.801819, abs=1e-6)
    assert normalized_entropy.item() == pytest.approx(0.729847, abs=1e-6)
    assert sequence_scores.grad is not None
    assert torch.isfinite(sequence_scores.grad).all()
    assert not torch.allclose(sequence_scores.grad, torch.zeros_like(sequence_scores.grad))


def test_uniform_complete_action_distribution_has_unit_normalized_entropy():
    scores = torch.zeros(2, len(FIRST_TURN_RESTORATION_ACTIONS))

    raw_entropy, normalized_entropy = action_sequence_entropies(scores)

    assert torch.allclose(raw_entropy, torch.full((2,), math.log(16)))
    assert torch.allclose(normalized_entropy, torch.ones(2))


@pytest.mark.parametrize(
    ("surface", "expected_action"),
    [
        ("I will use HVI-CIDNet next.", "hvicidnet"),
        ("Choose LightenDiffusion for this image.", "lightdiff"),
        ("Apply KA-Net before reassessing.", "kanet"),
    ],
)
def test_detector_recognizes_exact_sft_action_surfaces(surface, expected_action):
    match = find_first_restoration_action(surface)

    assert match is not None
    assert match[1] == expected_action


def test_candidate_builder_replaces_selected_branch_and_preserves_token_positions():
    input_values = torch.tensor([10, 11, 20, 21], dtype=torch.long)
    offsets = torch.tensor([0, 4], dtype=torch.int64)
    input_ids = torch.nested.nested_tensor_from_jagged(input_values, offsets)
    position_ids = torch.nested.nested_tensor_from_jagged(torch.arange(4), offsets)
    action_token_ids = torch.tensor([[[30, 0], [40, 41]]], dtype=torch.long)
    action_token_mask = torch.tensor([[[True, False], [True, True]]])

    candidate_batch = _build_decision_action_candidate_chunk(
        input_ids,
        position_ids,
        torch.tensor([2]),
        action_token_ids,
        action_token_mask,
        [0, 1],
        pad_token_id=0,
    )
    candidate_input_ids, attention_mask, candidate_position_ids, targets, lengths, origins = candidate_batch

    assert candidate_input_ids.tolist() == [[0, 10, 11, 30], [10, 11, 40, 41]]
    assert attention_mask.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert candidate_position_ids.tolist() == [[0, 0, 1, 2], [0, 1, 2, 3]]
    assert targets.tolist() == [[30, 0], [40, 41]]
    assert lengths == [1, 2]
    assert origins == [0, 0]


def test_teacher_forced_candidate_forward_keeps_action_entropy_in_autograd_graph():
    class TransitionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transition_logits = torch.nn.Parameter(torch.zeros(6, 6))
            with torch.no_grad():
                self.transition_logits[0, 1] = 3.0
                self.transition_logits[1, 2] = 2.0
                self.transition_logits[1, 3] = -2.0

        def forward(self, input_ids, logits_to_keep, **kwargs):
            del kwargs
            logits = self.transition_logits[input_ids]
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    offsets = torch.tensor([0, 2], dtype=torch.int64)
    micro_batch = TensorDict(
        {
            "decision_action_found": torch.tensor([True]),
            "decision_action_token_ids": torch.tensor([[[1, 2], [1, 3]]]),
            "decision_action_token_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "decision_point_sequence_index": torch.tensor([1]),
            "input_ids": torch.nested.nested_tensor_from_jagged(torch.tensor([0, 5]), offsets),
            "position_ids": torch.nested.nested_tensor_from_jagged(torch.arange(2), offsets),
            "temperature": torch.tensor([1.0]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        micro_batch,
        batch_num_decision_points=1,
        decision_point_entropy_candidate_micro_batch_size=2,
        pad_token_id=0,
        use_fused_kernels=False,
    )
    standard_log_probs = torch.nested.nested_tensor_from_jagged(
        torch.zeros(2, requires_grad=True),
        offsets,
    )
    model = TransitionModel()
    engine = SimpleNamespace(use_ulysses_sp=False, module=model)

    raw_entropy, normalized_entropy = FSDPEngineWithLMHead._compute_decision_action_entropies(
        engine,
        micro_batch,
        standard_log_probs,
    )
    (-normalized_entropy.values().sum()).backward()

    assert 0.0 < normalized_entropy.values().item() < 1.0
    assert raw_entropy.values().item() > 0.0
    assert model.transition_logits.grad is not None
    assert model.transition_logits.grad[1, 2].abs().item() > 0.0
    assert model.transition_logits.grad[1, 3].abs().item() > 0.0
