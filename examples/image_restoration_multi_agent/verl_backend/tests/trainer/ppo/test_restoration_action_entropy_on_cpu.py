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
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tensordict import TensorDict

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _compute_nonpositive_advantage_entropy_gate,
    _compute_positive_advantage_entropy_gate,
    _compute_quality_validity_entropy_gate,
    _compute_reverse_quality_validity_entropy_gate,
)
from verl.trainer.ppo.restoration_action_entropy import (
    ALL_TURN_RESTORATION_ACTIONS,
    FIRST_TURN_RESTORATION_ACTIONS,
    SFT_THINKING_ACTION_SURFACES,
    action_sequence_entropies,
    find_called_restoration_action,
    find_first_restoration_action,
    find_restoration_decision_in_assistant_turn,
    first_token_action_entropies,
)
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    _build_decision_action_candidate_chunk,
)


def test_action_entropy_surfaces_match_shared_tool_registry() -> None:
    tools_config = Path(__file__).resolve().parents[4] / "config" / "tools.yaml"
    payload = yaml.safe_load(tools_config.read_text(encoding="utf-8"))
    registry_surfaces = {item["name"]: item["model_name"] for item in payload["tools"] if item.get("enabled", True)}

    assert dict(SFT_THINKING_ACTION_SURFACES) == registry_surfaces


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


@pytest.mark.parametrize("num_actions", [16, 17])
def test_uniform_first_token_action_distribution_has_unit_normalized_entropy(num_actions):
    logits = torch.zeros(2, 17, requires_grad=True)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[:, :num_actions] = True

    raw_entropy, normalized_entropy = first_token_action_entropies(logits, legal_mask)
    (-normalized_entropy.mean()).backward()

    assert torch.allclose(raw_entropy, torch.full((2,), math.log(num_actions)))
    assert torch.allclose(normalized_entropy, torch.ones(2))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_first_token_action_entropy_is_low_for_a_concentrated_distribution():
    logits = torch.full((1, 16), -20.0)
    logits[0, 3] = 20.0

    raw_entropy, normalized_entropy = first_token_action_entropies(logits)

    assert raw_entropy.item() < 1e-12
    assert normalized_entropy.item() < 1e-12


def test_positive_advantage_entropy_gate_uses_masked_trajectory_mean():
    advantages = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [-0.25, -0.25, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )

    gate = _compute_positive_advantage_entropy_gate(advantages, response_mask)

    assert gate.tolist() == [True, False, False]


def test_nonpositive_advantage_entropy_gate_uses_masked_trajectory_mean():
    advantages = torch.tensor(
        [
            [-0.5, -0.5, 0.0],
            [0.25, -0.25, 100.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )

    gate = _compute_nonpositive_advantage_entropy_gate(advantages, response_mask)

    assert gate.tolist() == [True, True, False, False]


def test_quality_validity_entropy_gate_requires_quality_and_valid_actions():
    advantages = torch.ones((4, 2), dtype=torch.float32)
    response_mask = torch.ones_like(advantages, dtype=torch.bool)
    pure_image_rewards = np.asarray([0.5, -0.1, 0.2, 0.2], dtype=object)
    penalty_records = np.empty(4, dtype=object)
    penalty_records[:] = [
        [],
        [],
        [{"reason": "invalid_restoration_action", "value": -10.0}],
        [{"reason": "repeated_restoration_action", "value": -1.0, "occurrences": 1}],
    ]

    gate = _compute_quality_validity_entropy_gate(
        advantages,
        response_mask,
        {
            "pure_image_restoration_reward": pure_image_rewards,
            "penalty_records": penalty_records,
        },
        min_pure_image_reward=0.0,
        max_repeated_actions=1,
    )

    assert gate.tolist() == [True, False, False, True]


def test_reverse_quality_validity_entropy_gate_opens_on_advantage_or_quality_failure():
    advantages = torch.tensor(
        [
            [-0.5, -0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [-0.5, -0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
        ]
    )
    response_mask = torch.ones_like(advantages, dtype=torch.bool)
    response_mask[5] = False
    pure_image_rewards = np.asarray([1.0, -0.1, 1.0, -0.1, -0.1, -0.1, np.nan], dtype=object)
    penalty_records = np.empty(7, dtype=object)
    penalty_records[:] = [
        [],
        [],
        [],
        [{"reason": "invalid_restoration_action", "value": -10.0}],
        [{"reason": "repeated_restoration_action", "value": -2.0, "occurrences": 2}],
        [],
        [],
    ]

    gate = _compute_reverse_quality_validity_entropy_gate(
        advantages,
        response_mask,
        {
            "pure_image_restoration_reward": pure_image_rewards,
            "penalty_records": penalty_records,
        },
        min_pure_image_reward=0.0,
        max_repeated_actions=1,
    )

    assert gate.tolist() == [True, True, False, False, False, False, False]


def test_turn_detector_uses_the_xml_action_to_disambiguate_thinking_text():
    text = (
        "I considered stop, but I will select N_focalnet_dehaze now.\n"
        "</think>\n<tool_call><function=restore_image><parameter=action>\n"
        "N_focalnet_dehaze\n</parameter></function></tool_call>"
    )

    assert find_called_restoration_action(text) == "focalnet_dehaze"
    decision = find_restoration_decision_in_assistant_turn(
        text,
        legal_actions=ALL_TURN_RESTORATION_ACTIONS,
    )
    assert decision == (text.index("N_focalnet_dehaze"), "focalnet_dehaze")


def test_stop_decision_does_not_match_stopping_as_a_substring():
    text = (
        "I will inspect the result before stopping. I will stop now.\n"
        "</think><tool_call><function=restore_image><parameter=action>stop</parameter>"
        "</function></tool_call>"
    )

    decision = find_restoration_decision_in_assistant_turn(
        text,
        legal_actions=ALL_TURN_RESTORATION_ACTIONS,
    )

    assert decision == (text.index("stop now"), "stop")


def test_full_trajectory_metadata_contains_every_assistant_decision_point():
    class BoundaryTokenizer:
        def __init__(self):
            self.token_to_id = {}
            self.id_to_token = {}

        @staticmethod
        def _pieces(text):
            pattern = r" [A-P]_[a-z_]+| stop|[\s\S]"
            return [(match.group(), match.span()) for match in re.finditer(pattern, text)]

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            ids = []
            for piece, _ in self._pieces(text):
                if piece not in self.token_to_id:
                    token_id = len(self.token_to_id) + 1
                    self.token_to_id[piece] = token_id
                    self.id_to_token[token_id] = piece
                ids.append(self.token_to_id[piece])
            return ids

        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(self.id_to_token[int(token_id)] for token_id in token_ids)

        def __call__(self, text, return_offsets_mapping, add_special_tokens):
            assert return_offsets_mapping
            return {
                "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
                "offset_mapping": [span for _, span in self._pieces(text)],
            }

    tokenizer = BoundaryTokenizer()
    first_turn = (
        "I will select A_real_esrgan.\n</think>\n<tool_call><function=restore_image>"
        "<parameter=action>A_real_esrgan</parameter></function></tool_call>"
    )
    tool_feedback = "\n<tool_response>result</tool_response>\n"
    second_turn = (
        "The result is sufficient, so I will stop.\n</think>\n<tool_call><function=restore_image>"
        "<parameter=action>stop</parameter></function></tool_call>"
    )
    first_ids = tokenizer.encode(first_turn, add_special_tokens=False)
    feedback_ids = tokenizer.encode(tool_feedback, add_special_tokens=False)
    second_ids = tokenizer.encode(second_turn, add_special_tokens=False)
    response_ids = first_ids + feedback_ids + second_ids
    response_mask = [1] * len(first_ids) + [0] * len(feedback_ids) + [1] * len(second_ids)
    batch = SimpleNamespace(
        batch=TensorDict(
            {
                "attention_mask": torch.ones(1, 2 + len(response_ids), dtype=torch.long),
                "prompts": torch.tensor([[90, 91]]),
                "response_mask": torch.tensor([response_mask]),
                "responses": torch.tensor([response_ids]),
            },
            batch_size=[1],
        )
    )
    trainer = SimpleNamespace(tokenizer=tokenizer)

    metadata, metrics = RayPPOTrainer._compute_decision_first_token_metadata(trainer, batch)

    assert metadata["decision_first_token_found"].tolist() == [[True, True]]
    assert metadata["decision_first_token_legal_mask"].sum(dim=-1).tolist() == [[16, 17]]
    assert metrics["actor/decision_point_first_token_found_rate"] == pytest.approx(1.0)
    assert set(metrics) == {"actor/decision_point_first_token_found_rate"}


@pytest.mark.parametrize(
    ("surface", "expected_action"),
    [
        ("I will use D_hvicidnet next.", "hvicidnet"),
        ("Choose E_lightdiff for this image.", "lightdiff"),
        ("Apply J_kanet before reassessing.", "kanet"),
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


def test_first_token_entropy_reuses_standard_forward_logits_and_backpropagates():
    offsets = torch.tensor([0, 5], dtype=torch.int64)
    micro_batch = TensorDict(
        {
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "decision_first_token_found": torch.tensor([[True]]),
            "decision_first_token_ids": torch.tensor([[[1, 2, 3]]]),
            "decision_first_token_legal_mask": torch.ones(1, 1, 3, dtype=torch.bool),
            "decision_first_token_response_index": torch.tensor([[1]]),
            "input_ids": torch.nested.nested_tensor_from_jagged(torch.arange(5), offsets),
            "prompts": torch.tensor([[10, 11]]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        micro_batch,
        batch_num_decision_trajectories=1,
        use_fused_kernels=False,
        use_remove_padding=False,
    )
    standard_log_probs = torch.nested.nested_tensor_from_jagged(
        torch.zeros(5, requires_grad=True),
        offsets,
    )
    model_logits = torch.zeros(1, 5, 6, requires_grad=True)
    with torch.no_grad():
        model_logits[0, 2, 1:4] = torch.tensor([3.0, 1.0, -1.0])
    engine = SimpleNamespace(use_ulysses_sp=False)

    raw, normalized, effective_count, legal_mass = FSDPEngineWithLMHead._compute_decision_first_token_entropies(
        engine,
        micro_batch,
        model_logits,
        standard_log_probs,
    )
    (-normalized.sum()).backward()

    assert 0.0 < normalized.item() < 1.0
    assert effective_count.item() == pytest.approx(raw.exp().item())
    assert 0.0 < legal_mass.item() <= 1.0
    assert model_logits.grad is not None
    assert model_logits.grad[0, 2, 1:4].abs().sum().item() > 0.0
    assert model_logits.grad[0, :2].abs().sum().item() == 0.0


@pytest.mark.parametrize(("logits_to_keep", "expected_length"), [(0, 7), (3, 3)])
def test_qwen3_5_normal_forward_applies_logits_to_keep_before_lm_head(logits_to_keep, expected_length):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from verl.models.transformers.qwen3_5 import forward_with_normal_backend

    class FakeOutputs:
        def __init__(self, hidden_states):
            self.last_hidden_state = hidden_states
            self.hidden_states = None

        def __getitem__(self, index):
            assert index == 0
            return self.last_hidden_state

    class FakeBackbone:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states
            self.kwargs = None

        def __call__(self, input_ids, **kwargs):
            del input_ids
            self.kwargs = kwargs
            return FakeOutputs(self.hidden_states)

    class RecordingLMHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shape = None

        def forward(self, hidden_states):
            self.input_shape = hidden_states.shape
            return hidden_states

    hidden_states = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4)
    backbone = FakeBackbone(hidden_states)
    lm_head = RecordingLMHead()
    model = SimpleNamespace(model=backbone, lm_head=lm_head)

    output = forward_with_normal_backend(
        model,
        input_ids=torch.ones(2, 7, dtype=torch.long),
        logits_to_keep=logits_to_keep,
        use_cache=False,
    )

    assert backbone.kwargs == {"use_cache": False}
    assert lm_head.input_shape == torch.Size([2, expected_length, 4])
    assert output.logits.shape == torch.Size([2, expected_length, 4])
