from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch


PROBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "old_verl_grpo"
    / "scripts"
    / "per_token_entropy_probe.py"
)


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("per_token_entropy_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe_module()


def test_entropy_is_computed_in_fp32(probe) -> None:
    logits = torch.tensor([0.0, -1.0, -3.0], dtype=torch.bfloat16)
    expected_logits = logits.float()
    expected_probs = torch.softmax(expected_logits, dim=-1)
    expected = -(expected_probs * torch.log_softmax(expected_logits, dim=-1)).sum().item()

    assert probe.entropy_from_logits(logits) == pytest.approx(expected, abs=1e-7)


def test_probe_defaults_match_current_grpo_sampling(probe) -> None:
    args = probe.build_argument_parser().parse_args([])

    assert args.temperature == 1.0
    assert args.top_p == 1.0


def test_sampling_filters_apply_visual_bias_temperature_and_top_p(probe) -> None:
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 100.0])

    probabilities = probe.apply_sampling_filters(
        logits,
        temperature=0.5,
        top_p=0.9,
        top_k=-1,
        blocked_token_ids=[4],
    )

    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[4].item() == 0.0
    assert torch.count_nonzero(probabilities).item() == 2


def test_sequence_metadata_grows_with_generated_tokens(probe) -> None:
    inputs = {
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
        "mm_token_type_ids": torch.tensor([[1, 0]], dtype=torch.int64),
        "pixel_values": torch.ones((2, 3)),
    }

    probe.extend_sequence_inputs_for_generated_token(inputs)

    assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
    assert inputs["mm_token_type_ids"].tolist() == [[1, 0, 0]]
    assert inputs["pixel_values"].shape == (2, 3)


def test_initial_context_uses_structured_image_and_canonical_stop_schema(probe) -> None:
    registry = probe.ToolRegistry.from_yaml(probe.DEFAULT_TOOL_REGISTRY_PATH)
    row = pd.Series({"extra_info": {"expert_name": "fog"}})

    messages, tools = probe.build_grpo_initial_context(row, registry)

    assert messages[1]["content"][0] == {"type": "image"}
    assert "<image>" not in messages[1]["content"][1]["text"]
    action_enum = tools[0]["function"]["parameters"]["properties"]["action"]["enum"]
    assert "stop" in action_enum


class _CharacterTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_decision_point_maps_exact_action_offset_and_ignores_stop(probe) -> None:
    tokenizer = _CharacterTokenizer()
    text = "The image is hazy. I will use I_ridcp now.\n</think>"

    position, action, leading_text = probe.find_decision_point(
        text, list(range(len(text))), tokenizer
    )

    assert action == "ridcp"
    assert position == text.index("I_ridcp")
    assert leading_text == ""

    stop_text = "I should stop now.\n</think>"
    assert probe.find_decision_point(stop_text, list(range(len(stop_text))), tokenizer) == (-1, None, "")
