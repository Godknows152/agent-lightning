# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0

import json
from pathlib import Path

import pytest

from verl.experimental.agent_loop.tool_agent_loop import find_tool_choice_decisions
from verl.experimental.agent_loop.tool_parser import FunctionCall


class _CharacterTokenizer:
    """Small reversible fast tokenizer with one token per character."""

    is_fast = True

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _RoundTripMismatchTokenizer(_CharacterTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, list]:
        encoding = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        encoding["input_ids"] = [*encoding["input_ids"], 0]
        return encoding


class _CandidateBoundaryCrossingTokenizer(_CharacterTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, list]:
        encoding = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        candidate_marker = "\nridcp\n</parameter>"
        if candidate_marker in text:
            action_start = text.index(candidate_marker) + 1
            encoding["offset_mapping"][action_start] = (action_start - 1, action_start + 1)
        return encoding


def _response(action: str, reasoning: str = "") -> str:
    return (
        f"{reasoning}<tool_call>\n"
        "<function=restore_image>\n"
        "<parameter=action>\n"
        f"{action}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def _call(action: str, **extra_arguments: str) -> FunctionCall:
    return FunctionCall(name="restore_image", arguments=json.dumps({"action": action, **extra_arguments}))


def test_regular_action_has_one_root_decision() -> None:
    tokenizer = _CharacterTokenizer()
    text = _response("scunet", reasoning="I considered scunet in prose first.\n")
    result = find_tool_choice_decisions(
        tokenizer.encode(text),
        tokenizer,
        [_call("scunet")],
        {"scunet", "ridcp"},
    )

    assert result.matched is True
    assert result.action == "scunet"
    assert len(result.decision_positions) == 1
    assert result.candidate_token_ids == ((ord("r"), ord("s")),)
    assert result.candidate_leaf_counts == ((1, 1),)
    assert text[result.decision_positions[0]] == "s"


@pytest.mark.parametrize(
    ("action", "other"),
    [("turbo_rain", "turbo_snow"), ("focalnet_dehaze", "focalnet_desnow")],
)
def test_shared_prefix_records_root_and_conditional_decisions(action: str, other: str) -> None:
    tokenizer = _CharacterTokenizer()
    text = _response(action)
    result = find_tool_choice_decisions(
        tokenizer.encode(text),
        tokenizer,
        [_call(action)],
        {action, other, "scunet"},
    )

    assert result.matched is True
    assert len(result.decision_positions) == 2
    assert sorted(result.candidate_leaf_counts[0]) == [1, 2]
    assert result.candidate_leaf_counts[1] == (1, 1)


def test_two_action_schema_still_has_a_valid_root_decision() -> None:
    tokenizer = _CharacterTokenizer()
    text = _response("scunet")
    result = find_tool_choice_decisions(tokenizer.encode(text), tokenizer, [_call("scunet")], {"scunet", "ridcp"})

    assert result.matched is True
    assert len(result.decision_positions) == 1
    assert sum(result.candidate_leaf_counts[0]) == 2


@pytest.mark.parametrize(
    ("text", "calls", "allowed_actions", "expected_reason"),
    [
        (_response("stop"), [_call("stop")], {"scunet"}, "action_not_in_active_schema"),
        (_response("scunet"), [], {"scunet", "ridcp"}, "no_parsed_tool_call"),
        (
            _response("scunet"),
            [_call("scunet"), _call("ridcp")],
            {"scunet", "ridcp"},
            "multiple_tool_calls",
        ),
        (
            "<tool_call><function=restore_image><parameter=action>scunet",
            [_call("scunet")],
            {"scunet", "ridcp"},
            "missing_action_xml_span",
        ),
    ],
)
def test_invalid_calls_fail_closed(
    text: str,
    calls: list[FunctionCall],
    allowed_actions: set[str],
    expected_reason: str,
) -> None:
    tokenizer = _CharacterTokenizer()
    result = find_tool_choice_decisions(tokenizer.encode(text), tokenizer, calls, allowed_actions)

    assert result.matched is False
    assert result.failure_reason == expected_reason
    assert result.decision_positions == ()


def test_rejects_tokenizer_round_trip_mismatch() -> None:
    tokenizer = _RoundTripMismatchTokenizer()
    text = _response("scunet")
    result = find_tool_choice_decisions(
        tokenizer.encode(text),
        tokenizer,
        [_call("scunet")],
        {"scunet", "ridcp"},
    )

    assert result.matched is False
    assert result.failure_reason == "token_id_roundtrip_mismatch"


def test_rejects_candidate_token_crossing_action_boundary() -> None:
    tokenizer = _CandidateBoundaryCrossingTokenizer()
    text = _response("scunet")
    result = find_tool_choice_decisions(
        tokenizer.encode(text),
        tokenizer,
        [_call("scunet")],
        {"scunet", "ridcp"},
    )

    assert result.matched is False
    assert result.failure_reason == "candidate_token_crosses_action_boundary"
    assert result.decision_positions == ()


@pytest.mark.parametrize(
    ("action", "other", "shared_first_token"),
    [("turbo_rain", "turbo_snow", 63161), ("focalnet_dehaze", "focalnet_desnow", 69)],
)
def test_qwen_tokenizer_collisions_create_conditional_decisions(
    action: str,
    other: str,
    shared_first_token: int,
) -> None:
    transformers = pytest.importorskip("transformers")
    model_path = Path("/home/LXJ/Python_Projects/Models/Qwen3.5-9B")
    if not model_path.exists():
        pytest.skip("Local Qwen3.5 tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = _response(action)
    response_ids = tokenizer.encode(text, add_special_tokens=False)
    result = find_tool_choice_decisions(
        response_ids,
        tokenizer,
        [_call(action)],
        {action, other, "scunet"},
    )

    assert result.matched is True
    assert len(result.decision_positions) == 2
    assert shared_first_token in result.candidate_token_ids[0]
    assert sorted(result.candidate_leaf_counts[0]) == [1, 2]
    assert result.candidate_leaf_counts[1] == (1, 1)
