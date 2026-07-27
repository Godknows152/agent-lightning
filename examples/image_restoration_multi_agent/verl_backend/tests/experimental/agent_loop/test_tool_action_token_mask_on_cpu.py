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

import json

import pytest

from verl.experimental.agent_loop.tool_agent_loop import build_tool_action_token_mask
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
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        encoding = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        encoding["input_ids"] = [*encoding["input_ids"], 0]
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
    arguments = {"action": action, **extra_arguments}
    return FunctionCall(name="restore_image", arguments=json.dumps(arguments))


@pytest.mark.parametrize("action", ["scunet", "retinexformer_fivek", "mb_taylorformer_dehaze", "stop"])
def test_marks_only_the_structured_action_value(action: str) -> None:
    tokenizer = _CharacterTokenizer()
    text = _response(action, reasoning=f"I considered {action}, then selected it.\n")
    response_ids = tokenizer.encode(text)

    result = build_tool_action_token_mask(response_ids, tokenizer, [_call(action)], {action})

    assert result.matched is True
    assert result.action == action
    selected_text = "".join(character for character, selected in zip(text, result.mask, strict=True) if selected)
    assert selected_text == action
    assert sum(result.mask) == len(action)


def test_rejects_action_that_is_hidden_from_the_active_schema() -> None:
    tokenizer = _CharacterTokenizer()
    text = _response("stop")

    result = build_tool_action_token_mask(tokenizer.encode(text), tokenizer, [_call("stop")], {"scunet"})

    assert result.matched is False
    assert result.failure_reason == "action_not_in_active_schema"
    assert not any(result.mask)


@pytest.mark.parametrize(
    ("calls", "expected_reason"),
    [
        ([], "no_parsed_tool_call"),
        ([_call("scunet"), _call("ridcp")], "multiple_tool_calls"),
        ([FunctionCall(name="other_tool", arguments='{"action": "scunet"}')], "unexpected_function_name"),
        ([_call("scunet", unexpected="value")], "missing_or_extra_action_arguments"),
    ],
)
def test_rejects_ambiguous_or_non_restoration_calls(
    calls: list[FunctionCall],
    expected_reason: str,
) -> None:
    tokenizer = _CharacterTokenizer()
    text = _response("scunet")

    result = build_tool_action_token_mask(tokenizer.encode(text), tokenizer, calls, {"scunet", "ridcp"})

    assert result.matched is False
    assert result.failure_reason == expected_reason
    assert not any(result.mask)


def test_rejects_malformed_xml_without_fuzzy_fallback() -> None:
    tokenizer = _CharacterTokenizer()
    text = "<tool_call><function=restore_image><parameter=action>scunet"

    result = build_tool_action_token_mask(tokenizer.encode(text), tokenizer, [_call("scunet")], {"scunet"})

    assert result.matched is False
    assert result.failure_reason == "missing_action_xml_span"
    assert not any(result.mask)


def test_rejects_tokenizer_round_trip_mismatch() -> None:
    tokenizer = _RoundTripMismatchTokenizer()
    text = _response("scunet")

    result = build_tool_action_token_mask(tokenizer.encode(text), tokenizer, [_call("scunet")], {"scunet"})

    assert result.matched is False
    assert result.failure_reason == "token_id_roundtrip_mismatch"
    assert not any(result.mask)
