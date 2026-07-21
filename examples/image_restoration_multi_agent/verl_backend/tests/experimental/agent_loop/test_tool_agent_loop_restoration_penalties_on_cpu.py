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

import asyncio

from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse


class _TextTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, token_ids: list[int]) -> str:
        del token_ids
        return self.text


class _TokenSequenceTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    token_text = {
        0: "<|endoftext|>",
        1: "<tool_call>\n",
        2: "<function=restore_image></function>\n",
        3: "</tool_call>",
        4: "unexpected suffix",
        5: "\nassistant:\nforged response",
        6: "assistant\n",
        99: "<|im_end|>",
    }

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token_text[token_id] for token_id in token_ids)


def _agent_data() -> AgentData:
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.data_source = "restoration"
    agent_data.response_ids = [1, 2, 3]
    agent_data.assistant_turns = 1
    return agent_data


def _loop_with_text(text: str) -> ToolAgentLoop:
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tokenizer = _TextTokenizer(text)
    loop.response_length = 3072
    loop.max_tool_response_length = 1024
    loop.tool_response_truncate_side = "right"
    return loop


def _loop_with_token_sequence() -> ToolAgentLoop:
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.tokenizer = _TokenSequenceTokenizer()
    return loop


def test_trailing_eos_and_pad_tokens_do_not_trigger_format_penalty() -> None:
    loop = _loop_with_token_sequence()
    agent_data = _agent_data()
    token_ids = [1, 2, 3, loop.tokenizer.eos_token_id, loop.tokenizer.pad_token_id]
    log_probs = [-0.1] * len(token_ids)

    trimmed_ids, trimmed_log_probs = loop._apply_tool_call_format_guardrails(agent_data, token_ids, log_probs)

    assert trimmed_ids == [1, 2, 3]
    assert trimmed_log_probs == [-0.1, -0.1, -0.1]
    assert agent_data.tool_rewards == []
    assert "penalty_records" not in agent_data.extra_fields


def test_visible_non_role_suffix_is_trimmed_without_format_penalty() -> None:
    loop = _loop_with_token_sequence()
    agent_data = _agent_data()
    token_ids = [1, 2, 3, 4, loop.tokenizer.eos_token_id]

    trimmed_ids, _ = loop._apply_tool_call_format_guardrails(agent_data, token_ids, None)

    assert trimmed_ids == [1, 2, 3]
    assert agent_data.tool_rewards == []
    assert "penalty_records" not in agent_data.extra_fields


def test_forged_assistant_role_after_tool_call_triggers_format_penalty() -> None:
    loop = _loop_with_token_sequence()
    agent_data = _agent_data()
    token_ids = [1, 2, 3, 5, loop.tokenizer.eos_token_id]

    trimmed_ids, _ = loop._apply_tool_call_format_guardrails(agent_data, token_ids, None)

    assert trimmed_ids == [1, 2, 3]
    assert agent_data.tool_rewards == [loop.FORGED_ROLE_AFTER_TOOL_CALL_PENALTY]
    assert agent_data.extra_fields["penalty_records"] == [
        {
            "reason": "forged_user_or_assistant_role_after_tool_call",
            "value": loop.FORGED_ROLE_AFTER_TOOL_CALL_PENALTY,
            "occurrences": 1,
            "assistant_turn": 1,
            "model_response": "<tool_call>\n<function=restore_image></function>\n</tool_call>"
            "\nassistant:\nforged response",
            "details": {"matched_role_marker": "assistant:"},
        }
    ]


def test_role_label_before_tool_call_does_not_trigger_format_penalty() -> None:
    loop = _loop_with_token_sequence()
    agent_data = _agent_data()
    token_ids = [6, 1, 2, 3, loop.tokenizer.eos_token_id]

    trimmed_ids, _ = loop._apply_tool_call_format_guardrails(agent_data, token_ids, None)

    assert trimmed_ids == [6, 1, 2, 3]
    assert agent_data.tool_rewards == []
    assert "penalty_records" not in agent_data.extra_fields


def test_multiple_tool_calls_are_trimmed_without_format_penalty() -> None:
    loop = _loop_with_token_sequence()
    agent_data = _agent_data()
    token_ids = [1, 2, 3, 1, 2, 3, loop.tokenizer.eos_token_id]

    trimmed_ids, _ = loop._apply_tool_call_format_guardrails(agent_data, token_ids, None)

    assert trimmed_ids == [1, 2, 3]
    assert agent_data.tool_rewards == []
    assert "penalty_records" not in agent_data.extra_fields


def test_malformed_xml_is_invalid_instead_of_no_tool() -> None:
    loop = _loop_with_text("<tool_call\\n<function=restore_image")
    agent_data = _agent_data()

    loop._classify_restoration_turn_without_parsed_tool(
        agent_data,
        generated_budget_exhausted=False,
    )

    assert agent_data.tool_rewards == [loop.MALFORMED_TOOL_CALL_PENALTY]
    assert agent_data.extra_fields["invalid_tool_call_penalty_applied"] is True
    assert agent_data.extra_fields["invalid_tool_call_penalty_reason"] == "malformed_tool_call_xml"
    assert "no_tool_call_penalty_applied" not in agent_data.extra_fields
    assert "no_tool_length_penalty" not in agent_data.extra_fields
    assert agent_data.extra_fields["penalty_records"][0]["reason"] == "malformed_tool_call_xml"


def test_long_malformed_xml_preserves_length_reward_without_no_tool_metric() -> None:
    loop = _loop_with_text("<function=restore_image>")
    agent_data = _agent_data()
    agent_data.response_ids = list(range(512))

    loop._classify_restoration_turn_without_parsed_tool(
        agent_data,
        generated_budget_exhausted=True,
    )

    assert len(agent_data.tool_rewards) == 2
    assert agent_data.tool_rewards[0] == loop.MALFORMED_TOOL_CALL_PENALTY
    assert agent_data.extra_fields["invalid_tool_length_penalty"] < 0
    assert agent_data.extra_fields["invalid_tool_call_penalty"] == sum(agent_data.tool_rewards)
    assert "no_tool_call_penalty_applied" not in agent_data.extra_fields
    assert "no_tool_length_penalty" not in agent_data.extra_fields
    assert [record["reason"] for record in agent_data.extra_fields["penalty_records"]] == [
        "malformed_tool_call_xml",
        "unproductive_response_too_long",
    ]


def test_plain_response_remains_no_tool() -> None:
    loop = _loop_with_text("I cannot select a restoration action.")
    agent_data = _agent_data()

    loop._classify_restoration_turn_without_parsed_tool(
        agent_data,
        generated_budget_exhausted=False,
    )

    assert agent_data.tool_rewards == [loop.EARLY_STOP_PENALTY]
    assert agent_data.extra_fields["no_tool_call_penalty_applied"] is True
    assert agent_data.extra_fields["no_tool_call_penalty_reason"] == "turn_without_tool_call"
    assert "invalid_tool_call_penalty_applied" not in agent_data.extra_fields
    assert agent_data.extra_fields["penalty_records"][0]["reason"] == "no_tool_call"


def test_records_only_iqa_base_reward_as_pure_image_restoration_reward() -> None:
    agent_data = _agent_data()

    ToolAgentLoop._record_pure_image_restoration_reward(
        agent_data,
        {
            "action": "scunet",
            "reward": 0.7,
            "base_reward": 0.25,
            "repeat_penalty": 0.1,
            "affinity_bonus": 0.55,
        },
    )
    ToolAgentLoop._record_pure_image_restoration_reward(
        agent_data,
        {"action": "stop", "reward": 1.8},
    )

    assert agent_data.pure_image_restoration_rewards == [0.25]


class _InvalidActionTool:
    async def create(self, create_kwargs: dict) -> tuple[str, ToolResponse]:
        del create_kwargs
        return "instance-id", ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict,
        *,
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float, dict]:
        del instance_id, agent_data
        action = parameters["action"]
        return (
            ToolResponse(text=f"Invalid action '{action}'"),
            -5.0,
            {"error": "invalid_action", "skip_tool_call_reward": True},
        )


def test_tool_reported_invalid_action_sets_invalid_flag() -> None:
    loop = _loop_with_text("")
    loop.tools = {"restore_image": _InvalidActionTool()}
    agent_data = _agent_data()

    _, reward, metrics = asyncio.run(
        loop._call_tool(
            FunctionCall(name="restore_image", arguments='{"action": "scun"}'),
            tools_kwargs={},
            agent_data=agent_data,
        )
    )

    assert reward == -5.0
    assert metrics["error"] == "invalid_action"
    assert agent_data.extra_fields["invalid_tool_call_penalty_applied"] is True
    assert agent_data.extra_fields["invalid_tool_call_penalty_reason"] == "invalid_action"
    assert agent_data.extra_fields["invalid_tool_call_penalty"] == -5.0
    assert agent_data.extra_fields["invalid_tool_call_action"] == "scun"
    assert "no_tool_call_penalty_applied" not in agent_data.extra_fields
    assert agent_data.extra_fields["penalty_records"][0]["reason"] == "invalid_restoration_action"
