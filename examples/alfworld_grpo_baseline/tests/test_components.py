from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "image_restoration_multi_agent" / "verl_backend"))
from alfworld_baseline.parser import ParseStatus, parse_tool_call
from alfworld_baseline.prompts import build_messages
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from alfworld_baseline.validator import ValidationStatus, validate_tool_call

def test_registry_preserves_exact_admissible_actions():
    registry = ALFWorldToolRegistry(["go to cabinet 1", "open cabinet 1"])
    assert registry.build_tool_schema()["function"]["parameters"]["properties"]["action"]["enum"] == ["go to cabinet 1", "open cabinet 1"]

def test_parser_and_validator_accept_qwen25_native_json():
    raw = '<tool_call>{"name":"alfworld_action","arguments":{"action":"go to cabinet 1"}}</tool_call>'
    parsed = parse_tool_call(raw)
    result = validate_tool_call(parsed, ALFWorldToolRegistry(["go to cabinet 1"]))
    assert parsed.status is ParseStatus.VALID
    assert result.status is ValidationStatus.VALID
    assert result.action == "go to cabinet 1"

def test_parser_and_validator_accept_qwen35_native_xml():
    raw = '<tool_call>\n<function=alfworld_action>\n<parameter=action>\ngo to cabinet 1\n</parameter>\n</function>\n</tool_call>'
    parsed = parse_tool_call(raw)
    result = validate_tool_call(parsed, ALFWorldToolRegistry(["go to cabinet 1"]))
    assert parsed.status is ParseStatus.VALID
    assert result.status is ValidationStatus.VALID
    assert result.action == "go to cabinet 1"

def test_parser_rejects_multiple_calls_and_validator_rejects_unknown_action():
    assert parse_tool_call('<tool_call>{}</tool_call><tool_call>{}</tool_call>').status is ParseStatus.MULTIPLE_TOOL_CALLS
    raw = '<tool_call>\n<function=alfworld_action>\n<parameter=action>\nlook\n</parameter>\n</function>\n</tool_call>'
    assert validate_tool_call(parse_tool_call(raw), ALFWorldToolRegistry(["go to cabinet 1"])).status is ValidationStatus.UNKNOWN_ACTION

def test_build_messages_contains_current_state_and_schema():
    messages, tools = build_messages(mission="put the apple in the drawer", observation="You see a drawer.", registry=ALFWorldToolRegistry(["open drawer 1"]))
    assert "put the apple" in messages[1]["content"]
    assert tools[0]["function"]["name"] == "alfworld_action"


def test_prompt_requires_qwen25_json_and_overrides_generic_template_guidance():
    from alfworld_baseline.prompts import PROMPT_VERSION, SYSTEM_PROMPT

    assert PROMPT_VERSION == "alfworld_qwen25_json_strict_v1"
    assert "STRICT QWEN2.5 TOOL-ONLY" in SYSTEM_PROMPT
    assert "take precedence over generic tool-use examples" in SYSTEM_PROMPT
    assert '"name":"alfworld_action"' in SYSTEM_PROMPT


def test_qwen35_prompt_profile_keeps_native_xml_contract():
    from alfworld_baseline.prompt_profiles import get_prompt_profile

    profile = get_prompt_profile("qwen35")
    assert profile.PROMPT_VERSION == "alfworld_qwen35_xml_strict_v2"
    assert "<function=alfworld_action>" in profile.SYSTEM_PROMPT
    assert "<parameter=action>" in profile.SYSTEM_PROMPT

def test_alfworld_agent_loop_marks_environment_terminal(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    async def fake_call(self, tool_call, tools_kwargs, agent_data):
        return "response", 1.0, {"done": True, "action": "look"}

    monkeypatch.setattr(ToolAgentLoop, "_call_tool", fake_call)
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    data = SimpleNamespace(data_source="alfworld", extra_fields={})
    response = asyncio.run(
        loop._call_tool(
            FunctionCall(name="alfworld_action", arguments='{"action":"look"}'),
            {},
            data,
        )
    )
    assert response[2]["done"] is True
    assert data.extra_fields["alfworld_environment_finished"] is True
    assert data.extra_fields["alfworld_terminal_reason"] == "done"


def test_alfworld_reward_sums_tool_rewards():
    from alfworld_baseline.reward import compute_score

    assert compute_score("alfworld", extra_info={"tool_rewards": [0.0, 1.0]}) == 1.0


def test_alfworld_loop_uses_isolated_protocol_penalties():
    from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop

    assert ALFWorldToolAgentLoop.NO_TOOL_CALL_PENALTY == -0.05
    assert ALFWorldToolAgentLoop.INVALID_TOOL_CALL_PENALTY == -0.05
    assert ALFWorldToolAgentLoop.MALFORMED_TOOL_CALL_PENALTY == -0.05
    assert ALFWorldToolAgentLoop.FORGED_ROLE_AFTER_TOOL_CALL_PENALTY == 0.0
    assert ALFWorldToolAgentLoop.TRAJECTORY_REPEAT_PENALTY_SCALE == 0.0


def test_invalid_action_has_isolated_single_step_penalty(monkeypatch):
    import asyncio
    import pandas as pd
    from alfworld_baseline.alfworld_tool import ALFWorldTool

    row = pd.read_parquet(ROOT / "data" / "qwen25_1_5b" / "test.parquet").iloc[0].to_dict()
    game_file = row["extra_info"]["game_file"]
    monkeypatch.setenv(
        "ALFWORLD_DATA",
        str(ROOT.parent.parent / "contrib" / "recipes" / "envs" / "agl_envs" / "alfworld" / "alfworld_source"),
    )

    async def run():
        tool = ALFWorldTool({"max_steps": 2, "invalid_action_penalty": -0.05})
        instance, _ = await tool.create(create_kwargs={"game_file": game_file})
        try:
            _, reward, metrics = await tool.execute(instance, {"action": "not admissible"})
            return reward, metrics
        finally:
            await tool.release(instance)

    reward, metrics = asyncio.run(run())
    assert reward == -0.05
    assert metrics["error"] == "invalid_action"


def test_alfworld_penalty_metrics_count_explicit_reasons():
    from types import SimpleNamespace
    from alfworld_baseline.metrics import compute_alfworld_penalty_metrics

    batch = SimpleNamespace(
        non_tensor_batch={
            "penalty_records": [
                [
                    {"reason": "no_tool_call", "value": -0.05, "occurrences": 1},
                    {"reason": "malformed_tool_call_xml", "value": -0.05, "occurrences": 1},
                    {"reason": "format_error", "value": -0.05, "occurrences": 1},
                    {"reason": "invalid_action", "value": -0.05, "occurrences": 1},
                ],
                [{"reason": "unknown_tool_name", "value": -0.05, "occurrences": 1}],
            ]
        }
    )
    metrics = compute_alfworld_penalty_metrics(batch)
    assert metrics["alfworld_penalty/no_tool_call_count"] == 1
    assert metrics["alfworld_penalty/malformed_tool_call_count"] == 1
    assert metrics["alfworld_penalty/format_error_count"] == 1
    assert metrics["alfworld_penalty/invalid_action_count"] == 1
    assert metrics["alfworld_penalty/unknown_tool_count"] == 1
    assert metrics["alfworld_penalty/total_count"] == 5
    assert metrics["alfworld_penalty/total_value"] == -0.25


def test_dataset_loader_accepts_verl_nested_game_file(tmp_path):
    import pandas as pd
    from alfworld_baseline.datasets import load_tasks

    source = pd.read_parquet(ROOT / "data" / "qwen25_1_5b" / "test.parquet").iloc[[0]]
    path = tmp_path / "sample.parquet"
    source.to_parquet(path, index=False)
    assert load_tasks(path, limit=1)[0]["data_source"] == "alfworld"
