from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
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


def test_qwen35_prompt_profile_uses_text_actions():
    from alfworld_baseline.prompt_profiles import get_prompt_profile

    profile = get_prompt_profile("qwen35")
    assert profile.PROMPT_VERSION == "alfworld_qwen35_v7_compact_xml_history_thinking"
    assert profile.SYSTEM_PROMPT == ""
    user_prompt = profile.build_user_prompt(
        mission="put the apple in the drawer",
        observation="You see a drawer.",
        admissible_actions=["open drawer 1"],
    )
    assert "Task goal (not an executable action):" in user_prompt
    assert "Current admissible actions" in user_prompt
    assert "alfworld_action" not in user_prompt
    assert "Recent action/tool history" not in user_prompt
    assert "Your task is:" not in user_prompt

def test_qwen35_dynamic_schema_enum_matches_latest_admissible_actions():
    from alfworld_baseline.tool_registry import ALFWorldToolRegistry

    schema = ALFWorldToolRegistry(("look", "open drawer 1")).build_tool_schema()
    action = schema["function"]["parameters"]["properties"]["action"]
    assert action["enum"] == ["look", "open drawer 1"]


def test_qwen35_prompt_has_history_and_text_protocol():
    from alfworld_baseline.prompt_profiles import get_prompt_profile

    profile = get_prompt_profile("qwen35")
    user_prompt = profile.build_user_prompt(
        mission="put the apple in the drawer",
        observation="You see a drawer.\nYour task is: put the apple in the drawer.",
        admissible_actions=["open drawer 1"],
    )
    assert "Task goal (not an executable action):" in user_prompt
    assert "Your task is:" not in user_prompt
    assert "Recent action/tool history" not in user_prompt
    assert "example_function_name" not in profile.QWEN35_ALFWORLD_CHAT_TEMPLATE
    assert "If you choose to call a function" not in profile.QWEN35_ALFWORLD_CHAT_TEMPLATE


def test_alfworld_agent_loop_marks_environment_terminal(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    class FakeTool:
        async def execute(self, instance_id, parameters, **kwargs):
            return "response", 1.0, {"done": True, "action": "look"}

    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._get_or_create_tool_instance = lambda *args, **kwargs: _ready_instance()
    data = SimpleNamespace(data_source="alfworld", extra_fields={}, _active_tools={"alfworld_action": FakeTool()})

    async def _ready_instance():
        return "instance"

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


def test_alfworld_rollout_metrics_expose_three_penalty_series():
    from types import SimpleNamespace
    from alfworld_baseline.metrics import compute_alfworld_rollout_metrics

    metrics = compute_alfworld_rollout_metrics(
        SimpleNamespace(
            non_tensor_batch={
                "alfworld_terminal_reason": ["done", "max_steps", "max_steps"],
                "alfworld_no_tool_call_penalty_count": [2, 0, 1],
                "alfworld_invalid_tool_call_penalty_count": [0, 3, 1],
                "alfworld_repeated_action_penalty_count": [1, 2, 3],
                "alfworld_valid_tool_call_count": [4, 1, 2],
            }
        )
    )
    assert not any("tool_call" in key for key in metrics)
    assert metrics.pop("alfworld_penalty/thinking_truncated_no_action_count") == 0
    assert metrics == {
        "alfworld_termination/no_action_count": 0,
        "alfworld_termination/done_count": 1,
        "alfworld_termination/max_steps_count": 2,
        "alfworld_penalty/no_action_count": 3,
        "alfworld_penalty/invalid_action_count": 4,
        "alfworld_penalty/repeated_action_count": 6,
        "alfworld/valid_action_count/min": 1,
        "alfworld/valid_action_count/max": 4,
        "alfworld/valid_action_count/mean": 7 / 3,
    }
    assert set(metrics) <= {
        "alfworld_termination/no_action_count",
        "alfworld_termination/done_count",
        "alfworld_termination/max_steps_count",
        "alfworld_penalty/no_action_count",
        "alfworld_penalty/invalid_action_count",
        "alfworld_penalty/repeated_action_count",
        "alfworld/valid_action_count/min",
        "alfworld/valid_action_count/max",
        "alfworld/valid_action_count/mean",
    }


def test_alfworld_validation_does_not_select_legacy_num_turns():
    from verl.trainer.ppo.ray_trainer import _select_validation_turn_counts

    counts, are_tool_calls = _select_validation_turn_counts(
        {
            "data_source": np.array(["alfworld", "alfworld"], dtype=object),
            "__num_turns__": np.array([5, 7]),
            "tool_call_counts": np.array([3, 4]),
        }
    )
    assert counts is None
    assert are_tool_calls is False


def test_alfworld_reward_sums_tool_rewards():
    from alfworld_baseline.reward import compute_score

    assert compute_score("alfworld", extra_info={"tool_rewards": [0.0, 1.0]}) == 1.0
    assert compute_score("alfworld", extra_info={"tool_rewards": [-0.1, 1.0]}) == 0.9


def test_invalid_action_is_reported_to_agent_loop_without_tool_level_reward(monkeypatch):
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
        tool = ALFWorldTool({"max_steps": 2})
        instance, _ = await tool.create(create_kwargs={"game_file": game_file})
        try:
            _, reward, metrics = await tool.execute(instance, {"action": "not admissible"})
            return reward, metrics
        finally:
            await tool.release(instance)

    reward, metrics = asyncio.run(run())
    assert reward == 0.0
    assert metrics["error"] == "invalid_action"


def test_alfworld_tool_reuses_pooled_environment_without_sharing_state(monkeypatch):
    import asyncio
    from alfworld_baseline.alfworld_tool import ALFWorldTool

    class FakeBatch:
        def __init__(self):
            self.loaded = []

        def load(self, game_files):
            self.loaded = list(game_files)

        def reset(self):
            return ["initial observation"], {"admissible_commands": [["look"]]}

    class FakeEnv:
        def __init__(self):
            self.batch_env = FakeBatch()
            self.last_commands = None
            self.closed = False

        def close(self):
            self.closed = True

    fake = FakeEnv()
    game_file = "/tmp/example.tw-pddl"

    async def run():
        tool = ALFWorldTool({"env_pool_size": 1})
        monkeypatch.setattr(tool, "_build_env", lambda _: fake)
        first, _ = await tool.create(create_kwargs={"game_file": game_file})
        first_env = tool._instances[first]["env"]
        await tool.release(first)
        second, _ = await tool.create(create_kwargs={"game_file": game_file})
        second_env = tool._instances[second]["env"]
        await tool.release(second)
        return first_env, second_env, fake.batch_env.loaded

    first_env, second_env, loaded = asyncio.run(run())
    assert first_env is second_env
    assert loaded == [game_file]
    assert fake.closed is False


def test_dataset_loader_accepts_verl_nested_game_file(tmp_path):
    import pandas as pd
    from alfworld_baseline.datasets import load_tasks

    source = pd.read_parquet(ROOT / "data" / "qwen25_1_5b" / "test.parquet").iloc[[0]]
    path = tmp_path / "sample.parquet"
    source.to_parquet(path, index=False)
    assert load_tasks(path, limit=1)[0]["data_source"] == "alfworld"


def test_alfworld_metrics_include_thinking_truncation_no_action_count():
    from types import SimpleNamespace
    from alfworld_baseline.metrics import compute_alfworld_rollout_metrics
    metrics = compute_alfworld_rollout_metrics(SimpleNamespace(non_tensor_batch={
        'alfworld_terminal_reason': ['no_tool_call', 'no_tool_call', 'done'],
        'alfworld_no_tool_call_penalty_count': [1, 1, 0],
        'alfworld_invalid_tool_call_penalty_count': [0, 0, 0],
        'alfworld_repeated_action_penalty_count': [0, 0, 0],
        'alfworld_thinking_truncated_no_action': [1, 0, 0],
        'alfworld_valid_tool_call_count': [0, 0, 1],
    }))
    assert metrics['alfworld_penalty/thinking_truncated_no_action_count'] == 1
