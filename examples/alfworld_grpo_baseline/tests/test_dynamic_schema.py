"""Regressions for per-instance schemas and optional legacy adapters.

The production ALFWorld path uses native veRL multi-turn storage.  Tests that
exercise the removed turn-context FSDP adapter remain conditional so this
suite can still be run against the historical shared backend when needed.
"""
import asyncio
from types import SimpleNamespace

import pytest

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.tools.schemas import OpenAIFunctionToolSchema



def test_qwen35_template_contains_compact_schema_without_action_enum():
    assert "<tools>" in QWEN35_ALFWORLD_CHAT_TEMPLATE
    assert "<tool_call>" in QWEN35_ALFWORLD_CHAT_TEMPLATE
    assert "<parameter=action>" in QWEN35_ALFWORLD_CHAT_TEMPLATE

    user_prompt = build_user_prompt(
        mission="find an apple", observation="A drawer.", admissible_actions=("look", "open drawer 1")
    )
    assert "<tool_call>" not in user_prompt
    assert "alfworld_action" not in user_prompt
    assert "Current observation:\nA drawer." in user_prompt
    assert "- look\n- open drawer 1" in user_prompt


def test_initial_schema_and_refresh_are_per_trajectory():
    class Tool:
        async def create(self, **kwargs):
            return kwargs["create_kwargs"]["game_file"], None

        def get_state(self, instance):
            return "A drawer.\nYour task is to: find an apple.", ("look", f"open {instance}")

    async def run():
        loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
        loop.tools = {"alfworld_action": Tool()}
        loop.tool_schemas = [ALFWorldToolRegistry.static_tool_schema()]
        captures = []

        async def template(messages, **kwargs):
            captures.append((messages, kwargs["tools"]))
            return [1, 2, 3]

        loop.apply_chat_template = template
        def data(game):
            try:
                result = AgentData(
                    messages=[{"role": "user", "content": "stale action"}],
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs={},
                    metrics={},
                    request_id=game,
                    tools_kwargs={"alfworld_action": {"create_kwargs": {"game_file": game}}},
                )
            except TypeError:
                # Historical shared backend AgentData has no audio/mm kwargs.
                result = AgentData(
                    [{"role": "user", "content": "stale action"}], None, None, {}, game,
                    {"alfworld_action": {"create_kwargs": {"game_file": game}}},
                )
            result.data_source = "alfworld"
            return result
        first, second = data("drawer 1"), data("cabinet 2")
        await asyncio.gather(loop._set_authoritative_initial_prompt(first), loop._set_authoritative_initial_prompt(second))
        def actions(item):
            return item._active_tool_schemas[0]["function"]["parameters"]["properties"]["action"]["enum"]
        assert actions(first) == ["look", "open drawer 1"]
        assert actions(second) == ["look", "open cabinet 2"]
        assert "stale action" not in first.messages[0]["content"]
        assert "Your task is" not in first.messages[0]["content"]
        assert "enum" not in loop.tool_schemas[0]["function"]["parameters"]["properties"]["action"]
        first.prompt_ids = [9, 8, 7]
        for index in range(5):
            first.alfworld_last_tool_metrics = {"action": f"action {index}", "observation": f"state {index}",
                                               "admissible_commands": ["look", f"take apple {index}"]}
            await loop._rebuild_generation_prompt_after_tool(first)
        assert actions(first) == ["look", "take apple 4"]
        assert actions(second) == ["look", "open cabinet 2"]
        assert first.alfworld_recent_history == []
        assert "Recent action/tool history" not in first.messages[0]["content"]
        assert "action 0" not in first.messages[0]["content"]
        assert "take apple 4" in first.messages[0]["content"]
        assert first.prompt_ids == [9, 8, 7]
        assert first.generation_prompt_ids == [1, 2, 3]
        typed = OpenAIFunctionToolSchema.model_validate(captures[-1][1][0])
        assert typed.function.parameters.properties["action"].enum == ["look", "take apple 4"]
        # Terminal environments may return an empty action list; no next prompt is needed.
        first.extra_fields["alfworld_environment_finished"] = True
        first.alfworld_last_tool_metrics = {"admissible_commands": []}
        await loop._rebuild_generation_prompt_after_tool(first)
    asyncio.run(run())


def test_qwen35_9b_uses_full_sft_model():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "config" / "alfworld" / "qwen35_9b" / "v1"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="alfworld_config_2gpu")

    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.model.lora_adapter_path is None
    assert config.actor_rollout_ref.model.lora.rank == 0
    assert config.actor_rollout_ref.model.path == config.variables.SFT_MODEL
    assert "skip_load_balancer_for_single_server" not in config.actor_rollout_ref.rollout.agent
    assert "normalize_non_tensor_batch_keys" not in config.actor_rollout_ref.rollout.agent
    assert config.actor_rollout_ref.model.enable_activation_offload is False
