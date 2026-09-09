"""Regressions for per-instance schemas and compact-context training replay."""
import asyncio
from types import SimpleNamespace

import pytest

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.tools.schemas import OpenAIFunctionToolSchema



def test_qwen35_tool_protocol_has_one_concrete_xml_source():
    xml_example = (
        "<tool_call>\n<function=alfworld_action>\n<parameter=action>\n"
        "ACTION\n</parameter>\n</function>\n</tool_call>"
    )
    assert QWEN35_ALFWORLD_CHAT_TEMPLATE.count(xml_example.replace("\n", "\\n")) == 1
    assert "example_function_name" not in QWEN35_ALFWORLD_CHAT_TEMPLATE
    assert "Do not output ACTION literally" in QWEN35_ALFWORLD_CHAT_TEMPLATE

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
            result = AgentData([{"role": "user", "content": "stale action"}], None, None, {}, game,
                               {"alfworld_action": {"create_kwargs": {"game_file": game}}})
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


def make_replay_batch():
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    batch = TensorDict({}, batch_size=[2])
    # Prompt [10,11], answer [12,13], feedback [14], second answer [15,16].
    batch["input_ids"] = torch.nested.as_nested_tensor([torch.tensor([10,11,12,13,14,15,16]),
                                                       torch.tensor([20,21,22])], layout=torch.jagged)
    batch["loss_mask"] = torch.tensor([[1,1,0,1,1], [1,0,0,0,0]])
    batch["response_mask"] = torch.tensor([[1,1,0,1,1], [1,0,0,0,0]])
    batch["temperature"] = torch.tensor([1., 0.7])
    tu.assign_non_tensor(batch, multi_modal_inputs=[None, {}], alfworld_turn_contexts=[
        [{"prompt_ids": [10,11], "response_ids": [12,13], "response_offset": 0},
         {"prompt_ids": [30,31,32], "response_ids": [15,16], "response_offset": 3}],
        [{"prompt_ids": [20,21], "response_ids": [22], "response_offset": 0}],
    ])
    return batch


def test_training_replays_actual_prompts_and_preserves_loss_slots_and_gradients():
    import torch
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts, restore_trajectory_outputs
    batch = make_replay_batch()
    expanded, source, target = expand_turn_contexts(batch)
    assert [row.tolist() for row in expanded["input_ids"].unbind()] == [[10,11,12,13], [30,31,32,15,16], [20,21,22]]
    assert [row.tolist() for row in expanded["position_ids"].unbind()] == [[0,1,2,3], [0,1,2,3,4], [0,1,2]]
    assert source.tolist() == [1,2,6,7,10]
    assert target.tolist() == [1,2,4,5,8]
    values = torch.arange(12, dtype=torch.float32, requires_grad=True)
    out = {"log_probs": torch.nested.nested_tensor_from_jagged(values, expanded["input_ids"].offsets())}
    restored = restore_trajectory_outputs(out, batch, source, target)["log_probs"]
    assert restored.values().tolist() == [0,1,2,0,6,7,0,0,10,0]
    restored.values().sum().backward()
    assert values.grad.nonzero().flatten().tolist() == source.tolist()


def test_training_replay_keeps_uniform_temperature_as_scalar_for_fused_ppo():
    import torch
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts
    from verl.utils import tensordict_utils as tu

    batch = make_replay_batch()
    batch["temperature"] = torch.tensor([1.0, 1.0])
    expanded, _, _ = expand_turn_contexts(batch)

    assert tu.get_non_tensor_data(expanded, "temperature", None) == 1.0


def test_training_replay_rejects_nonempty_multimodal_payload():
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts
    batch = make_replay_batch()
    from verl.utils import tensordict_utils as tu
    tu.assign_non_tensor(batch, multi_modal_inputs=[{"pixel_values": [1]}, None])
    with pytest.raises(ValueError, match="non-empty multi_modal_inputs"):
        expand_turn_contexts(batch)


def test_training_replay_fails_closed_on_missing_or_mismatched_context():
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts
    batch = make_replay_batch()
    batch["input_ids"].values()[2] = 99
    with pytest.raises(ValueError, match="does not match"):
        expand_turn_contexts(batch)


def test_sglang_lora_target_modules_preserve_all_linear_sentinel():
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _normalize_sglang_lora_target_modules

    assert _normalize_sglang_lora_target_modules("all-linear") == ["all"]
    assert _normalize_sglang_lora_target_modules("all") == ["all"]
    assert _normalize_sglang_lora_target_modules("q_proj") == ["q_proj"]
    assert _normalize_sglang_lora_target_modules(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]


def test_qwen35_9b_lora_targets_are_supported_by_sglang_dynamic_loading():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "config" / "alfworld" / "qwen35_9b" / "v1"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="alfworld_config_2gpu")

    assert config.actor_rollout_ref.model.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert config.actor_rollout_ref.model.enable_activation_offload is False
