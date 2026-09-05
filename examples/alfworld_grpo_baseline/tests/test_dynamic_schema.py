"""Regressions for per-instance schemas and compact-context training replay."""
import asyncio
from types import SimpleNamespace

import pytest

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.tools.schemas import OpenAIFunctionToolSchema


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
        assert len(first.alfworld_recent_history) == 3
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
    batch["temperature"] = torch.tensor([1., 0.7])
    tu.assign_non_tensor(batch, alfworld_turn_contexts=[
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


def test_training_replay_fails_closed_on_missing_or_mismatched_context():
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts
    batch = make_replay_batch()
    batch["input_ids"].values()[2] = 99
    with pytest.raises(ValueError, match="does not match"):
        expand_turn_contexts(batch)
