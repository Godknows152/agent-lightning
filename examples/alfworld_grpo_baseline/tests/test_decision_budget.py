"""CPU regressions for environment-driven, per-decision ALFWorld generation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.budget import ALFWorldDecisionBudget, configure_environment_driven_rollout
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

ROOT = Path(__file__).resolve().parents[1]
CALL = '<tool_call>\n<function=alfworld_action>\n<parameter=action>\nlook\n</parameter>\n</function>\n</tool_call>'


class Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class Parser:
    async def extract_tool_calls(self, ids, tools):
        text = Tokenizer().decode(ids)
        if text.endswith('</tool_call>'):
            return '', [FunctionCall(name='alfworld_action', arguments='{"action":"look"}')]
        return '', []


def make_loop(max_steps=50, per_turn=256):
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._environment_budget = ALFWorldDecisionBudget(max_steps, per_turn)
    loop.response_length = max_steps * per_turn
    # These legacy cutoffs must not influence the opted-in path.
    loop.max_generated_response_length = 1
    loop.max_assistant_turns = 1
    loop.max_user_turns = 1
    loop._thinking_enabled = False
    loop.tokenizer = Tokenizer()
    loop.tool_parser = Parser()
    loop.tool_schemas = [ALFWorldToolRegistry(('look',)).build_tool_schema()]
    loop.max_tool_response_length = 100000
    loop.tool_response_truncate_side = 'right'
    loop.tools = {}
    loop.current_restoration_prompt = None
    loop.apply_chat_template = AsyncMock(side_effect=lambda *args, **kwargs: [900, 901, 902])
    return loop


def make_data(loop):
    data = AgentData([], None, None, {}, 'test', {})
    data.data_source = 'alfworld'
    data.prompt_ids = [900, 901]
    data.generation_prompt_ids = [900, 901]
    data.alfworld_mission = 'finish task'
    data.alfworld_current_observation = 'Current room.'
    data.alfworld_current_actions = ('look',)
    data._active_tool_schemas = loop.tool_schemas
    data.extra_fields.update(alfworld_decision_steps=0, alfworld_terminal_reason='running')
    return data


def set_server(loop, text):
    ids = loop.tokenizer.encode(text)
    output = SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids), num_preempted=0, routed_experts=None)
    loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))


def test_full_50_step_run_retains_all_12800_tokens_and_releases_tool():
    """Exercise inherited run/finalization, not just individual state handlers."""
    loop = make_loop()
    # No calls: every failed decision must still be bounded and trainable.
    set_server(loop, 'x' * 256)
    loop.process_vision_info = AsyncMock(return_value={})
    loop._release_tool_instances = AsyncMock()

    async def initial(data):
        data.alfworld_mission = 'finish task'
        data.alfworld_current_observation = 'Current room.'
        data.alfworld_current_actions = ('look',)
        data._active_tool_schemas = loop.tool_schemas

    loop._set_authoritative_initial_prompt = initial
    output = asyncio.run(loop.run({}, raw_prompt=[{'role': 'user', 'content': 'task'}], data_source='alfworld'))
    assert len(output.response_ids) == 12800
    assert len(output.response_logprobs) == 12800
    assert output.response_mask == [1] * 12800
    assert len(output.extra_fields['alfworld_turn_contexts']) == 50
    assert output.extra_fields['alfworld_turn_contexts'][-1]['response_offset'] == 49 * 256
    assert output.extra_fields['alfworld_decision_steps'] == 50
    assert output.extra_fields['alfworld_terminal_reason'] == 'max_steps'
    assert 'penalty_records' not in output.extra_fields
    assert output.extra_fields['alfworld_no_tool_call_penalty_count'] == 50
    assert output.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == 0
    assert output.extra_fields['alfworld_valid_tool_call_count'] == 0
    assert output.extra_fields['tool_rewards'] == [-0.1] * 50
    assert sum(output.extra_fields['tool_rewards']) == pytest.approx(-5.0)
    assert loop.server_manager.generate.await_count == 50
    for call in loop.server_manager.generate.await_args_list:
        assert call.kwargs['sampling_params']['max_new_tokens'] == 256
        assert call.kwargs['prompt_ids'] == [900, 901, 902]
    loop._release_tool_instances.assert_awaited_once()

    # Verify packed model-only loss slots still replay every real prompt,
    # including the final token beyond the historical 4096 capacity.
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.turn_context import expand_turn_contexts

    batch = TensorDict({}, batch_size=[1])
    batch["input_ids"] = torch.nested.as_nested_tensor(
        [torch.tensor(output.prompt_ids + output.response_ids)], layout=torch.jagged
    )
    batch["loss_mask"] = torch.ones((1, 12800))
    batch["temperature"] = torch.tensor([1.0])
    tu.assign_non_tensor(batch, alfworld_turn_contexts=[output.extra_fields["alfworld_turn_contexts"]])
    expanded, source, target = expand_turn_contexts(batch)
    assert len(expanded["input_ids"].unbind()) == 50
    assert len(source) == len(target) == 12800
    assert target[-1].item() == len(output.prompt_ids) + 12800 - 2


@pytest.mark.parametrize('text', ['x' * 256, '<tool_call>' + 'x' * 245])
def test_per_turn_exhaustion_retries_and_does_not_end_trajectory(text):
    loop = make_loop(max_steps=2)
    data = make_data(loop)
    set_server(loop, text)
    params = {'max_tokens': 1, 'max_new_tokens': 1, 'temperature': 0.7}
    assert asyncio.run(loop._handle_generating_state(data, params)) == AgentState.PROCESSING_TOOLS
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.GENERATING
    sent = loop.server_manager.generate.await_args.kwargs['sampling_params']
    assert sent == {'max_new_tokens': 256, 'temperature': 0.7}
    assert params['max_new_tokens'] == 1  # caller's shared parameters unchanged
    assert 'penalty_records' not in data.extra_fields
    assert asyncio.run(loop._handle_generating_state(data, params)) == AgentState.PROCESSING_TOOLS
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.TERMINATED
    assert data.extra_fields['alfworld_terminal_reason'] == 'max_steps'
    assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 2
    assert data.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == 0
    assert data.tool_rewards == [-0.1, -0.1]


@pytest.mark.parametrize('done_at', [1, 3, None])
def test_valid_actions_finish_only_on_done_or_max_steps_and_ignore_long_observations(done_at):
    loop = make_loop(max_steps=3)
    data = make_data(loop)
    set_server(loop, CALL)
    executed = []

    class Tool:
        async def execute(self, instance, args, **kwargs):
            executed.append(args['action'])
            return ToolResponse(text='observation' * 5000), 1.0 if len(executed) == done_at else 0.0, {
                'action': 'look', 'done': len(executed) == done_at,
                'observation': 'observation' * 5000, 'admissible_commands': ['look'],
            }

    loop.tools = {'alfworld_action': Tool()}
    loop._get_or_create_tool_instance = AsyncMock(return_value='instance')

    async def run():
        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            state = await loop._handle_generating_state(data, {})
            assert state == AgentState.PROCESSING_TOOLS
            state = await loop._handle_processing_tools_state(data)

    asyncio.run(run())
    n = done_at or 3
    assert len(executed) == n
    assert len(data.response_mask) == n * len(CALL)
    assert data.extra_fields['alfworld_terminal_reason'] == ('done' if done_at else 'max_steps')
    assert sum(data.tool_rewards) == pytest.approx((1.0 if done_at else 0.0) - 0.1 * max(n - 1, 0))
    assert data.extra_fields['alfworld_valid_tool_call_count'] == n
    assert data.successful_action_history == ['look'] * n
    assert len(data.extra_fields['alfworld_turn_contexts']) == n


@pytest.mark.parametrize('name,args', [
    ('unknown_tool', '{"action":"look"}'),
    ('alfworld_action', '{broken'),
    ('alfworld_action', '{"wrong":"look"}'),
    ('alfworld_action', '{"action":"bad"}'),
])
def test_invalid_calls_consume_exactly_one_decision_without_synthetic_reward(name, args):
    from alfworld_baseline.alfworld_tool import ALFWorldTool

    loop = make_loop(max_steps=2)
    data = make_data(loop)
    data.response_ids = loop.tokenizer.encode(CALL)
    data.assistant_turns = 1
    tool = ALFWorldTool({'max_steps': 2})
    tool._instances['instance'] = {
        'env': SimpleNamespace(step=lambda *_: pytest.fail('invalid action reached env.step')),
        'observation': 'room', 'info': {'admissible_commands': [['look']]}, 'steps': 0,
    }
    loop.tools = {'alfworld_action': tool}
    loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
    for index in range(2):
        data.tool_calls = [FunctionCall(name=name, arguments=args)]
        state = asyncio.run(loop._handle_processing_tools_state(data))
        assert state == (AgentState.GENERATING if index == 0 else AgentState.TERMINATED)
    assert data.extra_fields['alfworld_decision_steps'] == 2
    assert data.tool_rewards == [-0.1, -0.1]
    assert data.extra_fields.get('alfworld_no_tool_call_penalty_count', 0) == 0
    assert data.extra_fields['alfworld_invalid_tool_call_penalty_count'] == 2
    assert data.extra_fields.get('alfworld_valid_tool_call_count', 0) == 0
    assert 'penalty_records' not in data.extra_fields
    assert tool._instances['instance']['steps'] == 0


def test_complete_call_at_exact_per_turn_limit_still_executes():
    loop = make_loop(max_steps=2, per_turn=len(CALL))
    data = make_data(loop)
    set_server(loop, CALL)
    assert asyncio.run(loop._handle_generating_state(data, {})) == AgentState.PROCESSING_TOOLS
    assert data.extra_fields['alfworld_decision_steps'] == 0


def test_penalty_categories_are_mutually_exclusive_and_accumulate_across_steps():
    """A no-call step and a parsed-invalid step each contribute exactly once."""
    loop = make_loop(max_steps=2)
    data = make_data(loop)

    # Category 1: parser produced no call.
    data.tool_calls = []
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.GENERATING

    # Category 2: a parsed call names an unavailable tool.
    data.tool_calls = [FunctionCall(name='unknown_tool', arguments='{"action":"look"}')]
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.TERMINATED

    assert data.tool_rewards == [-0.1, -0.1]
    assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1
    assert data.extra_fields['alfworld_invalid_tool_call_penalty_count'] == 1
    assert data.extra_fields.get('alfworld_valid_tool_call_count', 0) == 0


def test_incomplete_xml_envelope_is_no_tool_call_not_invalid_tool_call():
    loop = make_loop(max_steps=1)
    data = make_data(loop)
    set_server(loop, CALL.replace('</function>\n', ''))

    assert asyncio.run(loop._handle_generating_state(data, {})) == AgentState.PROCESSING_TOOLS
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.TERMINATED
    assert data.tool_rewards == [-0.1]
    assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1
    assert data.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == 0


@pytest.mark.parametrize('text', ['', 'x' * 257])
def test_generation_contract_errors_fail_loudly(text):
    loop = make_loop()
    set_server(loop, text)
    with pytest.raises(RuntimeError, match='generation must return'):
        asyncio.run(loop._handle_generating_state(make_data(loop), {}))


def test_undersized_storage_fails_instead_of_silently_truncating():
    loop = make_loop()
    loop.response_length = 1
    set_server(loop, CALL)
    with pytest.raises(RuntimeError, match='refusing to truncate'):
        asyncio.run(loop._handle_generating_state(make_data(loop), {}))


@pytest.mark.parametrize('key,value', [('max_steps', 0), ('max_steps', -1), ('max_new_tokens_per_turn', 0), ('max_steps', True)])
def test_invalid_budget_rejected(key, value):
    with pytest.raises(ValueError, match='positive integer'):
        ALFWorldDecisionBudget.from_tool_config({'environment_driven': True, key: value})


@pytest.mark.parametrize('profile,expected_steps', [('qwen35_2b', 16), ('qwen35_9b', 50)])
def test_composed_config_disables_thinking_and_derives_storage_from_shared_tool_budget(profile, expected_steps):
    path = ROOT / 'config' / 'alfworld' / profile / 'v1'
    with initialize_config_dir(config_dir=str(path), version_base=None):
        config = compose(config_name='alfworld_config_2gpu')
    budget = configure_environment_driven_rollout(config)
    assert config.data.apply_chat_template_kwargs.enable_thinking is False
    assert config.trainer.enable_penalty_logging is False
    assert budget == ALFWorldDecisionBudget(expected_steps, 256)
    assert config.data.max_response_length == expected_steps * 256
    assert config.actor_rollout_ref.rollout.response_length == expected_steps * 256
    assert config.actor_rollout_ref.rollout.multi_turn.max_generated_response_length is None
    assert config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns is None


def test_storage_resizes_when_tool_budget_changes(tmp_path):
    tool_path = tmp_path / 'tool.yaml'
    OmegaConf.save(OmegaConf.create({'tools': [{
        'class_name': 'alfworld_baseline.alfworld_tool.ALFWorldTool',
        'config': {'environment_driven': True, 'max_steps': 7, 'max_new_tokens_per_turn': 123},
    }]}), tool_path)
    config = OmegaConf.create({
        'data': {'max_response_length': 1},
        'actor_rollout_ref': {
            'model': {'use_remove_padding': True},
            'rollout': {'multi_turn': {'tool_config_path': str(tool_path)}},
        },
    })
    assert configure_environment_driven_rollout(config).response_capacity == 861
    assert config.data.max_response_length == 861
    assert config.actor_rollout_ref.rollout.response_length == 861


def test_terminal_metrics_distinguish_environment_done_and_max_steps():
    from alfworld_baseline.metrics import compute_alfworld_rollout_metrics

    batch = SimpleNamespace(non_tensor_batch={"alfworld_terminal_reason": ["done", "max_steps", "max_steps"]})
    metrics = compute_alfworld_rollout_metrics(batch)
    assert metrics["alfworld_termination/done_count"] == 1
    assert metrics["alfworld_termination/max_steps_count"] == 2


def test_invalid_decisions_participate_in_tool_phase_without_deadlock():
    from verl.experimental.agent_loop.agent_loop import _GenerationToolPhaseCoordinator

    async def run():
        coordinator = _GenerationToolPhaseCoordinator(2)
        jobs = []
        for _ in range(2):
            loop = make_loop(max_steps=2)
            set_server(loop, "no call")
            loop.process_vision_info = AsyncMock(return_value={})
            loop._release_tool_instances = AsyncMock()

            async def initial(data):
                data.alfworld_mission = "task"
                data.alfworld_current_observation = "room"
                data.alfworld_current_actions = ("look",)

            loop._set_authoritative_initial_prompt = initial
            jobs.append(loop.run(
                {}, raw_prompt=[{"role": "user", "content": "task"}],
                data_source="alfworld", phase_coordinator=coordinator,
            ))
        return await asyncio.wait_for(asyncio.gather(*jobs), timeout=3)

    outputs = asyncio.run(run())
    assert [out.extra_fields["alfworld_decision_steps"] for out in outputs] == [2, 2]


def test_multiple_generated_calls_are_trimmed_and_feedback_precedes_next_generation():
    """Flattened rollout output is not evidence of concurrent tool execution."""
    loop = make_loop(max_steps=2)
    data = make_data(loop)
    set_server(loop, CALL + CALL)
    events = []
    original_generate = loop.server_manager.generate

    async def generate(*args, **kwargs):
        events.append("generate")
        if len(events) > 1:
            assert data.alfworld_current_observation == "feedback after action 1"
        return await original_generate(*args, **kwargs)

    loop.server_manager.generate = generate

    class Tool:
        async def execute(self, instance, args, **kwargs):
            events.append("tool")
            return ToolResponse(text="feedback"), 0.0, {
                "action": "look", "done": False,
                "observation": "feedback after action 1", "admissible_commands": ["look"],
            }

    loop.tools = {"alfworld_action": Tool()}
    loop._get_or_create_tool_instance = AsyncMock(return_value="instance")

    async def run():
        for _ in range(2):
            assert await loop._handle_generating_state(data, {}) == AgentState.PROCESSING_TOOLS
            assert loop.tokenizer.decode(data.response_ids) == CALL
            await loop._handle_processing_tools_state(data)

    asyncio.run(run())
    assert events == ["generate", "tool", "generate", "tool"]
    assert data.total_tool_calls == 2
    assert len(data.extra_fields["alfworld_turn_contexts"]) == 2
    assert loop.tokenizer.decode(data.prompt_ids[2:]) == CALL + CALL


def test_repeated_action_penalties_are_trajectory_local_and_mutually_exclusive():
    loop = make_loop(max_steps=10)
    data = make_data(loop)
    loop._get_or_create_tool_instance = AsyncMock(return_value="instance")

    class Tool:
        async def execute(self, instance, args, **kwargs):
            action = args["action"]
            return ToolResponse(text="feedback"), 1.0, {
                "action": action, "error": "invalid_action" if action == "bad" else None,
                "observation": "room", "admissible_commands": ["look", "inventory", "bad"],
            }

    loop.tools = {"alfworld_action": Tool()}
    data.alfworld_current_actions = ("look", "inventory", "bad")

    async def step(target, action):
        target.tool_calls = [] if action is None else [
            FunctionCall(name="alfworld_action", arguments='{"action":"' + action + '"}')
        ]
        return await loop._process_environment_decision(target)

    for action in ["look", "inventory", "look", "bad", None, "look", "inventory", "bad"]:
        asyncio.run(step(data, action))
    assert data.tool_rewards == pytest.approx([1, 1, .9, -.1, -.1, .9, .9, -.1])
    assert data.extra_fields["alfworld_repeated_action_penalty_count"] == 3
    assert data.extra_fields["alfworld_invalid_tool_call_penalty_count"] == 2
    assert data.extra_fields["alfworld_no_tool_call_penalty_count"] == 1
    assert data.alfworld_action_occurrences == {"look": 3, "inventory": 2}
    other = make_data(loop)
    asyncio.run(step(other, "look"))
    assert other.tool_rewards == [1.0]
    assert other.extra_fields.get("alfworld_repeated_action_penalty_count", 0) == 0


@pytest.mark.parametrize("prior", [1, 2, 4, 5, 6, 15])
def test_repeat_penalty_is_fixed_for_every_repeated_occurrence(prior):
    loop = make_loop()
    data = make_data(loop)
    value = loop._record_alfworld_penalty(data, "repeated_action", append_reward=True, prior_occurrences=prior)
    assert value == pytest.approx(-0.1)
    assert data.tool_rewards == pytest.approx([-0.1])
    assert data.extra_fields["alfworld_repeated_action_penalty_count"] == 1

def test_sixteen_identical_actions_match_no_call_trajectory_penalty():
    loop = make_loop()
    data = make_data(loop)
    penalties = [loop._record_alfworld_penalty(data, "repeated_action", append_reward=False,
                                              prior_occurrences=i) for i in range(1, 16)]
    assert sum(penalties) == pytest.approx(-1.5)
    assert sum(penalties) == pytest.approx(15 * loop.ALFWORLD_NO_TOOL_CALL_PENALTY)
