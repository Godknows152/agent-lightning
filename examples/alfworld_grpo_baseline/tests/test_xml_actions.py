"""v7 schema classification, exclusive penalties and exact rollout replay."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from alfworld_baseline.xml_actions import parse_xml_decision
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.tools.schemas import ToolResponse
from test_decision_budget import make_data, make_loop, set_server


def call(action='look', name='alfworld_action'):
    return f'<tool_call><function={name}><parameter=action>{action}</parameter></function></tool_call>'


@pytest.mark.parametrize('output,status,reason', [
    ('still thinking ' + call(), 'no_action', 'unclosed_thinking'),
    ('</think>Action: look', 'no_action', 'missing_or_incomplete_call'),
    ('</think>' + call()[:-12], 'no_action', 'missing_or_incomplete_call'),
    ('</think>' + call().replace('</parameter>', ''), 'no_action', 'incomplete_parameter'),
    ('</think>' + call().replace('</function>', ''), 'no_action', 'malformed_xml'),
    ('</think>' + call(name='other'), 'invalid_action', 'unknown_tool'),
    ('</think>' + call(''), 'invalid_action', 'invalid_action_value'),
    ('</think>' + call('look\ninventory'), 'invalid_action', 'invalid_action_value'),
    ('</think>' + call().replace('<parameter=action>look</parameter>', ''), 'invalid_action', 'invalid_arguments_schema'),
    ('</think>' + call().replace('</function>', '<parameter=extra>x</parameter></function>'), 'invalid_action', 'invalid_arguments_schema'),
    ('</think>' + call().replace('</function>', '<parameter=action>inventory</parameter></function>'), 'invalid_action', 'invalid_arguments_schema'),
    ('</think>' + call() + call(), 'invalid_action', 'multiple_calls'),
    ('</think>' + call() + '<tool_call>', 'invalid_action', 'multiple_calls'),
    ('</think>explanation\n' + call(), 'invalid_action', 'text_outside_call'),
    ('</think>' + call() + '\nexplanation', 'invalid_action', 'text_outside_call'),
    ('An XML example: ' + call('inventory') + '</think>\n' + call() + '<|im_end|>', 'valid', 'parsed'),
])
def test_xml_classification_and_penalty(output, status, reason):
    result = parse_xml_decision(output)
    assert (result.status, result.reason) == (status, reason)
    loop = make_loop(max_steps=4, per_turn=1024)
    loop._xml_actions = loop._thinking_enabled = True
    loop._text_actions = False
    data = make_data(loop)
    execute = AsyncMock(return_value=(ToolResponse(text='room'), 0.0, {
        'action': 'look', 'admissible_commands': ['look'], 'observation': 'room'
    }))
    loop.tools = {'alfworld_action': type('Tool', (), {'execute': execute})()}
    loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
    set_server(loop, output)

    async def run():
        assert await loop._generate_environment_decision(data, {}) == AgentState.PROCESSING_TOOLS
        return await loop._process_environment_decision(data)

    state = asyncio.run(run())
    assert state == (AgentState.TERMINATED if status == 'no_action' else AgentState.GENERATING)
    assert data.tool_rewards == ([0.0] if status == 'valid' else [-5.0 if status == 'no_action' else -0.1])
    assert execute.await_count == int(status == 'valid')
    assert data.extra_fields.get('alfworld_no_tool_call_penalty_count', 0) == int(status == 'no_action')
    assert data.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == int(status == 'invalid_action')
    assert data.extra_fields.get('alfworld_repeated_action_penalty_count', 0) == 0
    assert data.extra_fields['alfworld_last_decision_reason'] == ('executed' if status == 'valid' else reason)
    assert data.extra_fields['alfworld_decision_steps'] == 1
    assert data.extra_fields['alfworld_turn_contexts'][0]['response_ids'] == loop.tokenizer.encode(output)
    assert data.response_logprobs == [-0.1] * len(output)
    assert data.response_mask == [1] * len(output)
    if status == 'invalid_action':
        assert data.alfworld_decision_history[-1].endswith('[rejected]')


def test_xml_repeat_compares_command_and_schema_failure_resets_streak():
    loop = make_loop(max_steps=10, per_turn=512)
    loop._xml_actions = loop._thinking_enabled = True
    loop._text_actions = False
    loop._get_or_create_tool_instance = AsyncMock(return_value='instance')

    class Tool:
        async def execute(self, instance, args, **kwargs):
            return ToolResponse(text='room'), 0.0, {
                'action': args['action'], 'observation': 'room',
                'admissible_commands': ['look', 'inventory'],
                'error': 'invalid_action' if args['action'] == 'bad' else None,
            }

    loop.tools = {'alfworld_action': Tool()}
    data = make_data(loop)
    outputs = [call('look'), call('inventory'), call('look'), call('look'),
               call('look') + call('look'), call('look'), call('bad'), call('look')]

    async def run():
        for output in outputs:
            set_server(loop, 'Choose.</think>' + output)
            await loop._generate_environment_decision(data, {})
            assert await loop._process_environment_decision(data) == AgentState.GENERATING

    asyncio.run(run())
    assert data.tool_rewards == [0, 0, 0, -.1, -.1, 0, -.1, 0]
    assert data.extra_fields['alfworld_repeated_action_penalty_count'] == 1
    assert data.extra_fields['alfworld_invalid_tool_call_penalty_count'] == 2
    assert data.extra_fields['alfworld_valid_tool_call_count'] == 6


@pytest.mark.parametrize("output,expected", [
    ("x" * 768, 1),
    ("x" * 767, 0),
    ("</think>" + "x" * 760, 0),
])
def test_thinking_truncation_requires_exhausted_budget_and_unclosed_thinking(output, expected):
    loop = make_loop(max_steps=2, per_turn=768)
    loop._xml_actions = loop._thinking_enabled = True
    loop._text_actions = False
    data = make_data(loop)
    set_server(loop, output)

    async def run():
        await loop._generate_environment_decision(data, {})
        return await loop._process_environment_decision(data)

    assert asyncio.run(run()) == AgentState.TERMINATED
    assert data.tool_rewards == [-5.0]
    assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1
    assert (data.extra_fields.get('alfworld_no_action_category') == 'overlong_thinking') == bool(expected)
