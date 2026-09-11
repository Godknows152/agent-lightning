"""CPU v5 regressions: no schema, action history, exact replay and exclusive penalties."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from alfworld_baseline.prompts_qwen35 import (
    PROMPT_VERSION, QWEN35_ALFWORLD_CHAT_TEMPLATE, build_messages, build_user_prompt,
)
from alfworld_baseline.text_actions import parse_text_action
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.tools.schemas import ToolResponse
from test_decision_budget import make_data, make_loop, set_server


@pytest.mark.parametrize('text,expected', [
    ('think</think>\nAction: look', 'look'),
    ('<think>Action: forbidden</think>\nlook<|im_end|>', 'look'),
    ('<tool_call>example</tool_call></think>\nAction: open drawer 1', 'open drawer 1'),
    ('plan</think>Action: nonexistent command', 'nonexistent command'),
    ('plan</think>Action: look<|im_end|><|endoftext|>', 'look'),
    ('Action: look', None),  # thinking never closed, cannot execute reasoning
    ('</think>', None),
    ('</think>Action:', None),
    ('</think>Action: look\nAction: look', None),
    ('</think>Action: look\nexplanation', None),
    ('</think><tool_call><function=alfworld_action>look</function></tool_call>', None),
    ('</think>{"action": "look"}', None),
    ('</think>```\nlook\n```', None),
])
def test_parse_text_action(text, expected):
    assert parse_text_action(text) == expected


def test_v5_template_and_messages_have_no_tool_schema():
    messages, tools = build_messages(mission='find apple', observation='drawer',
                                    registry=ALFWorldToolRegistry(['look']))
    assert tools and "enum" not in str(tools)
    assert PROMPT_VERSION == 'alfworld_qwen35_v7_compact_xml_history_thinking'
    rendered = Environment().from_string(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(
        # Even an accidental tools argument must never leak native schema.
        tools=[{'function': {'name': 'alfworld_action'}}], messages=messages,
        add_generation_prompt=True, enable_thinking=True,
    )
    assert rendered.endswith('<think>\n')
    assert 'alfworld_action(action: string)' in rendered
    assert rendered.count('<tools>') == 1 and '<tool_call>' in rendered
    assert '<parameter=action>' in rendered
    assert 'only the next action, not a complete plan' in rendered
    assert '1-2 short sentences' in rendered
    assert 'enumerate available actions' in rendered
    prompt = build_user_prompt(mission='goal', observation='new state', admissible_actions=['look'],
                               history=['open drawer 1 [executed]', 'look [executed]'])
    assert '1. open drawer 1 [executed]\n2. look [executed]' in prompt


def test_text_decisions_history_penalties_and_unmodified_replay():
    class Tool:
        async def execute(self, instance, params, **kwargs):
            action = params['action']
            metrics = {'action': action, 'observation': f'result of {action}',
                       'admissible_commands': ['look', 'inventory']}
            if action not in metrics['admissible_commands']:
                metrics['error'] = 'invalid_action'
            return ToolResponse(text='result'), 0.0, metrics

    async def run():
        loop = make_loop(max_steps=8)
        loop._text_actions = loop._thinking_enabled = True
        loop.tools = {'alfworld_action': Tool()}
        loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
        data = make_data(loop)
        data.alfworld_current_actions = ('look', 'inventory')
        data._active_tool_schemas = []
        texts = ['look', 'look', 'inventory', 'look', 'not admissible', '', 'look', 'look']
        all_ids = []
        for i, action in enumerate(texts):
            output = f'Example <tool_call> in reasoning.\n</think>\nAction: {action}'
            set_server(loop, output)
            state = await loop._generate_environment_decision(data, {})
            assert state == AgentState.PROCESSING_TOOLS
            ids = loop.tokenizer.encode(output)
            all_ids += ids
            replay = data.extra_fields['alfworld_turn_contexts'][-1]
            assert replay['response_ids'] == ids
            assert replay['response_offset'] == len(all_ids) - len(ids)
            state = await loop._process_environment_decision(data)
            assert state == (AgentState.TERMINATED if i == 5 else AgentState.GENERATING)
            assert data._active_tool_schemas == []
            if i < 5:
                assert 'Previous actions' in data.messages[0]['content']
                assert data.alfworld_decision_history[-1] in data.messages[0]['content']
                assert loop.apply_chat_template.call_args.kwargs['tools'] == []
            if state == AgentState.TERMINATED:
                break
        assert data.extra_fields["alfworld_terminal_reason"] == "no_tool_call"
        assert data.prompt_ids[2:] == all_ids
        assert data.response_mask == [1] * len(all_ids)
        assert data.response_logprobs == [-0.1] * len(all_ids)
        assert data.extra_fields['alfworld_repeated_action_penalty_count'] == 1
        assert data.extra_fields['alfworld_invalid_tool_call_penalty_count'] == 1
        assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1
        assert data.extra_fields['alfworld_valid_tool_call_count'] == 4
        assert sum(data.tool_rewards) == pytest.approx(-5.2)
        assert len(data.alfworld_decision_history) == 6
        assert data.alfworld_decision_history[4].endswith('[rejected]')
        assert data.alfworld_decision_history[5].endswith('[no action]')
    asyncio.run(run())


def test_v5_initial_state_has_no_schema_or_stale_history():
    class Tool:
        def get_state(self, instance):
            return 'room\nYour task is to: find apple.', ('look',)

    async def run():
        loop = make_loop()
        loop._text_actions = True
        loop.tools = {'alfworld_action': Tool()}
        loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
        data = make_data(loop)
        data.alfworld_decision_history = ['stale history']
        await loop._set_authoritative_initial_prompt(data)
        assert data._active_tool_schemas == []
        assert data.alfworld_decision_history == []
        assert 'stale' not in data.messages[0]['content']
        assert '(none)' in data.messages[0]['content']
    asyncio.run(run())
