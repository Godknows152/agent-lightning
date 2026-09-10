"""Thinking stays trainable; protocol penalties apply only after parsing output."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.budget import ALFWorldDecisionBudget
from alfworld_baseline.prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE
from alfworld_baseline.thinking import ThinkingToolParser, tool_output
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

CALL = '<tool_call>\n<function=alfworld_action>\n<parameter=action>\nlook\n</parameter>\n</function>\n</tool_call>'


class Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class ParsedCallParser:
    def __init__(self, call=None):
        self.call = call

    async def extract_tool_calls(self, ids, tools):
        return '', [] if self.call is None else [self.call]


def make_thinking_environment_loop(parser):
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._environment_budget = ALFWorldDecisionBudget(max_steps=1, max_new_tokens_per_turn=256)
    loop.response_length = 256
    loop._thinking_enabled = True
    loop.tokenizer = Tokenizer()
    loop.tool_parser = ThinkingToolParser(parser, loop.tokenizer)
    loop.tool_schemas = [ALFWorldToolRegistry(('look',)).build_tool_schema()]
    loop.max_tool_response_length = 100000
    loop.tool_response_truncate_side = 'right'
    loop.tools = {}
    loop.apply_chat_template = AsyncMock(return_value=[900, 901])
    return loop


def make_thinking_data(loop):
    data = AgentData([], None, None, {}, 'thinking-test', {})
    data.data_source = 'alfworld'
    data.prompt_ids = [900, 901]
    data.generation_prompt_ids = [900, 901]
    data.alfworld_mission = 'finish task'
    data.alfworld_current_observation = 'room'
    data.alfworld_current_actions = ('look',)
    data._active_tool_schemas = loop.tool_schemas
    data.extra_fields.update(alfworld_decision_steps=0, alfworld_terminal_reason='running')
    return data


def set_thinking_server(loop, text):
    output = SimpleNamespace(
        token_ids=loop.tokenizer.encode(text),
        log_probs=[-0.1] * len(text),
        num_preempted=0,
        routed_experts=None,
    )
    loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=output))


@pytest.mark.parametrize('thinking', [True, False])
def test_template_modes(thinking):
    rendered = Environment().from_string(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(
        tools=[{'function': {'name': 'alfworld_action'}}],
        messages=[{'role': 'user', 'content': 'Choose.'}],
        enable_thinking=thinking, add_generation_prompt=True,
    )
    assert rendered.endswith('<think>\n' if thinking else '<think>\n\n</think>\n\n')
    assert 'restrictions apply only after </think>' in rendered
    assert '<tools>' not in rendered and '<function=' not in rendered


@pytest.mark.parametrize('prefix', ['Choose look.\n</think>\n', '<think>Choose look.</think>\n', CALL + '\n</think>\n'])
def test_reasoning_is_not_format_error_and_original_tokens_are_kept(prefix):
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._thinking_enabled = True
    loop.tokenizer = Tokenizer()
    data = SimpleNamespace(data_source='alfworld', tool_rewards=[])
    loop._record_penalty = lambda *args, **kwargs: pytest.fail('unexpected penalty')
    ids = loop.tokenizer.encode(prefix + CALL)
    probs = [0.1] * len(ids)
    result, logs = loop._apply_tool_call_format_guardrails(data, ids, probs)
    assert result == ids
    assert logs == probs
    assert not data.tool_rewards


@pytest.mark.parametrize('extra', ['explanation\n' + CALL, CALL + 'suffix', CALL + CALL])
def test_extra_visible_output_is_not_penalized(extra):
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._thinking_enabled = True
    loop.tokenizer = Tokenizer()
    data = SimpleNamespace(data_source='alfworld', tool_rewards=[])
    records = []
    loop._record_penalty = lambda *args, **kwargs: records.append(kwargs)
    ids = loop.tokenizer.encode('Reason.\n</think>\n' + extra)
    result, logs = loop._apply_tool_call_format_guardrails(data, ids, [0.1] * len(ids))
    # Thinking-mode output is trainable as emitted; visible-prefix/suffix
    # protocol text must not add a synthetic reward term.
    assert data.tool_rewards == []
    assert records == []
    assert len(result) == len(logs)


def test_parser_never_executes_reasoning_examples_or_unclosed_thinking():
    tokenizer = Tokenizer()
    seen = []

    class Parser:
        async def extract_tool_calls(self, ids, tools):
            seen.append(tokenizer.decode(ids))
            return '', ['actual_call']

    parser = ThinkingToolParser(Parser(), tokenizer)
    assert asyncio.run(parser.extract_tool_calls(tokenizer.encode(CALL))) == ('', [])
    assert seen == []
    result = asyncio.run(parser.extract_tool_calls(tokenizer.encode(CALL + '</think>' + CALL)))
    assert result == ('', ['actual_call'])
    assert seen == [CALL]
    assert tool_output(CALL, enable_thinking=False) == (CALL, 0)


def test_thinking_output_without_post_think_tool_call_gets_no_tool_penalty():
    loop = make_thinking_environment_loop(ParsedCallParser())
    data = make_thinking_data(loop)
    set_thinking_server(loop, 'The answer is look.\n</think>\nNo tool call.')

    assert asyncio.run(loop._generate_environment_decision(data, {})) == AgentState.PROCESSING_TOOLS
    assert asyncio.run(loop._process_environment_decision(data)) == AgentState.TERMINATED
    assert data.tool_rewards == [-0.1]
    assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1
    assert data.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == 0
    assert data.extra_fields.get('alfworld_valid_tool_call_count', 0) == 0


def test_thinking_output_with_parsed_tool_call_has_no_protocol_penalty():
    loop = make_thinking_environment_loop(
        ParsedCallParser(FunctionCall(name='alfworld_action', arguments='{"action":"look"}'))
    )
    data = make_thinking_data(loop)

    class Tool:
        async def execute(self, instance, args, **kwargs):
            return ToolResponse(text='done'), 0.0, {'action': 'look', 'done': True}

    loop.tools = {'alfworld_action': Tool()}
    loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
    set_thinking_server(loop, 'Reason about the room.\n</think>\n' + CALL)

    assert asyncio.run(loop._generate_environment_decision(data, {})) == AgentState.PROCESSING_TOOLS
    assert asyncio.run(loop._process_environment_decision(data)) == AgentState.TERMINATED
    assert data.tool_rewards == [0.0]
    assert data.extra_fields.get('alfworld_no_tool_call_penalty_count', 0) == 0
    assert data.extra_fields.get('alfworld_invalid_tool_call_penalty_count', 0) == 0
    assert data.extra_fields['alfworld_valid_tool_call_count'] == 1
