"""Thinking stays trainable but cannot execute tools or incur prefix penalties."""
import asyncio
from types import SimpleNamespace

import pytest
from jinja2 import Environment

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE
from alfworld_baseline.thinking import ThinkingToolParser, tool_output

CALL = '<tool_call>\n<function=alfworld_action>\n<parameter=action>\nlook\n</parameter>\n</function>\n</tool_call>'


class Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


@pytest.mark.parametrize('thinking', [True, False])
def test_template_modes(thinking):
    rendered = Environment().from_string(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(
        tools=[{'function': {'name': 'alfworld_action'}}],
        messages=[{'role': 'user', 'content': 'Choose.'}],
        enable_thinking=thinking, add_generation_prompt=True,
    )
    assert rendered.endswith('<think>\n' if thinking else '<think>\n\n</think>\n\n')
    assert ('restrictions apply only after </think>' in rendered) == thinking


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
def test_extra_visible_output_still_penalized(extra):
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._thinking_enabled = True
    loop.tokenizer = Tokenizer()
    data = SimpleNamespace(data_source='alfworld', tool_rewards=[])
    records = []
    loop._record_penalty = lambda *args, **kwargs: records.append(kwargs)
    ids = loop.tokenizer.encode('Reason.\n</think>\n' + extra)
    result, logs = loop._apply_tool_call_format_guardrails(data, ids, [0.1] * len(ids))
    assert data.tool_rewards == [-0.05]
    assert len(records) == 1
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
