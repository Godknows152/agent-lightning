"""The 9B non-thinking RL path must match its SFT generation boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from jinja2 import Template

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_qwen35 import NONTHINKING_PROMPT_VERSION, QWEN35_ALFWORLD_CHAT_TEMPLATE
from alfworld_baseline.xml_actions import parse_xml_decision
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


@pytest.mark.parametrize('thinking', [False, True])
def test_constructor_respects_config(monkeypatch, thinking):
    def init(self, *args, **kwargs):
        self.apply_chat_template_kwargs = {'enable_thinking': thinking}
        self.processor = None
        self.tool_parser = object()
        self.tokenizer = object()
        self.tools = {'alfworld_action': SimpleNamespace(config={'environment_driven': True, 'max_steps': 50, 'max_new_tokens_per_turn': 768})}
        self.response_length = 38400
    monkeypatch.setenv('ALFWORLD_MODEL_PROFILE', 'qwen35_9b')
    monkeypatch.setattr(ToolAgentLoop, '__init__', init)
    loop = ALFWorldToolAgentLoop()
    assert loop._thinking_enabled is thinking
    assert loop.apply_chat_template_kwargs['enable_thinking'] is thinking


def test_9b_composed_config_disables_thinking():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'config/alfworld/qwen35_9b/v1'), version_base=None):
        cfg = compose(config_name='alfworld_config_2gpu')
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is False
    assert cfg.variables.PROMPT_VERSION == NONTHINKING_PROMPT_VERSION


def test_nonthinking_state_prompt_and_parser():
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._thinking_enabled = False
    messages = loop._state_prompt_messages(mission='find a mug', observation='a room', actions=('look',),
                                          history=('inventory [executed]',))
    rendered = Template(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(messages=messages, enable_thinking=False,
                                                            add_generation_prompt=True)
    assert 'Keep thinking' not in rendered
    assert 'Close </think>' not in rendered
    assert 'Emit exactly one XML tool call directly' in rendered
    assert 'inventory [executed]' in rendered
    assert rendered.endswith('<think>\n\n</think>\n\n')
    call = '<tool_call><function=alfworld_action><parameter=action>look</parameter></function></tool_call>'
    assert parse_xml_decision(call, enable_thinking=loop._thinking_enabled).status == 'valid'
    assert parse_xml_decision('', enable_thinking=False).status == 'no_action'
