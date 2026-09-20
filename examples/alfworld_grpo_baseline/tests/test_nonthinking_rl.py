"""The 9B profile uses the shared GiGPO generation boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_gigpo import PROMPT_VERSION, QWEN3_ALFWORLD_CHAT_TEMPLATE
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
    assert loop._xml_actions is True
    assert loop._text_actions is False
    assert loop.apply_chat_template_kwargs['chat_template'] == QWEN3_ALFWORLD_CHAT_TEMPLATE


def test_9b_composed_config_uses_gigpo_thinking():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'config/alfworld/qwen35_9b/v1'), version_base=None):
        cfg = compose(config_name='alfworld_config_2gpu')
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is True
    assert cfg.variables.PROMPT_VERSION == PROMPT_VERSION


def test_gigpo_state_prompt_and_parser():
    loop = ALFWorldToolAgentLoop.__new__(ALFWorldToolAgentLoop)
    loop._thinking_enabled = True
    loop._text_actions = False
    loop._xml_actions = True
    loop._gigpo_prompt = True
    messages = loop._state_prompt_messages(mission='find a mug', observation='a room', actions=('look',),
                                          history=('inventory [executed]',))
    rendered = messages[0]['content']
    assert '<think>' in rendered and '<parameter=action>' in rendered
    assert '<action>' not in rendered
    assert 'inventory [executed]' in rendered
    call = '<think>look</think><tool_call><function=alfworld_action><parameter=action>look</parameter></function></tool_call>'
    assert parse_xml_decision(call).action == 'look'
