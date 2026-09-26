"""GiGPO context must preserve the baseline's Qwen3 XML execution contract."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hydra import compose, initialize_config_dir
from jinja2 import Environment

from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
from alfworld_baseline.prompts_gigpo import PROMPT_VERSION, QWEN3_ALFWORLD_CHAT_TEMPLATE, build_messages, build_user_prompt
from alfworld_baseline.tool_registry import ALFWorldToolRegistry
from alfworld_baseline.xml_actions import parse_xml_decision
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse
from test_decision_budget import make_data, make_loop, set_server

ROOT = Path(__file__).resolve().parents[1]
CALL = '<tool_call><function=alfworld_action><parameter=action>look</parameter></function></tool_call>'


@pytest.mark.parametrize('profile', ['qwen25_1_5b', 'qwen35_2b', 'qwen35_9b'])
def test_composed_profile_selects_xml_loop_and_template(monkeypatch, profile):
    with initialize_config_dir(config_dir=str(ROOT / 'config/alfworld' / profile / 'v1'), version_base=None):
        config = compose(config_name='alfworld_config_2gpu')
    assert config.variables.PROMPT_VERSION == PROMPT_VERSION
    assert config.variables.PROMPT_PROFILE == 'gigpo'
    assert config.actor_rollout_ref.rollout.multi_turn.format == 'qwen3_coder'

    def init(self, *args, **kwargs):
        self.config = config
        self.apply_chat_template_kwargs = dict(config.data.apply_chat_template_kwargs)
        self.processor = None
        self.tokenizer = self.tool_parser = object()
        self.tools = {'alfworld_action': SimpleNamespace(config={
            'environment_driven': True, 'max_steps': 50, 'max_new_tokens_per_turn': 768,
        })}
        self.response_length = 38400

    monkeypatch.setattr(ToolAgentLoop, '__init__', init)
    loop = ALFWorldToolAgentLoop()
    assert loop._gigpo_prompt and loop._xml_actions and not loop._text_actions
    assert loop.apply_chat_template_kwargs['chat_template'] == QWEN3_ALFWORLD_CHAT_TEMPLATE


@pytest.mark.parametrize('thinking', [True, False])
def test_prompt_and_chat_template_agree_on_xml(thinking):
    prompt = build_user_prompt(mission='find apple', observation='room', admissible_actions=['look'],
                               enable_thinking=thinking)
    rendered = Environment().from_string(QWEN3_ALFWORLD_CHAT_TEMPLATE).render(
        messages=[{'role': 'user', 'content': prompt}], add_generation_prompt=True, enable_thinking=thinking,
    )
    assert '<function=alfworld_action>' in prompt and '<parameter=action>' in prompt
    assert '<action>' not in rendered and '"arguments"' not in rendered
    assert rendered.endswith('<think>\n' if thinking else '<think>\n\n</think>\n\n')
    assert ('Think briefly' in prompt) is thinking
    output = ('Reasoning.</think>' if thinking else '') + CALL
    assert parse_xml_decision(output, enable_thinking=thinking).action == 'look'
    messages, tools = build_messages(mission='find apple', observation='room',
                                    registry=ALFWorldToolRegistry(['look']))
    assert messages[0]['role'] == 'user'
    assert tools[0]['function']['name'] == 'alfworld_action'


def test_history_window_keeps_absolute_steps_and_current_actions():
    history = [f"[Observation {i}: 'room {i}', Action {i}: 'look']" for i in range(1, 5)]
    prompt = build_user_prompt(mission='find apple', observation='current room',
                               admissible_actions=['open drawer 5', 'help'], history=history)
    assert 'already taken 4 step(s)' in prompt and 'now at step 5' in prompt
    assert 'most recent 2 observations' in prompt
    assert history[2] in prompt and history[3] in prompt
    assert history[0] not in prompt and history[1] not in prompt
    assert "['open drawer 5']" in prompt and "'help'" not in prompt


def gigpo_loop():
    loop = make_loop(max_steps=8)
    loop._gigpo_prompt = loop._xml_actions = loop._thinking_enabled = True
    loop._text_actions = False
    return loop


def test_xml_rollout_rebuilds_context_and_preserves_training_tokens():
    class Tool:
        def __init__(self):
            self.steps = 0

        def get_state(self, instance):
            return 'room 0\nYour task is to: find apple.', ('look',)

        async def execute(self, instance, params, **kwargs):
            assert params == {'action': 'look'}
            self.steps += 1
            return ToolResponse(text='result'), 0.0, {
                'action': 'look', 'observation': f'room {self.steps}',
                'admissible_commands': ['look', f'open drawer {self.steps}'],
            }

    async def run():
        loop = gigpo_loop()
        tool = Tool()
        loop.tools = {'alfworld_action': tool}
        loop._get_or_create_tool_instance = AsyncMock(return_value='instance')
        data = make_data(loop)
        await loop._handle_pending_state(data, {})
        assert data.extra_fields['alfworld_prompt_version'] == PROMPT_VERSION
        assert 'already taken' not in data.messages[0]['content']
        assert 'find apple' in data.messages[0]['content']
        all_ids = []
        for step in range(1, 5):
            output = '<think>Inspect this room.</think>' + CALL
            set_server(loop, output)
            assert await loop._generate_environment_decision(data, {}) == AgentState.PROCESSING_TOOLS
            ids = loop.tokenizer.encode(output)
            all_ids.extend(ids)
            assert data.step_outputs[-1].response_ids == ids
            assert await loop._process_environment_decision(data) == AgentState.GENERATING
            prompt = data.messages[0]['content']
            assert f'already taken {step} step(s)' in prompt
            assert f'now at step {step + 1}' in prompt
            assert f'Observation {step}: \'room {step - 1}' in prompt
            assert f"Action {step}: 'look'" in prompt
            assert f'current observation is: room {step}' in prompt
            assert f"'open drawer {step}'" in prompt
            assert 'Inspect this room.' not in prompt
            if step > 2:
                assert f'Observation {step - 2}:' not in prompt
        assert tool.steps == 4
        assert [v for out in data.step_outputs for v in out.response_mask] == [1] * len(all_ids)
        assert [v for out in data.step_outputs for v in out.response_logprobs] == [-0.1] * len(all_ids)
        assert [v for out in data.step_outputs for v in out.response_ids] == all_ids
        await loop._set_authoritative_initial_prompt(data)
        assert data.alfworld_prompt_history == []
        assert 'already taken' not in data.messages[0]['content']

    asyncio.run(run())


@pytest.mark.parametrize('output', [
    '<action>look</action>',
    '<tool_call>{"name":"alfworld_action","arguments":{"action":"look"}}</tool_call>',
])
def test_gigpo_loop_rejects_non_qwen3_output(output):
    async def run():
        loop = gigpo_loop()
        data = make_data(loop)
        loop._call_tool = AsyncMock()
        set_server(loop, '<think>Inspect.</think>' + output)
        assert await loop._generate_environment_decision(data, {}) == AgentState.PROCESSING_TOOLS
        assert data.tool_calls == []
        assert await loop._process_environment_decision(data) == AgentState.GENERATING
        loop._call_tool.assert_not_awaited()
        assert data.extra_fields['alfworld_no_tool_call_penalty_count'] == 1

    asyncio.run(run())
