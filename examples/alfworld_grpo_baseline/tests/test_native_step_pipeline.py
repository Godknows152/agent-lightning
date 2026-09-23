"""Exercise native queue serialization, GRPO and PPO loss entirely on CPU."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from alfworld_baseline.prompt_profiles import get_prompt_profile
from test_decision_budget import make_loop, CALL
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.protocol import DataProto
from verl.tools.schemas import ToolResponse
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_policy_loss_vanilla
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.v1 import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories


def episode(success=True, fail_generate=False):
    loop = make_loop(max_steps=3)
    loop._xml_actions = loop._gigpo_prompt = True
    loop._text_actions = False
    loop._prompt_profile = get_prompt_profile('gigpo')
    calls, released = [], []

    class Tool:
        async def create(self, **kwargs):
            return 'instance', ToolResponse(text='room 0')
        def get_state(self, instance):
            return 'room 0\nYour task is to: find apple.', ('look',)
        async def execute(self, instance, args, **kwargs):
            calls.append(args['action'])
            done = len(calls) == 3
            return ToolResponse(text=f'room {len(calls)}'), 10.0 if done else 0.0, {
                'action': args['action'], 'won': done, 'done': done,
                'observation': f'room {len(calls)}', 'admissible_commands': ['look'],
            }
        async def release(self, instance):
            released.append(instance)

    loop.tools = {'alfworld_action': Tool()}
    prompts = []
    def template(messages, **kwargs):
        # Different contexts must survive into separate native training rows.
        prompts.append(messages[0]['content'])
        return [10 + len(prompts), 42]
    loop.apply_chat_template = AsyncMock(side_effect=template)
    async def generate(**kwargs):
        if fail_generate:
            raise RuntimeError('server failed')
        text = CALL if success else 'no call'
        ids = loop.tokenizer.encode(text)
        return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids),
                               num_preempted=0, routed_experts=None,
                               extra_fields={'min_global_steps': 2, 'max_global_steps': 2})
    loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=generate))
    return loop, calls, released, prompts


def run_episode(success=True):
    loop, calls, released, prompts = episode(success)
    outputs = asyncio.run(loop.run({}, raw_prompt=[{'role': 'user', 'content': 'stale state'}]))
    assert released == ['instance']
    assert len(outputs) == 3
    assert [out.prompt_ids for out in outputs] == [[11 + i, 42] for i in range(len(outputs))]
    assert [call.kwargs['prompt_ids'] for call in loop.server_manager.generate.await_args_list] == [out.prompt_ids for out in outputs]
    assert all(out.reward_score == (10 if success else -6) for out in outputs)
    assert sum(out.extra_fields['alfworld_is_final_step'] for out in outputs) == 1
    assert all('alfworld_turn_contexts' not in out.extra_fields for out in outputs)
    if success:
        assert calls == ['look'] * 3
        assert 'Observation 1:' in prompts[2] and 'Observation 2:' in prompts[2]
        assert 'room 2' in prompts[2]
        assert outputs[-1].extra_fields['alfworld_repeated_action_penalty_count'] == 2
    return outputs


def test_native_queue_rows_rewards_advantages_padding_and_cpu_optimizer(monkeypatch):
    import transfer_queue as tq
    rows, tags, keys = [], [], []
    async def put(**kwargs):
        keys.extend(kwargs['keys'])
        tags.extend(kwargs['tags'])
        rows.extend(kwargs['fields'][i].to_dict() for i in range(len(kwargs['keys'])))
    monkeypatch.setattr(tq, 'async_kv_batch_put', put)
    worker = SimpleNamespace(
        reward_loop_worker_handles=[],
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(ids.shape[-1]).unsqueeze(0),
    )
    worker._compute_score = lambda outputs, kwargs: AgentLoopWorker._compute_score(worker, outputs, kwargs)
    method = AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess
    for session_id, success in enumerate((True, False)):
        outputs = run_episode(success)
        asyncio.run(method(worker, outputs, False, uid='task_with_underscore', session_id=session_id,
                           global_steps=3, partition_id='train'))
    assert keys == ['task_with_underscore_0_0', 'task_with_underscore_0_1',
                    'task_with_underscore_0_2', 'task_with_underscore_1_0',
                    'task_with_underscore_1_1', 'task_with_underscore_1_2']
    for row, tag in zip(rows, tags):
        assert row['input_ids'].tolist() == row['prompts'].tolist() + row['responses'].tolist()
        assert row['position_ids'].tolist() == list(range(len(row['input_ids'])))
        assert tag['min_global_steps'] == tag['max_global_steps'] == 2
        assert torch.equal(row['response_mask'], row['loss_mask'])
        assert len(row['rollout_log_probs']) == len(row['responses'])
    # Use native padding construction, with its separate uid and zero loss mask.
    padding, tag = construct_minimal_padding_template(rows[0], tags[0], eos_token_id=0)
    padding['uid'] = 'pad-only'
    assert tag['is_padding']
    assert padding['loss_mask'].sum() == 0
    rows.append(padding)
    keys.append('pad-only_0_0')
    # Shuffle rows: final-session selection must use output indices, not order.
    order = [3, 2, 6, 0, 5, 1, 4]
    rows = [rows[i] for i in order]
    keys = [keys[i] for i in order]
    masks = torch.nn.utils.rnn.pad_sequence([r['response_mask'] for r in rows], batch_first=True)
    rewards = torch.nn.utils.rnn.pad_sequence([r['rm_scores'] for r in rows], batch_first=True)
    batch = DataProto(batch=TensorDict({'response_mask': masks, 'token_level_rewards': rewards}, batch_size=[7]),
                      non_tensor_batch={'uid': np.array([r['uid'] for r in rows], dtype=object)})
    result = compute_advantage_for_multi_trajectories(batch, keys, AdvantageEstimator.GRPO)
    expected = 1 / np.sqrt(2)  # sample std over [10, -6], irrespective of step count
    for index, key in enumerate(keys):
        advantage = result.batch['advantages'][index]
        if key.startswith('pad'):
            assert torch.count_nonzero(advantage) == 0
        else:
            sign = -1 if key.rsplit('_', 2)[1] == '1' else 1
            assert torch.allclose(advantage[masks[index].bool()], torch.full_like(advantage[masks[index].bool()], sign * expected), atol=1e-6)
    # Native PPO objective/backward, including real per-step masks and padding.
    from omegaconf import OmegaConf
    parameter = torch.nn.Parameter(torch.zeros_like(rewards))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    config = OmegaConf.create({'clip_ratio': .2, 'clip_ratio_low': None, 'clip_ratio_high': None,
                              'clip_ratio_c': 3.0, 'global_batch_info': {}})
    loss, _ = compute_policy_loss_vanilla(torch.zeros_like(parameter), parameter,
                                         result.batch['advantages'], masks, config=config)
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter.grad[masks.bool()].abs().sum() > 0
    assert parameter.grad[~masks.bool()].abs().sum() == 0
    optimizer.step()
    assert parameter.detach().abs().sum() > 0


def test_environment_released_when_generation_raises():
    loop, _, released, _ = episode(fail_generate=True)
    with pytest.raises(RuntimeError, match='server failed'):
        asyncio.run(loop.run({}, raw_prompt=[]))
    assert released == ['instance']


def test_no_calls_preserve_environment_and_penalties_before_later_success():
    from alfworld_baseline.metrics import compute_alfworld_rollout_metrics

    loop, calls, released, prompts = episode()
    loop._thinking_enabled = True
    texts = iter(['x' * 256, '</think>No call.', '</think>' + CALL])

    async def generate(**kwargs):
        ids = loop.tokenizer.encode(next(texts))
        return SimpleNamespace(token_ids=ids, log_probs=[-.1] * len(ids),
                               num_preempted=0, routed_experts=None)

    tool = loop.tools['alfworld_action']
    tool.execute = AsyncMock(return_value=(ToolResponse(text='solved'), 10.0, {
        'action': 'look', 'won': True, 'done': True,
        'observation': 'solved', 'admissible_commands': ['look'],
    }))
    loop.server_manager.generate = AsyncMock(side_effect=generate)
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert len(outputs) == 3
    assert all(out.reward_score == 6.0 for out in outputs)
    assert released == ['instance']
    tool.execute.assert_awaited_once()
    assert tool.execute.await_args.args[0] == 'instance'
    assert all('room 0' in prompt for prompt in prompts)
    assert "Action 1: 'None'" in prompts[1]
    final = outputs[-1].extra_fields
    assert final['tool_rewards'] == [-2.0, -2.0, 10.0]
    assert final['alfworld_terminal_reason'] == 'success'
    assert final['alfworld_no_tool_call_penalty_count'] == 2
    assert final['alfworld_no_action_overlong_thinking_count'] == 1
    assert final['alfworld_no_action_tool_call_format_count'] == 1
    metrics = compute_alfworld_rollout_metrics(SimpleNamespace(
        non_tensor_batch={key: [value] for key, value in final.items()}
    ))
    assert metrics['alfworld_penalty/no_action_count'] == 2
    assert metrics['alfworld_penalty/no_action/overlong_thinking_count'] == 1
    assert metrics['alfworld_penalty/no_action/tool_call_format_count'] == 1
    assert metrics['alfworld_penalty/no_action/other_count'] == 0
    assert metrics['alfworld_termination/no_tool_call_count'] == 0


def test_prompt_limit_fails_without_truncating_or_calling_server():
    loop, _, released, _ = episode()
    loop.prompt_length = 1
    with pytest.raises(ValueError, match='exceeding prompt_length'):
        asyncio.run(loop.run({}, raw_prompt=[]))
    loop.server_manager.generate.assert_not_awaited()
    assert released == ['instance']


def test_real_tokenizer_rebuild_has_matching_ids():
    from transformers import AutoTokenizer
    from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
    from alfworld_baseline.prompts_gigpo import QWEN3_ALFWORLD_CHAT_TEMPLATE
    loop, _, _, _ = episode()
    loop.tokenizer = AutoTokenizer.from_pretrained('/home/LXJ/Python_Projects/Models/Qwen3.5-2B', local_files_only=True)
    loop.apply_chat_template_kwargs = {'chat_template': QWEN3_ALFWORLD_CHAT_TEMPLATE, 'enable_thinking': True}
    del loop.apply_chat_template
    messages = loop._state_prompt_messages(mission='find apple', observation='room', actions=('look',))
    ids = asyncio.run(ALFWorldToolAgentLoop.apply_chat_template(loop, messages, tools=loop.tool_schemas))
    from verl.utils.tokenizer import normalize_token_ids
    assert ids == normalize_token_ids(loop.tokenizer.apply_chat_template(messages, tools=loop.tool_schemas,
        tokenize=True, add_generation_prompt=True, **loop.apply_chat_template_kwargs))
    assert all(isinstance(token, int) for token in ids)
    assert loop.tokenizer.decode(ids).endswith('<think>\n')
    assert len(ids) < 2048


def test_real_constructor_and_native_qwen35_processor_without_model_weights():
    from alfworld_baseline.agent_loop import ALFWorldToolAgentLoop
    from alfworld_baseline.budget import configure_environment_driven_rollout
    from test_native_entrypoint import config_for
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils import hf_tokenizer, hf_processor
    from verl.utils.dataset.rl_dataset import RLHFDataset

    cfg = config_for('qwen35_2b')
    configure_environment_driven_rollout(cfg)
    tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, local_files_only=True)
    processor = hf_processor(cfg.actor_rollout_ref.model.path, local_files_only=True)
    fake_loop, _, released, _ = episode()
    tool = fake_loop.tools['alfworld_action']
    tool.name = 'alfworld_action'
    tool.config = {'environment_driven': True, 'max_steps': 3, 'max_new_tokens_per_turn': 768}
    tool.tool_schema = OpenAIFunctionToolSchema.model_validate(fake_loop.tool_schemas[0])
    ids = tokenizer.encode('No executable action.', add_special_tokens=False)
    server = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
        token_ids=ids, log_probs=[-.1] * len(ids), num_preempted=0, routed_experts=None,
        extra_fields={'min_global_steps': 0, 'max_global_steps': 0})))

    async def run():
        loop = ALFWorldToolAgentLoop(trainer_config=DictConfigWrap(cfg), server_manager=server,
            tokenizer=tokenizer, processor=processor, hf_model_type='qwen3_5',
            dataset_cls=RLHFDataset, data_config=DictConfigWrap(cfg.data), tools=ToolListWrap([tool]))
        return await loop.run({}, raw_prompt=[{'role': 'user', 'content': 'stale'}])
    output = asyncio.run(run())[-1]
    assert released == ['instance']
    assert output.reward_score == -6
    assert output.prompt_ids == server.generate.await_args.kwargs['prompt_ids']
    assert output.response_ids == ids
    worker = SimpleNamespace(processor=processor, tokenizer=tokenizer,
                             _get_mm_processor_kwargs=lambda audios: {})
    input_ids = torch.tensor(output.prompt_ids + output.response_ids)
    mm = AgentLoopWorker._compute_multi_modal_inputs(worker, output, input_ids)
    positions = AgentLoopWorker._compute_position_ids(worker, input_ids.unsqueeze(0), torch.ones_like(input_ids).unsqueeze(0), mm)
    assert positions.shape[-1] == len(input_ids)
    assert torch.equal(positions.reshape(-1, len(input_ids))[0], torch.arange(len(input_ids)))
