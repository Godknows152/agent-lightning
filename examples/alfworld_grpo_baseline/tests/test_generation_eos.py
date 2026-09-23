"""ALFWorld requests must stop at the SFT EOS even without a server tokenizer."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sglang.srt.managers.schedule_batch import Req

from test_decision_budget import CALL, make_data, make_loop
from verl.experimental.agent_loop.tool_agent_loop import AgentState


@pytest.mark.parametrize("temperature", [1.0, 0.0], ids=["training", "validation"])
@pytest.mark.parametrize("existing_stops", [None, [], [248044, 248046, 248046]])
def test_sft_eos_stops_native_scheduler_before_extra_dialogue(temperature, existing_stops):
    loop = make_loop()
    loop._xml_actions = True
    loop._thinking_enabled = True
    loop.tokenizer.eos_token_id = 248046
    loop.tokenizer.decode = lambda ids, **kwargs: ''.join(
        '<|im_end|>' if token == 248046 else chr(token) for token in ids
    )
    data = make_data(loop)
    answer_ids = loop.tokenizer.encode('Choose the next action.</think>\n' + CALL) + [248046]
    continued_ids = answer_ids + loop.tokenizer.encode('user\nInvented next observation')
    params = {'temperature': temperature, 'stop_token_ids': existing_stops,
              'ignore_eos': True, 'logprobs': True}
    original_params = deepcopy(params)

    async def generate(**kwargs):
        sent = kwargs['sampling_params']
        # Use the actual installed SGLang finish check, with the problematic
        # base-model EOS and no tokenizer, without starting a model/server.
        req = Req.__new__(Req)
        req.sampling_params = SimpleNamespace(
            ignore_eos=sent.get('ignore_eos', False), stop_token_ids=sent.get('stop_token_ids')
        )
        req.eos_token_ids = {248044}
        req.tokenizer = None
        req.output_ids = []
        for token in continued_ids:
            req.output_ids.append(token)
            if req._check_token_based_finish([token]):
                break
        return SimpleNamespace(token_ids=req.output_ids, log_probs=[-0.1] * len(req.output_ids),
                               num_preempted=0, routed_experts=None)

    loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=generate))

    async def run():
        for _ in range(2):
            assert await loop._generate_environment_decision(data, params) == AgentState.PROCESSING_TOOLS
            assert data.alfworld_xml_decision.status == 'valid'
            assert data.alfworld_pending_action == 'look'
            # EOS and its log-prob remain aligned in the trainable response.
            assert data.response_ids == answer_ids
            assert data.response_logprobs == [-0.1] * len(answer_ids)
            assert data.response_mask == [1] * len(answer_ids)

    asyncio.run(run())
    assert params == original_params
    for call in loop.server_manager.generate.await_args_list:
        sent = call.kwargs['sampling_params']
        assert sent['temperature'] == temperature
        assert sent['stop_token_ids'] == sorted(set(existing_stops or []) | {248046})
        assert sent['ignore_eos'] is False
