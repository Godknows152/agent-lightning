# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for request-level stop tokens in the SGLang rollout server."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import torch

from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLANG_STOP_TOKEN_IDS,
    SGLangHttpServer,
)


def test_generate_sends_explicit_stop_token_ids() -> None:
    class _TokenizerManager:
        request: Any = None

        async def generate_request(self, request: Any, _: Any):
            self.request = request
            yield {
                "output_ids": [7],
                "meta_info": {"finish_reason": {"type": "stop"}},
            }

    tokenizer_manager = _TokenizerManager()
    server = SGLangHttpServer.__new__(SGLangHttpServer)
    untyped_server = cast(Any, server)
    untyped_server.config = SimpleNamespace(
        max_model_len=128,
        response_length=32,
        prompt_length=16,
        disable_multimodal_special_token_generation=False,
        enable_rollout_routing_replay=False,
    )
    untyped_server.model_config = SimpleNamespace(
        processor=None,
        lora_rank=0,
    )
    untyped_server.tokenizer_manager = tokenizer_manager
    untyped_server.global_steps = 0

    asyncio.run(
        server.generate(
            prompt_ids=torch.tensor([1, 2, 3]),
            sampling_params={"temperature": 0.5, "stop_token_ids": [1]},
            request_id="test-request",
        )
    )

    assert tokenizer_manager.request.sampling_params["stop_token_ids"] == [248046, 248044]
    assert tokenizer_manager.request.sampling_params["stop_token_ids"] == list(SGLANG_STOP_TOKEN_IDS)
