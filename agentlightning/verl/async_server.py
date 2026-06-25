# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from copy import deepcopy

import ray
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

from agentlightning.instrumentation.vllm import ChatCompletionResponsePatched, instrument_vllm


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def _mapping_get(value, key):
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _chat_completion_supports_chat_template_kwargs():
    fields = getattr(ChatCompletionRequest, "model_fields", None)
    if isinstance(fields, dict):
        return "chat_template_kwargs" in fields
    fields = getattr(ChatCompletionRequest, "__fields__", None)
    return isinstance(fields, dict) and "chat_template_kwargs" in fields


def _configured_chat_template_kwargs(config):
    agentlightning_config = _mapping_get(config, "agentlightning")
    value = _mapping_get(agentlightning_config, "chat_template_kwargs")
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _apply_configured_chat_template_kwargs(request_json, config):
    if not _chat_completion_supports_chat_template_kwargs():
        request_json.pop("chat_template_kwargs", None)
        return

    configured_kwargs = _configured_chat_template_kwargs(config)
    if not configured_kwargs:
        return

    request_kwargs = request_json.get("chat_template_kwargs")
    merged_kwargs = dict(request_kwargs.items()) if hasattr(request_kwargs, "items") else {}
    merged_kwargs.update(configured_kwargs)
    request_json["chat_template_kwargs"] = merged_kwargs


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: [OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
        """
        request_json = await raw_request.json()
        _apply_configured_chat_template_kwargs(request_json, self.config)
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.model_dump())
