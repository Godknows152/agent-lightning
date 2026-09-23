"""Verify the installed SGLang request contract without starting a server."""
import asyncio
from dataclasses import asdict
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from sglang.srt.utils import MultiprocessingSerializer

from alfworld_baseline.sglang_rollout import ALFWorldServerAdapter, lora_request


def test_lora_request_roundtrip():
    tensors = {"layer.lora_A.weight": torch.arange(6).reshape(2, 3)}
    request = lora_request({"peft_type": "LORA", "r": 2, "target_modules": ["q_proj"], "lora_alpha": 4}, tensors)
    payload = asdict(request)
    assert isinstance(payload["serialized_tensors"], str)
    assert "serialized_named_tensors" not in payload
    decoded = MultiprocessingSerializer.deserialize(payload["serialized_tensors"])
    torch.testing.assert_close(decoded["layer.lora_A.weight"], tensors["layer.lora_A.weight"])


def test_lora_payload_decodes_in_independent_process():
    request = lora_request({"peft_type": "LORA", "r": 2, "target_modules": ["q_proj"]},
                           {"layer.lora_A.weight": torch.arange(6).reshape(2, 3)})
    # A fresh interpreter has a different multiprocessing authkey, as do Ray
    # actors and scheduler processes. No producer-side FD broker is needed.
    result = subprocess.run(
        [sys.executable, "-c", """
import multiprocessing, sys, torch
from sglang.srt.utils import MultiprocessingSerializer
multiprocessing.current_process().authkey = b'independent-scheduler'
payload = sys.stdin.read()
for _ in range(2):
    tensors = MultiprocessingSerializer.deserialize(payload)
    tensor = tensors['layer.lora_A.weight']
    assert tensor.device.type == 'cpu'
    torch.testing.assert_close(tensor, torch.arange(6).reshape(2, 3))
print('decoded twice')
"""], input=request.serialized_tensors, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert 'decoded twice' in result.stdout


@pytest.mark.parametrize("leader", [False, True])
def test_every_rank_consumes_lora_generator(monkeypatch, leader):
    monkeypatch.setattr("alfworld_baseline.sglang_rollout._preprocess_tensor_for_update_weights", lambda tensor: tensor)
    engine = SimpleNamespace(
        available_models=AsyncMock(return_value={"data": []}),
        _make_async_request=AsyncMock(return_value={"success": True}),
    )
    adapter = SimpleNamespace(_init_server_adapter=AsyncMock(), _engine=engine if leader else None)
    consumed = []
    def weights():
        consumed.append(True)
        yield "layer.lora_A.weight", torch.ones(2, 3)
    asyncio.run(ALFWorldServerAdapter.update_weights(
        adapter, weights(), peft_config={"peft_type": "LORA", "r": 2, "target_modules": ["q_proj"]}, base_sync_done=True,
    ))
    assert consumed == [True]
    assert engine._make_async_request.await_count == int(leader)


def test_adapter_load_owns_snapshot_of_source_weights(monkeypatch):
    monkeypatch.setattr("alfworld_baseline.sglang_rollout._preprocess_tensor_for_update_weights", lambda tensor: tensor)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = source.clone()

    async def load(endpoint, payload):
        assert endpoint == "load_lora_adapter_from_tensors"
        # Simulate the source storage being reused after materialization.
        source.zero_()
        decoded = MultiprocessingSerializer.deserialize(payload["serialized_tensors"])
        received = decoded["layer.lora_A.weight"]
        assert received.device.type == "cpu"
        torch.testing.assert_close(received, expected)
        return {"success": True}

    engine = SimpleNamespace(available_models=AsyncMock(return_value={"data": []}),
                             _make_async_request=AsyncMock(side_effect=load))
    adapter = SimpleNamespace(_init_server_adapter=AsyncMock(), _engine=engine)
    asyncio.run(ALFWorldServerAdapter.update_weights(
        adapter, iter([("layer.lora_A.weight", source)]),
        peft_config={"peft_type": "LORA", "r": 2, "target_modules": ["q_proj"]}, base_sync_done=True,
    ))
    engine._make_async_request.assert_awaited_once()
