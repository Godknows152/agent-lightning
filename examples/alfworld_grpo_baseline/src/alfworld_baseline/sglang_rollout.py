"""ALFWorld adapter for SGLang's serialized_tensors LoRA loading protocol."""
import base64
from dataclasses import asdict
import pickle
from typing import Any, Generator

import torch

from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput
from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights

from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter
from verl.workers.rollout.sglang_rollout.utils import SGLANG_LORA_NAME, normalize_peft_config_for_sglang


def lora_request(config: dict[str, Any], tensors: dict[str, torch.Tensor]) -> LoadLoRAAdapterFromTensorsReqInput:
    """Send CPU weights by value, without CUDA IPC or multiprocessing FD handles."""
    if any(tensor.device.type != "cpu" for tensor in tensors.values()):
        raise ValueError("ALFWorld LoRA transport requires CPU tensor snapshots")
    # SGLang decodes this field with a base64-aware SafeUnpickler. Ordinary
    # pickle embeds CPU storage bytes; ForkingPickler instead uses an FD broker
    # whose authkey differs between independent Ray actors and TP processes.
    payload = base64.b64encode(pickle.dumps(tensors, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
    return LoadLoRAAdapterFromTensorsReqInput(
        lora_name=SGLANG_LORA_NAME,
        config_dict=normalize_peft_config_for_sglang(config),
        serialized_tensors=payload,
    )


class ALFWorldServerAdapter(ServerAdapter):
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        wire_format: str = "named_tensors",
        **kwargs: Any,
    ) -> None:
        peft_config = kwargs.get("peft_config")
        if wire_format != "named_tensors" or not peft_config or not kwargs.get("base_sync_done", False):
            return await super().update_weights(weights, global_steps=global_steps, wire_format=wire_format, **kwargs)

        await self._init_server_adapter()
        # Every FSDP rank must consume the generator to complete its collectives.
        # SGLang shards this full adapter dictionary internally across its TP ranks.
        # Keep owned CPU snapshots alive through the load request. Passing CUDA
        # storage handles across the scheduler processes can fail in
        # _new_shared_cuda; SGLang accepts CPU weights and stages them locally.
        tensors = {
            name: _preprocess_tensor_for_update_weights(tensor.detach()).to(device="cpu", copy=True)
            for name, tensor in weights
        }
        if self._engine is None:
            return
        models = await self._engine.available_models()
        if any(item["id"] == SGLANG_LORA_NAME for item in models["data"]):
            await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)
        request = lora_request(peft_config, tensors)
        result = await self._engine._make_async_request("load_lora_adapter_from_tensors", asdict(request))
        if not result.get("success", False):
            raise RuntimeError(f"SGLang failed to load ALFWorld LoRA: {result}")
