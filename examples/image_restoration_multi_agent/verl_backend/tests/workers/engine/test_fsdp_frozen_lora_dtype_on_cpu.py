# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, _cast_frozen_lora_model_to_dtype


class _ToyLoraModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.lora_adapter = nn.Linear(4, 2, bias=False, dtype=torch.float32)


def _build_tiny_bf16_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).to(torch.bfloat16)


def test_casts_frozen_lora_model_to_configured_dtype() -> None:
    model = _ToyLoraModel().requires_grad_(False)

    _cast_frozen_lora_model_to_dtype(model, forward_only=True, model_dtype="bfloat16")

    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_preserves_trainable_lora_model_dtypes() -> None:
    model = _ToyLoraModel()

    _cast_frozen_lora_model_to_dtype(model, forward_only=False, model_dtype="bfloat16")

    assert model.base_layer.weight.dtype == torch.bfloat16
    assert model.lora_adapter.weight.dtype == torch.float32


@pytest.mark.filterwarnings("ignore:Could not find a config file")
def test_pretrained_fp32_adapter_is_cast_only_for_frozen_reference(tmp_path) -> None:
    adapter_path = tmp_path / "adapter"
    source_model = get_peft_model(
        _build_tiny_bf16_model(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    source_model.save_pretrained(adapter_path)
    assert {parameter.dtype for name, parameter in source_model.named_parameters() if "lora_" in name} == {
        torch.float32
    }

    frozen_engine = object.__new__(FSDPEngine)
    frozen_engine.model_config = SimpleNamespace(lora_adapter_path=str(adapter_path), use_shm=False)
    frozen_engine.engine_config = SimpleNamespace(forward_only=True, model_dtype="bfloat16")
    frozen_model = frozen_engine._build_lora_module(_build_tiny_bf16_model())
    assert {parameter.dtype for parameter in frozen_model.parameters()} == {torch.bfloat16}

    actor_engine = object.__new__(FSDPEngine)
    actor_engine.model_config = SimpleNamespace(lora_adapter_path=str(adapter_path), use_shm=False)
    actor_engine.engine_config = SimpleNamespace(forward_only=False, model_dtype="bfloat16")
    actor_model = actor_engine._build_lora_module(_build_tiny_bf16_model())
    assert {parameter.dtype for name, parameter in actor_model.named_parameters() if "lora_" in name} == {torch.float32}
