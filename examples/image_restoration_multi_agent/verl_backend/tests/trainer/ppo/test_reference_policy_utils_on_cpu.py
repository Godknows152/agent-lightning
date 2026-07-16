# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf

from verl.trainer.ppo.utils import use_reference_policy_in_actor


def _config(*, lora_rank=0, lora_adapter_path=None, use_separate_lora_reference=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "lora_rank": lora_rank,
                    "lora_adapter_path": lora_adapter_path,
                },
                "ref": {
                    "use_separate_lora_reference": use_separate_lora_reference,
                },
            }
        }
    )


def test_non_lora_reference_does_not_run_in_actor():
    assert not use_reference_policy_in_actor(_config())


def test_lora_rank_uses_raw_base_in_actor_reference_by_default():
    assert use_reference_policy_in_actor(_config(lora_rank=16))


def test_lora_adapter_path_uses_raw_base_in_actor_reference_by_default():
    assert use_reference_policy_in_actor(_config(lora_adapter_path="/tmp/sft_adapter"))


def test_separate_lora_reference_keeps_initial_adapter_enabled():
    config = _config(
        lora_rank=16,
        lora_adapter_path="/tmp/sft_adapter",
        use_separate_lora_reference=True,
    )

    assert not use_reference_policy_in_actor(config)
