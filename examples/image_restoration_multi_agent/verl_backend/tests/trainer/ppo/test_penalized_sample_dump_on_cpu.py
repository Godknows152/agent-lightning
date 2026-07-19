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

import json

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _Tokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        prefix = "visible" if skip_special_tokens else "raw"
        return f"{prefix}:" + ",".join(str(token_id) for token_id in token_ids)


def test_dump_penalized_samples_saves_only_sampled_negative_records(tmp_path) -> None:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.tokenizer = _Tokenizer()
    trainer.global_steps = 7
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "penalized_samples_dir": str(tmp_path),
                "penalized_samples_per_step": 1,
                "logger": [],
            }
        }
    )

    penalty_records = np.empty(3, dtype=object)
    penalty_records[:] = [
        [
            {
                "reason": "forged_user_or_assistant_role_after_tool_call",
                "value": -1.0,
                "model_response": "raw forged response",
            }
        ],
        [{"reason": "tool_execution_error", "value": 0.0}],
        [{"reason": "no_tool_call", "value": -10.0}],
    ]
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[10, 11], [12, 13], [14, 15]]),
                "responses": torch.tensor(
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ]
                ),
                "response_mask": torch.tensor(
                    [
                        [1, 1, 0, 1],
                        [1, 0, 1, 0],
                        [1, 1, 0, 0],
                    ]
                ),
                "token_level_scores": torch.tensor(
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-10.0, 0.0, 0.0, 0.0],
                    ]
                ),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "penalty_records": penalty_records,
            "uid": np.array(["uid-0", "uid-1", "uid-2"], dtype=object),
            "data_source": np.array(["restoration"] * 3, dtype=object),
        },
    )

    samples = trainer._dump_penalized_samples(batch)

    assert len(samples) == 1
    assert samples[0]["batch_index"] in {0, 2}
    assert all(record["value"] < 0 for record in samples[0]["penalties"])
    assert samples[0]["model_answers"]

    output_path = tmp_path / "step_000007.jsonl"
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == samples
