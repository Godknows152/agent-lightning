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
# Queue serialization adapted from veRL's trainer/ppo/v1/agent_loop_tq.py.
"""Isolated V1 workers that preserve ALFWorld decision rewards."""

from __future__ import annotations

from typing import Any

import ray
import torch
import transfer_queue as tq

from verl.experimental.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


class StepGRPOWorkerMixin:
    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs: Any,
    ) -> None:
        # Existing shared parquet files name the original loop explicitly. Route
        # them here without rewriting datasets or changing the original registry.
        if agent_name not in {"alfworld_tool_agent", "alfworld_step_grpo_agent"}:
            raise ValueError(f"Unexpected agent for ALFWorld step GRPO: {agent_name}")
        return await super()._run_agent_loop(
            sampling_params,
            trajectory,
            agent_name="alfworld_step_grpo_agent",
            trace=trace,
            **kwargs,
        )

    async def _agent_loop_postprocess(
        self,
        output: AgentLoopOutput | list[AgentLoopOutput],
        validate: bool,
        **kwargs: Any,
    ) -> None:
        """Serialize each score intact; native V1 otherwise broadcasts the last one."""
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            return
        keys, fields, tags = [], [], []
        for index, output in enumerate(outputs):
            if (
                output.extra_fields.get("alfworld_training_backend") != "gigpo_grpo"
                or output.reward_score is None
            ):
                raise ValueError(
                    "Step GRPO requires outputs scored by ALFWorldStepGRPOAgentLoop"
                )
            # GiGPO evaluates pure environment outcomes; local penalties are a
            # training signal. This also keeps previews correct without a dump dir.
            if validate:
                output.reward_score = output.extra_fields["alfworld_episode_reward"]
                output.extra_fields["reward_extra_info"] = {
                    "score": output.reward_score,
                    "episode_reward": output.reward_score,
                }
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses])
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0),
                torch.ones_like(input_ids).unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)
            field = output.as_dict()
            field.update(kwargs)
            field.pop("multi_modal_data", None)
            field.update(
                loss_mask=field["response_mask"],
                input_ids=input_ids,
                position_ids=position_ids,
                multi_modal_inputs=multi_modal_inputs,
            )
            keys.append(f"{kwargs['uid']}_{kwargs['session_id']}_{index}")
            fields.append(field)
            prompt_len, response_len = (
                field["prompts"].size(0),
                field["responses"].size(0),
            )
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="val" if validate else "train",
        )


@ray.remote
class StepGRPOAgentLoopWorkerTQ(
    StepGRPOWorkerMixin, AgentLoopWorkerTQ.__ray_metadata__.modified_class
):
    """Retain native background-session handling, replacing only ALFWorld hooks."""


class StepGRPOAgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = StepGRPOAgentLoopWorkerTQ
