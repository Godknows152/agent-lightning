"""Select ALFWorld's rollout adapter only inside its dedicated Ray workers."""
from typing import Any

import ray
from omegaconf import DictConfig

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.utils import Role
from verl.workers.engine_workers import ActorRolloutRefWorker


class ALFWorldActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        if "rollout" in role and config.rollout.name == "sglang":
            from verl.workers.rollout.base import _ROLLOUT_REGISTRY

            _ROLLOUT_REGISTRY[("sglang", "async")] = "alfworld_baseline.sglang_rollout.ALFWorldServerAdapter"
        super().__init__(config=config, role=role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        super().init_model()
        if self.actor is not None and self.config.model.lora_rank == 0:
            parameters = list(self.actor.engine.module.named_parameters())
            frozen = [name for name, parameter in parameters if not parameter.requires_grad]
            if frozen or any("lora_" in name for name, _ in parameters):
                raise RuntimeError(f"Full-parameter ALFWorld actor contains frozen or LoRA parameters: {frozen[:5]}")
            count = sum(parameter.numel() for _, parameter in parameters)
            print(f"ALFWorld full-parameter actor: local_trainable={count} local_total={count} frozen=0", flush=True)


class ALFWorldWorkerMixin:
    def _init_resource_pool_mgr(self) -> None:
        super()._init_resource_pool_mgr()
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(ALFWorldActorRolloutRefWorker)
