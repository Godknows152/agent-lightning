"""Select ALFWorld's rollout adapter only inside its dedicated Ray workers."""
from typing import Any

import ray
from omegaconf import DictConfig

from verl.trainer.ppo.utils import Role
from verl.workers.engine_workers import ActorRolloutRefWorker


class ALFWorldActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        if "rollout" in role and config.rollout.name == "sglang":
            from verl.workers.rollout.base import _ROLLOUT_REGISTRY

            _ROLLOUT_REGISTRY[("sglang", "async")] = "alfworld_baseline.sglang_rollout.ALFWorldServerAdapter"
        super().__init__(config=config, role=role, **kwargs)


class ALFWorldWorkerMixin:
    def _init_resource_pool_mgr(self) -> None:
        super()._init_resource_pool_mgr()
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(ALFWorldActorRolloutRefWorker)
