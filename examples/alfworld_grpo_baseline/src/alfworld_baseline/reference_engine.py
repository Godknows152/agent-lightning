"""Opt-in resident reference weights for hosts with fragmented pinned memory."""
from dataclasses import replace
from typing import Any

from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import TrainingWorker


@EngineRegistry.register(model_type="alfworld_resident_reference", backend="fsdp", device="cuda")
class ResidentReferenceEngine(FSDPEngineWithLMHead):
    """Keep native reference inference, dtype and sharding; omit CPU offload."""

    def __init__(self, model_config, engine_config, *args: Any, **kwargs: Any) -> None:
        if not engine_config.forward_only or engine_config.strategy != "fsdp":
            raise ValueError("Resident reference requires a forward-only FSDP engine")
        model_config.model_type = "language_model"
        super().__init__(model_config, engine_config, *args, **kwargs)
        self._is_offload_param = False
        self._is_offload_optimizer = False

    def _build_fsdp_module(self, module):
        # Native FSDP forces CPUOffload for every forward-only model, ignoring
        # param_offload=False. Suppress that choice ONLY during wrapping. Model
        # loading and optimizer construction still see forward_only=True, so
        # reference weights retain BF16 and no optimizer is created.
        original = self.engine_config
        self.engine_config = replace(original, forward_only=False)
        try:
            wrapped = super()._build_fsdp_module(module)
        finally:
            self.engine_config = original
        print("ALFWorld reference: GPU-resident FSDP weights; native forward-only inference", flush=True)
        return wrapped


class ResidentReferenceWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig) -> None:
        super().__init__(replace(config, model_type="alfworld_resident_reference"))
