"""Opt-in optimizer configuration for ALFWorld's native branch-entropy engine."""

from dataclasses import dataclass
import math

from verl.workers.config.optimizer import FSDPOptimizerConfig


@dataclass
class ALFWorldFSDPOptimizerConfig(FSDPOptimizerConfig):
    """Expose the native cosine scheduler's initial warmup learning-rate ratio."""

    init_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.init_lr_ratio) or not 0 <= self.init_lr_ratio <= 1:
            raise ValueError("init_lr_ratio must be finite and between 0 and 1")
        if self.init_lr_ratio and self.lr_scheduler_type != "cosine":
            raise ValueError("Nonzero init_lr_ratio currently requires the cosine scheduler")
