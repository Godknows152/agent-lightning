"""Keep the best validation checkpoint and recent complete FSDP checkpoints."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
import shutil
from typing import Any

from omegaconf import OmegaConf, open_dict

logger = logging.getLogger(__name__)


def complete_checkpoint(path: Path, use_critic: bool = False) -> bool:
    """Require dataloader state and every model/optimizer/RNG shard on disk."""
    if path.is_symlink() or not (path / "data.pt").is_file():
        return False
    for role in ("actor", "critic") if use_critic else ("actor",):
        models = list((path / role).glob("model_world_size_*_rank_*.pt"))
        if not models:
            return False
        match = re.fullmatch(r"model_world_size_(\d+)_rank_\d+\.pt", models[0].name)
        if match is None:
            return False
        world = int(match[1])
        if world < 1 or len(models) != world:
            return False
        for kind in ("model", "optim", "extra_state"):
            for rank in range(world):
                shard = path / role / f"{kind}_world_size_{world}_rank_{rank}.pt"
                if not shard.is_file() or shard.stat().st_size == 0:
                    return False
    return True


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace small retention metadata after a successful save."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def best_checkpoint(root: Path, metric: str, use_critic: bool = False) -> dict[str, Any] | None:
    """Recover best state from checkpoint-local records, including after restart."""
    records = []
    for path in root.glob("global_step_*"):
        record_path = path / "validation_metric.json"
        if not complete_checkpoint(path, use_critic) or not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if record["metric"] == metric and math.isfinite(record["value"]):
            records.append(record)
    # Keep the earlier checkpoint on an exact tie.
    return max(records, key=lambda r: (r["value"], -r["step"]), default=None)


def prune_checkpoints(root: Path, limit: int, best_step: int | None, use_critic: bool = False) -> list[int]:
    """Delete whole obsolete checkpoints, protecting the best and latest steps."""
    if limit < 2:
        raise ValueError("Retention needs at least two slots for best and latest checkpoints")
    checkpoints = {
        int(path.name.removeprefix("global_step_")): path
        for path in root.glob("global_step_*")
        if re.fullmatch(r"global_step_\d+", path.name) and complete_checkpoint(path, use_critic)
    }
    keep = {best_step} if best_step in checkpoints else set()
    for step in sorted(checkpoints, reverse=True):
        if len(keep) >= limit:
            break
        keep.add(step)
    removed = []
    for step, path in checkpoints.items():
        if step not in keep:
            shutil.rmtree(path)
            removed.append(step)
    return sorted(removed)


def validate_checkpoint_retention(config: Any) -> None:
    """Reject save modes whose completion cannot be safely observed locally."""
    if not config.trainer.get("keep_best_checkpoint", False):
        return
    if int(config.trainer.max_actor_ckpt_to_keep) < 2:
        raise ValueError("keep_best_checkpoint requires max_actor_ckpt_to_keep >= 2")
    if OmegaConf.select(config, "actor_rollout_ref.actor.checkpoint.async_save", default=False):
        raise ValueError("Best checkpoint retention requires synchronous checkpoint saving")
    if config.trainer.get("default_hdfs_dir"):
        raise ValueError("Best checkpoint retention currently supports local checkpoints only")


class ALFWorldCheckpointRetentionMixin:
    """Apply retention after validation, which native V1 runs after saving.

    While a newly saved checkpoint is being validated there can temporarily be
    limit + 1 checkpoints. Only prune after learning whether that checkpoint is
    the new best. Native worker rotation is disabled to protect the old best.
    """

    def _retention_metric(self) -> str:
        return self.config.trainer.get("best_checkpoint_metric") or (
            f"val-core/alfworld/reward/mean@{self.config.actor_rollout_ref.rollout.val_kwargs.n}"
        )

    def _prune_alfworld_checkpoints(self) -> None:
        root = Path(self.config.trainer.default_local_dir)
        best = best_checkpoint(root, self._retention_metric(), self.use_critic)
        if best is not None:
            write_json(root / "best_checkpoint.json", best)
        removed = prune_checkpoints(
            root, int(self.config.trainer.max_actor_ckpt_to_keep),
            best["step"] if best else None, self.use_critic,
        )
        logger.info("Checkpoint retention: best=%s, removed=%s", best, removed)

    def _save_checkpoint(self) -> None:
        if not self.config.trainer.get("keep_best_checkpoint", False):
            return super()._save_checkpoint()
        validate_checkpoint_retention(self.config)
        trainer = self.config.trainer
        names = ("max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep", "remove_previous_ckpt_in_save")
        previous = {key: trainer.get(key) for key in names}
        present = {key for key in names if key in trainer}
        try:
            with open_dict(trainer):
                trainer.max_actor_ckpt_to_keep = None
                trainer.max_critic_ckpt_to_keep = None
                trainer.remove_previous_ckpt_in_save = False
            super()._save_checkpoint()
        finally:
            with open_dict(trainer):
                for key in names:
                    if key in present:
                        trainer[key] = previous[key]
                    else:
                        del trainer[key]
        current = Path(trainer.default_local_dir) / f"global_step_{self.global_steps}"
        if not complete_checkpoint(current, self.use_critic):
            raise RuntimeError(f"Incomplete checkpoint; refusing to prune older checkpoints: {current}")
        validation_due = trainer.test_freq > 0 and (
            self.global_steps % trainer.test_freq == 0 or self.global_steps >= self.total_training_steps
        )
        if not validation_due:
            self._prune_alfworld_checkpoints()

    def _validate(self) -> dict[str, float]:
        metrics = super()._validate()
        if not self.config.trainer.get("keep_best_checkpoint", False) or self.config.trainer.get("val_only", False):
            return metrics
        metric = self._retention_metric()
        if metric not in metrics or not math.isfinite(float(metrics[metric])):
            raise ValueError(f"Missing or non-finite checkpoint selection metric: {metric}")
        root = Path(self.config.trainer.default_local_dir)
        best = best_checkpoint(root, metric, self.use_critic)
        current = root / f"global_step_{self.global_steps}"
        value = float(metrics[metric])
        # Validation can run more frequently than periodic checkpoint saving.
        if (best is None or value > best["value"]) and not complete_checkpoint(current, self.use_critic):
            self._save_checkpoint()
        if complete_checkpoint(current, self.use_critic):
            write_json(current / "validation_metric.json", {"step": self.global_steps, "metric": metric, "value": value})
        self._prune_alfworld_checkpoints()
        best = best_checkpoint(root, metric, self.use_critic)
        if best is not None:
            metrics["checkpoint/best_step"] = best["step"]
            metrics["checkpoint/best_validation_reward"] = best["value"]
        return metrics
