"""Exercise disk retention, restart recovery and native V1 save ordering on CPU."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

from alfworld_baseline.checkpoint_retention import (
    ALFWorldCheckpointRetentionMixin, best_checkpoint, complete_checkpoint,
    prune_checkpoints, validate_checkpoint_retention, write_json,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

METRIC = "val-core/alfworld/reward/mean@1"


def shards(path):
    path.mkdir(parents=True, exist_ok=True)
    for kind in ("model", "optim", "extra_state"):
        for rank in (0, 1):
            (path / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"test shard")


def checkpoint(root, step, score=None):
    path = root / f"global_step_{step}"
    shards(path / "actor")
    (path / "data.pt").write_bytes(b"dataloader")
    if score is not None:
        write_json(path / "validation_metric.json", {"step": step, "metric": METRIC, "value": score})
    return path


def steps(root):
    return sorted(int(p.name.split("_")[-1]) for p in root.glob("global_step_*") if complete_checkpoint(p))


class NativeSaveHarness:
    _save_checkpoint = PPOTrainer._save_checkpoint

    def __init__(self, root):
        self.config = OmegaConf.create({
            "trainer": {"keep_best_checkpoint": True, "max_actor_ckpt_to_keep": 5,
                        "default_local_dir": str(root), "default_hdfs_dir": None,
                        "test_freq": 10},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}},
                                  "rollout": {"val_kwargs": {"n": 1}}},
        })
        self.total_training_steps = 100
        self.global_steps = 10
        self.use_critic = False
        self.trainer_mode = "sync"
        self.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": self.global_steps})
        self.checkpoint_callback = SimpleNamespace(on_save=Mock())
        self.fail_save = False
        self.value = 1.0
        self.actor_rollout_wg = SimpleNamespace(save_checkpoint=Mock(side_effect=self.save_shards))

    def save_shards(self, local, remote, step, max_ckpt_to_keep):
        assert max_ckpt_to_keep is None  # Native worker cannot evict the protected best.
        if self.fail_save:
            raise OSError("disk failure")
        shards(Path(local))

    def _validate(self):
        return {self.config.trainer.get("best_checkpoint_metric") or METRIC: self.value}


class Trainer(ALFWorldCheckpointRetentionMixin, NativeSaveHarness):
    pass


def test_old_best_survives_rotation_and_restart(tmp_path):
    trainer = Trainer(tmp_path)
    for step in range(10, 71, 10):
        trainer.global_steps = step
        trainer.value = 10 if step == 10 else 1
        trainer._save_checkpoint()
        trainer._validate()
        assert len(steps(tmp_path)) <= 5
    assert steps(tmp_path) == [10, 40, 50, 60, 70]
    restarted = Trainer(tmp_path)
    restarted.global_steps = 80
    restarted._save_checkpoint()
    assert len(steps(tmp_path)) == 6  # Defer eviction until the new validation is known.
    restarted.value = 20
    result = restarted._validate()
    assert steps(tmp_path) == [40, 50, 60, 70, 80]
    assert result["checkpoint/best_step"] == 80
    assert json.loads((tmp_path / "best_checkpoint.json").read_text())["value"] == 20
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "80"
    assert restarted.config.trainer.max_actor_ckpt_to_keep == 5


def test_validation_between_save_intervals_saves_new_best(tmp_path):
    checkpoint(tmp_path, 10, 1)
    trainer = Trainer(tmp_path)
    trainer.global_steps = 15
    trainer.value = 2
    trainer.config.trainer.test_freq = 5
    trainer._validate()
    assert complete_checkpoint(tmp_path / "global_step_15")
    assert best_checkpoint(tmp_path, METRIC)["step"] == 15


def test_save_failure_never_prunes_and_restores_config(tmp_path):
    for step in range(10, 61, 10):
        checkpoint(tmp_path, step, 1)
    trainer = Trainer(tmp_path)
    before = OmegaConf.to_container(trainer.config)
    trainer.global_steps = 70
    trainer.fail_save = True
    with pytest.raises(OSError, match="disk failure"):
        trainer._save_checkpoint()
    assert OmegaConf.to_container(trainer.config) == before
    assert len(steps(tmp_path)) == 6


def test_incomplete_checkpoint_is_not_best_or_counted(tmp_path):
    broken = checkpoint(tmp_path, 10, 100)
    (broken / "actor/optim_world_size_2_rank_1.pt").unlink()
    for step in range(20, 81, 10):
        checkpoint(tmp_path, step, 1)
    best = best_checkpoint(tmp_path, METRIC)
    assert best["step"] == 20  # Ties retain the earliest complete checkpoint.
    prune_checkpoints(tmp_path, 5, best["step"])
    assert steps(tmp_path) == [20, 50, 60, 70, 80]
    assert broken.exists()  # Never automatically delete partially written checkpoints.


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_bad_metric_preserves_existing_best(tmp_path, value):
    checkpoint(tmp_path, 10, 2)
    trainer = Trainer(tmp_path)
    trainer.global_steps = 20
    trainer.value = value
    with pytest.raises(ValueError, match="non-finite"):
        trainer._validate()
    assert best_checkpoint(tmp_path, METRIC)["step"] == 10


def test_async_save_is_rejected(tmp_path):
    trainer = Trainer(tmp_path)
    trainer.config.actor_rollout_ref.actor.checkpoint.async_save = True
    with pytest.raises(ValueError, match="synchronous"):
        validate_checkpoint_retention(trainer.config)


def test_success_selection_ignores_old_reward_metric_and_keeps_highest_success(tmp_path):
    checkpoint(tmp_path, 10, 100)  # Old shaped reward is incomparable to success rate.
    trainer = Trainer(tmp_path)
    metric = "val-core/alfworld/success/mean@1"
    trainer.config.trainer.best_checkpoint_metric = metric
    for step, value in [(20, .3), (30, .2)]:
        trainer.global_steps = step
        trainer.value = value
        trainer._save_checkpoint()
        result = trainer._validate()
    assert best_checkpoint(tmp_path, metric)["step"] == 20
    assert result["checkpoint/best_validation_success_rate"] == .3
    assert "checkpoint/best_validation_reward" not in result

