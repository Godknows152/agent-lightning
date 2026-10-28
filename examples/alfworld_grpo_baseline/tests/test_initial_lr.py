"""Verify the actual optimizer schedule, including Hydra conversion and restore."""
from dataclasses import replace
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch

from alfworld_baseline.branch_entropy_engine import BranchEntropyEngine
from alfworld_baseline.optimizer_config import ALFWorldFSDPOptimizerConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.optimizer import FSDPOptimizerConfig


def make_scheduler(config):
    engine = object.__new__(BranchEntropyEngine)
    engine.optimizer_config = config
    engine.rank = 0
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=config.lr)
    return optimizer, engine._build_lr_scheduler(optimizer)


def configured_optimizer():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "config/alfworld/qwen35_2b/action_entropy_1019"), version_base=None,
    ):
        config = compose(config_name="alfworld_config_2gpu")
    config.actor_rollout_ref.actor.optim.total_training_steps = 666
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor.optim, ALFWorldFSDPOptimizerConfig)
    return actor.optim


def test_actual_profile_starts_at_peak_without_warmup_and_decays_to_floor():
    optimizer, scheduler = make_scheduler(configured_optimizer())
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(666):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] == pytest.approx(3e-6)
    assert values[333] == pytest.approx(2e-6)
    assert values[666] == pytest.approx(1e-6)
    assert all(a > b for a, b in zip(values[:-1], values[1:]))


def test_nonzero_warmup_schedule_restores_without_restarting_warmup():
    config = replace(configured_optimizer(), init_lr_ratio=1 / 3, lr_warmup_steps=133)
    optimizer, scheduler = make_scheduler(config)
    for _ in range(45):
        optimizer.step()
        scheduler.step()
    optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
    restored_optimizer, restored_scheduler = make_scheduler(config)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_existing_profiles_keep_zero_start():
    optimizer, scheduler = make_scheduler(FSDPOptimizerConfig(
        lr=3e-6, total_training_steps=666, lr_warmup_steps_ratio=0.2,
        lr_scheduler_type="cosine", min_lr_ratio=1/3,
    ))
    assert optimizer.param_groups[0]["lr"] == 0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6/133)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_initial_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="init_lr_ratio"):
        ALFWorldFSDPOptimizerConfig(init_lr_ratio=ratio, lr_scheduler_type="cosine")
