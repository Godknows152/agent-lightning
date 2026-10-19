"""Validate saved experiment configurations without initializing models."""
from pathlib import Path

from hydra import compose, initialize_config_dir

from alfworld_baseline.checkpoint_retention import validate_checkpoint_retention


def test_fresh_1013_config():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "config/alfworld/qwen35_2b/branch_entropy_1013"), version_base=None):
        cfg = compose(config_name="alfworld_config_2gpu")
    validate_checkpoint_retention(cfg)
    assert cfg.trainer.max_actor_ckpt_to_keep == 5
    assert cfg.trainer.keep_best_checkpoint
    assert cfg.trainer.resume_mode == "disable"
    assert cfg.trainer.resume_from_path is None
    assert cfg.trainer.experiment_name == "branch_entropy_1013"
    assert cfg.actor_rollout_ref.branch_entropy.enabled
    assert cfg.actor_rollout_ref.branch_entropy.coefficient == 0.0005
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert cfg.actor_rollout_ref.reference_resident
    assert Path(cfg.actor_rollout_ref.model.path).is_dir()
