"""Task metrics must use explicit outcomes and one vote per trajectory."""
from types import SimpleNamespace
import asyncio

from omegaconf import OmegaConf
import pytest

from alfworld_baseline.main_ppo import ALFWorldMetricsMixin
from alfworld_baseline.validation_logging import ALFWorldValidationLoggingMixin
from test_native_step_pipeline import episode
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class NativeMetricHarness:
    _val_metrics_update = PPOTrainer._val_metrics_update


class ValidationHarness(ALFWorldValidationLoggingMixin, NativeMetricHarness):
    config = OmegaConf.create({
        "variables": {"VALIDATION_TASK_METRICS": True},
        "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
    })


def test_validation_promotes_true_success_and_base_reward_not_positive_shaped_reward():
    rewards = {"reward": [11, 2, -1], "base_reward": [10, -.5, -2], "success": [1, 0, 0]}
    result = ValidationHarness()._val_metrics_update(
        ["alfworld"] * 3, ["a", "b", "c"], rewards, [1, 1, 1]
    )
    assert result["val-core/alfworld/success/mean@1"] == pytest.approx(1 / 3)
    assert result["val-core/alfworld/base_reward/mean@1"] == 2.5
    assert result["val-aux/alfworld/reward/mean@1"] == 4
    assert "val-core/alfworld/reward/mean@1" not in result
    assert rewards["reward"] == [11, 2, -1]


def test_validation_rejects_missing_explicit_success():
    with pytest.raises(ValueError, match="explicit validation task metric"):
        ValidationHarness()._val_metrics_update(
            ["alfworld"], ["a"], {"reward": [2], "base_reward": [-.5]}, [1]
        )


def test_training_critic_metrics_exclude_padding_and_intermediate_rows(monkeypatch):
    import transfer_queue as tq

    class Base:
        def _compute_metrics(self, batch, metrics, *args):
            metrics["critic/rewards/mean"] = 123  # Existing training metric is preserved.

    class Trainer(ALFWorldMetricsMixin, Base):
        pass

    final_success = {"alfworld_is_final_step": True, "alfworld_terminal_reason": "success",
                     "alfworld_base_reward": 10}
    final_failure = {"alfworld_is_final_step": True, "alfworld_terminal_reason": "decision_limit",
                     "alfworld_base_reward": -2}
    rows = [{**final_success, "alfworld_is_final_step": False}] * 4 + [final_success, final_failure]
    keys = [f"episode_0_{i}" for i in range(5)] + ["failure_0_0"]

    def read(**kwargs):
        assert kwargs["keys"] == keys
        return {"extra_fields": rows}

    monkeypatch.setattr(tq, "kv_batch_get", read)
    batch = SimpleNamespace(keys=keys + ["padding"], tags=[{}] * 6 + [{"is_padding": True}], partition_id="train")
    result = {}
    Trainer()._compute_metrics(batch, result, {}, 261, 1)
    assert result["critic/base_reward/mean"] == 4
    assert result["critic/success_rate"] == .5
    assert result["critic/rewards/mean"] == 123


@pytest.mark.parametrize("success", [True, False])
def test_episode_emits_task_metrics_even_without_auxiliary_rewards(success):
    loop, _, _, _ = episode(success=success)
    rows = asyncio.run(loop.run({}, raw_prompt=[]))
    info = rows[-1].extra_fields["reward_extra_info"]
    assert info["base_reward"] == pytest.approx(10 if success else -.6)
    assert info["success"] == float(success)
