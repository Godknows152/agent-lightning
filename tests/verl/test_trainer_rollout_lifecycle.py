from types import SimpleNamespace

from agentlightning.verl.trainer import AgentLightningTrainer


class _LifecycleRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def wake_up(self) -> None:
        self.calls.append("wake_up")

    def sleep(self) -> None:
        self.calls.append("sleep")

    def wake_up_replicas(self) -> None:
        self.calls.append("wake_up_replicas")

    def sleep_replicas(self) -> None:
        self.calls.append("sleep_replicas")


def _make_trainer(*, free_cache_engine: bool, checkpoint_manager: _LifecycleRecorder | None):
    trainer = object.__new__(AgentLightningTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=free_cache_engine))
    )
    trainer.async_rollout_manager = _LifecycleRecorder()
    if checkpoint_manager is not None:
        trainer.checkpoint_manager = checkpoint_manager
    return trainer


def test_rollout_lifecycle_uses_checkpoint_manager_when_cache_engine_is_freed() -> None:
    checkpoint_manager = _LifecycleRecorder()
    trainer = _make_trainer(free_cache_engine=True, checkpoint_manager=checkpoint_manager)

    trainer._set_rollout_replicas_awake(True)
    trainer._set_rollout_replicas_awake(False)

    assert checkpoint_manager.calls == ["wake_up_replicas", "sleep_replicas"]
    assert trainer.async_rollout_manager.calls == []


def test_rollout_lifecycle_preserves_manager_fallback_without_cache_engine() -> None:
    checkpoint_manager = _LifecycleRecorder()
    trainer = _make_trainer(free_cache_engine=False, checkpoint_manager=checkpoint_manager)

    trainer._set_rollout_replicas_awake(True)
    trainer._set_rollout_replicas_awake(False)

    assert checkpoint_manager.calls == []
    assert trainer.async_rollout_manager.calls == ["wake_up", "sleep"]


def test_rollout_lifecycle_falls_back_when_checkpoint_manager_is_unavailable() -> None:
    trainer = _make_trainer(free_cache_engine=True, checkpoint_manager=None)

    trainer._set_rollout_replicas_awake(True)
    trainer._set_rollout_replicas_awake(False)

    assert trainer.async_rollout_manager.calls == ["wake_up", "sleep"]
