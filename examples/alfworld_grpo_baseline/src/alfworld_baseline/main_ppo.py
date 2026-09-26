"""ALFWorld entrypoint for native veRL V1, with isolated rollout metrics."""
from __future__ import annotations

import hydra
from omegaconf import OmegaConf
import ray

from alfworld_baseline.backend import assert_native_verl, native_verl_root
from alfworld_baseline.ray_startup import install_ray_agent_port_guard

assert_native_verl()
from verl.trainer import main_ppo as _base_main
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging


class ALFWorldMetricsMixin:
    """Read trajectory telemetry from the V1 queue, excluding padding rows."""

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        import transfer_queue as tq
        from types import SimpleNamespace
        from alfworld_baseline.metrics import compute_alfworld_rollout_metrics

        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        keys = [key for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding", False)]
        if not keys:
            return
        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        # TensorDict indexing unwraps NonTensorStack to a LinkedList; numpy
        # arrays used by older queues are iterable as well.
        rows = [row for row in data["extra_fields"] if row.get("alfworld_is_final_step", True)]
        fields = {field for row in rows for field in row if field.startswith("alfworld_")}
        if fields:
            columns = {field: [row.get(field) for row in rows] for field in fields}
            columns["alfworld_valid_tool_call_count"] = [
                row.get("alfworld_valid_tool_call_count", 0) or 0 for row in rows
            ]
            metrics.update(compute_alfworld_rollout_metrics(SimpleNamespace(non_tensor_batch=columns)))


class ALFWorldTaskRunner:
    """Compose native V1 trainer/manager classes without inheriting a Ray actor.

    Follow the V1 TaskRunner lifecycle with ALFWorld metrics and worker selection.
    Shared backend files are unchanged.
    """

    def run(self, config):
        assert_native_verl()
        configure_verl_logging()
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ, get_trainer_cls
        from alfworld_baseline.validation_logging import ALFWorldValidationLoggingMixin
        from alfworld_baseline.workers import ALFWorldWorkerMixin

        base_trainer = get_trainer_cls(config.trainer.v1.trainer_mode)
        if OmegaConf.select(config, "variables.TRAINING_BACKEND", default="trajectory") == "gigpo_grpo":
            # Register only the new estimator, without replacing native GRPO.
            # This import must run inside the Ray driver as well as CPU tests.
            from alfworld_baseline import step_advantage  # noqa: F401
        trainer_cls = type(
            "ALFWorldTrainer",
            (ALFWorldWorkerMixin, ALFWorldMetricsMixin, ALFWorldValidationLoggingMixin, base_trainer),
            {},
        )
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = trainer_cls(config=config)
            trainer.init()
            manager_fqn = config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
            manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
            manager = manager_cls.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            from alfworld_baseline.tracking import swanlab_resume
            with swanlab_resume(config):
                trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path=str(native_verl_root() / "verl/trainer/config"), config_name="ppo_trainer", version_base=None)
def main(config):
    assert_native_verl()
    if not config.trainer.use_v1:
        raise ValueError("ALFWorld's native entrypoint requires trainer.use_v1=true")
    from alfworld_baseline.budget import configure_environment_driven_rollout, validate_native_step_config
    from alfworld_baseline.resume import validate_native_resume
    from alfworld_baseline.full_finetuning import validate_full_finetuning
    validate_full_finetuning(config)
    validate_native_step_config(config)
    configure_environment_driven_rollout(config)
    # Validate the effective config, including CLI overrides, before starting Ray.
    omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    validate_native_resume(config)
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    # Native run_ppo converts Ray kwargs without resolve=True. Resolve here,
    # before workers inherit literal ${oc.env:...} compiler paths.
    OmegaConf.resolve(config.ray_kwargs.ray_init)
    install_ray_agent_port_guard()
    _base_main.run_ppo(config, task_runner_class=ray.remote(ALFWorldTaskRunner))


if __name__ == "__main__":
    main()
