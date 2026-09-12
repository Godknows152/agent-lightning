"""ALFWorld-isolated old-VERL entrypoint with task-specific SwanLab metrics."""
from __future__ import annotations

from verl.trainer import main_ppo as _base_main

from alfworld_baseline.ray_startup import install_ray_agent_port_guard

# Install before the shared VERL entrypoint calls ray.init().
install_ray_agent_port_guard()


class ALFWorldTaskRunner(_base_main.TaskRunner):
    """Install ALFWorld metric aggregation inside the remote trainer actor."""

    def run(self, config):
        from verl.trainer.ppo import ray_trainer
        from alfworld_baseline.metrics import compute_alfworld_rollout_metrics
        from alfworld_baseline.budget import configure_environment_driven_rollout

        configure_environment_driven_rollout(config)

        # Ray executes TaskRunner in another process, so install the
        # task-specific ALFWorld rollout metrics in that remote trainer.
        ray_trainer.compute_rollout_metrics = compute_alfworld_rollout_metrics
        return super().run(config)


# run_ppo resolves this module global when it creates the remote actor.  The
# replacement therefore applies only to this ALFWorld entrypoint/process.
_base_main.TaskRunner = ALFWorldTaskRunner

main = _base_main.main


if __name__ == "__main__":
    main()
