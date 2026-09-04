"""ALFWorld-isolated old-VERL entrypoint with task-specific SwanLab metrics."""
from __future__ import annotations

from verl.trainer import main_ppo as _base_main


class ALFWorldTaskRunner(_base_main.TaskRunner):
    """Install ALFWorld metric aggregation inside the remote trainer actor."""

    def run(self, config):
        from verl.trainer.ppo import ray_trainer
        from alfworld_baseline.metrics import compute_alfworld_penalty_metrics

        # Ray executes TaskRunner in another process, so the hook must be
        # installed here rather than only in the launcher process.
        # The call site in old-VERL is shared with restoration experiments;
        # replace its result only in this remote ALFWorld trainer process so
        # SwanLab receives ALFWorld names rather than restoration names.
        ray_trainer.compute_restoration_penalty_metrics = compute_alfworld_penalty_metrics
        return super().run(config)


# run_ppo resolves this module global when it creates the remote actor.  The
# replacement therefore applies only to this ALFWorld entrypoint/process.
_base_main.TaskRunner = ALFWorldTaskRunner

main = _base_main.main


if __name__ == "__main__":
    main()
