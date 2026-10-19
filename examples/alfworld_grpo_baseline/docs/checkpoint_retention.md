# ALFWorld checkpoint retention

The shared ALFWorld configuration now sets `trainer.max_actor_ckpt_to_keep: 5`
and `trainer.keep_best_checkpoint: true`. The five slots include the checkpoint
with the highest validation mean reward and the most recent remaining checkpoints.
An exact tie retains the earlier best. This is validation reward, not training
reward or success count.

`trainer.best_checkpoint_metric: null` selects
`val-core/alfworld/reward/mean@N`, where N is the configured validation rollout count.
An explicit metric name can override it; missing or non-finite values stop the run
rather than silently discard the best checkpoint.

The ALFWorld-local trainer mixin disables veRL worker FIFO deletion. After native
synchronous saving, it checks all FSDP model, optimizer and extra-state shards and
the dataloader state. Native V1 saves before validation, so eviction waits until
validation finishes: a sixth checkpoint can temporarily exist during validation.
The winning checkpoint and the latest checkpoint are protected, and obsolete
checkpoint directories are removed in their entirety. A best result outside the
normal save interval is saved as well.

Each evaluated checkpoint has `validation_metric.json`; the run root contains
`best_checkpoint.json`. These records survive restarts. SwanLab receives
`checkpoint/best_step` and `checkpoint/best_validation_reward`. This policy requires
local, synchronous full FSDP checkpoints; asynchronous or remote saves are rejected.
Historical checkpoints without these records have unknown validation scores and
must have records populated before relying on best-checkpoint protection on resume.
The new policy does not retroactively prune inactive experiments.
