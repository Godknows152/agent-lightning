# ALFWorld entropy experiment profiles

## branch_entropy_1013

Launch with `bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_branch_entropy_1013.sh`.
The separate configuration at `config/alfworld/qwen35_2b/branch_entropy_1013/alfworld_config_2gpu.yaml`
starts from the action-explicit merged SFT model, with resume disabled. It uses
branch entropy 0.0005 (half of branch_entropy_1009), ordinary entropy 0.001,
one sampled validation trajectory per task, and the latest resident-reference
execution settings. Other optimizer, reward, prompt and training sampling settings
match no_branch_entropy_1012. Outputs and the SwanLab experiment are named
`branch_entropy_1013`. Default GPUs are 0 and 1.

## 新动作熵1019

`qwen35_2b_action_entropy_1019.sh` uses the full model merged from raw-model RL
checkpoint 150. As of the fresh restart on 2026-10-21, branch entropy coefficient
is 0.0007; ordinary entropy and KL
coefficients are both 0.001. Both thinking/action consistency bonuses are zero.
Training uses three epochs, eight rollouts, and the existing trajectory GRPO
reward. Output directory and SwanLab experiment name are `新动作熵1019`.
The base model lives outside the disposable run outputs at
`/home/LXJ/Python_Projects/Models/Qwen3.5-2B-ALFWorld-raw-RL-step150`.
The previous same-name run's checkpoints, rollouts, logs, diagnostics and cloud
record were deleted before restarting from this unchanged base model.
Mention selection uses `first_non_listing`, with generic-function glossary
filtering and case-insensitive command matching. For rule details and historical
validation, see `action_listing_filter.md`.

The fresh restart on 2026-10-21 removes warmup entirely: both warmup steps and
warmup ratio are zero. The first optimizer update uses the peak LR `3e-6`,
followed by half-cosine decay to `1e-6` at step 666. `init_lr_ratio=0` selects
the native scheduler path; with zero warmup this starts at the peak, not zero.
This replaces the 2026-10-20 schedule that warmed up from `1e-6` to `3e-6`
over 133 steps. The previous run's optimizer and scheduler states are discarded.

On 2026-10-23 the run was stopped and resumed from `global_step_190`, retaining
optimizer/scheduler state and SwanLab run `z4npvg4k`. The added
`variables.BRANCH_ENTROPY_ELIGIBLE_REWARD=0.05` awards each decision once only when
the actual branch alignment returns `status=matched` and at least one branch node.
Missing mentions, listing-only mentions, illegal actions, failed token alignment,
and single-candidate matches with no branch receive zero. The existing per-step
bonuses remain disabled; branch entropy stays at 0.0007. Eligibility bonuses sum
into the trajectory reward before native GRPO normalization. Per-row eligibility
and reward are retained in rollout metadata; episode bonus totals/counts and the
base reward are tracked separately under `alfworld_reward/`.
Pre-resume rollout files for steps 191–195 are backed up in
`resume_history/20261023_step190_eligible_reward/` before those steps are rerun.
SwanLab preserves existing scalar points at these steps on same-ID resume;
new reward series can record from 191, while existing series continue after 195.
Use the new local training log/rollouts for complete replay-step comparisons.

On 2026-10-24 the next resume uses `global_step_260` and a new same-name SwanLab
run (`variables.SWANLAB_NEW_RUN_ON_RESUME=true` for this launch only). Training
rewards and optimizer settings are unchanged. `VALIDATION_TASK_METRICS=true`
promotes `val-core/alfworld/success/mean@1` and
`val-core/alfworld/base_reward/mean@1`; shaped validation reward remains under
`val-aux`. Checkpoint retention now maximizes validation success, logging
`checkpoint/best_validation_success_rate`. The five retained checkpoints'
selection records were migrated from their existing validation JSONL files,
using explicit success flags and one final outcome per trajectory. Previous
selection records and rollouts 261–263 are preserved in
`resume_history/20261024_step260_task_metrics/`.

Training adds `critic/base_reward/mean` and `critic/success_rate`, both averaged
over final non-padding trajectory rows. Base reward excludes the eligibility
and thinking/action bonuses but retains the original task penalties. Success
uses the environment terminal reason, never the sign of the shaped reward.
The reward adapter always emits both task metrics, including when bonuses are off.

That short-lived step260 resume (`unrqpdza`) was stopped and deleted on 2026-10-24
before completing an update. Its local/cloud tracking records, launch logs,
Hydra output and Ray session were removed. Shared 1019 checkpoint selection
records and the prior `z4npvg4k` identity were restored from the pre-resume backups.

## 新动作熵1024

Launch with `bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_action_entropy_1024.sh`.
This independent profile copies the current 1019 training configuration, changing
only the experiment/output name to `新动作熵1024` and `trainer.resume_mode=disable`.
It starts at step0 from the unchanged
`/home/LXJ/Python_Projects/Models/Qwen3.5-2B-ALFWorld-raw-RL-step150` full model,
with fresh optimizer/scheduler state. Branch entropy is 0.0007, eligibility reward
is 0.05 per eligible decision, and LR starts at 3e-6 with zero warmup. Validation
success selects the best checkpoint; validation task reward and the two added
critic task metrics remain enabled. Total training length is 666 steps.
