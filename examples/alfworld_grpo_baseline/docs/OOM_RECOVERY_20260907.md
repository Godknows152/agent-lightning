# Qwen3.5-2B ALFWorld OOM recovery — 2026-09-07

## Scope and acceptance

User requested iterative parameter tuning until the full training run actually starts successfully.
Acceptance: completed Actor updates and finite training metrics, not merely a launch PID.
Keep model/data/seed, 150 full-run steps, 4 rollouts per prompt, reward, KL=0.004,
entropy coefficient=0.02 unchanged. Initially retained the 50-decision / 256-token
budgets. After repeated memory-only attempts failed, reduce Qwen3.5-2B to 128
tokens per turn and then 32 decisions. These are explicit behavioral changes;
results cannot be compared as an unchanged 50-decision baseline.
Preserve all failed logs and unrelated GPU processes. Do not resume failed checkpoints.

## Evidence

- `alfworld_v1_seed0_20260907_090424.log`: Actor backward OOM while allocating
  logits gradients (26.10 GiB on GPU 0; 31.35 GiB on GPU 1), not a KV-cache allocation.
- Turn-context replay expands each trajectory into its recorded prompt+answer sequences.
  Actor micro-batch size counts trajectories, not expanded turns/tokens.
- The current shared rollout memory fraction was already 0.2 when this task began;
  this task has not edited the shared configuration.

## Attempts (Asia/Shanghai)

1. 09:32:44 — Actor `ppo_micro_batch_size_per_gpu: 2 -> 1`.
   Isolated launch failed before Ray initialization because the chosen temporary
   directory made the Unix socket path exceed 107 bytes. Retained this attempt;
   shortened the isolated temp directory for subsequent launches.
2. 09:33:54 — micro-batch 1, all loss settings unchanged.
   Unit: `alfworld-qwen35-2b-memory-mb1-20260907_093354.service`.
   Output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0/memory_mb1_20260907_093354`.
   Failed during first Actor update around 09:44. GPU 0 requested 17.81 GiB
   with 13.33 GiB free; 63.85 GiB allocated by PyTorch. Micro-batch 1 alone
   does not guarantee sufficient memory for long replayed trajectories.
3. 09:46:14 — retain micro-batch 1 and enable
   `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing: true`.
   This is an engine memory/recomputation setting, not a change to entropy weight.
   The engine reads this nested setting; the similarly named top-level legacy
   Actor flag is not the one used by this engine.
   Unit: `alfworld-qwen35-2b-memory-ec-20260907_094614.service`.
   Output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0/memory_mb1_ec_20260907_094614`.
   Failed around 09:56: GPU 0 requested 16.90 GiB with 16.28 GiB free;
   60.89 GiB allocated by PyTorch. Checkpointing alone still leaves too little
   headroom for the first update. The nested engine flag was confirmed true
   in the effective config.

Each attempt has `launch_model_config.yaml`, `launch_common_config.yaml`,
`launch_tool_config.yaml`, `log/training.log`, and its own SwanLab output.
Successful unit should remain running after verification. No old artifacts have
been deleted; no pre-existing user changes have been committed or reverted.

## Focused entropy allocation probe

A temporary GPU-0-only synthetic probe used BF16 logits `[8192, 248320]`, the
installed FlashAttention cross-entropy, `torch.compile(dynamic=True)`, entropy
coefficient 0.02, and checkpointed entropy. It did not load/change model weights.

| Method | Forward allocated | Peak allocated |
| --- | ---: | ---: |
| Entropy checkpointing | 3.789 GiB | 11.367 GiB |
| Entropy checkpointing + existing chunking | 3.789 GiB | 13.301 GiB |

The existing chunked implementation casts chunks to FP32. This probe does not
measure the full model, but it provides no reason to enable chunking as a memory
fix here; the chunking flag remains false.

4. 09:59:29 — retain micro-batch 1 and entropy checkpointing; enable
   `actor_rollout_ref.model.enable_activation_offload: true` to offload saved
   layer activations to CPU. Model weights, task budgets, and loss coefficients
   remain unchanged. Host memory available before launch: approximately 410 GiB.
   Unit: `alfworld-qwen35-2b-memory-ao-20260907_095929.service`.
   Output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0/memory_mb1_ec_ao_20260907_095929`.
   Failed around 10:10 on a longer trajectory: cross-entropy backward requested
   23.46 GiB with 22.91 GiB free; 54.25 GiB allocated by PyTorch. Activation
   offload was effective but did not provide enough headroom for all trajectories.

5. 10:12:09 — retain micro-batch 1, entropy checkpointing, and activation offload;
   reduce the Qwen3.5-2B-only `max_new_tokens_per_turn: 256 -> 128`.
   Align `data.max_response_length: 12800 -> 6400`; the entrypoint recomputes
   this storage capacity as 50 * 128. Decision count remains 50; no prompt,
   reward, group size, KL/entropy coefficient, seed, or model changes.
   This cap can truncate overly long single-turn generations and is an explicit
   behavioral change, unlike the preceding memory-only switches.
   Unit: `alfworld-qwen35-2b-memory-t128-20260907_101209.service`.
   Output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0/memory_mb1_t128_20260907_101209`.
   Failed around 10:22: GPU 1 requested 21.68 GiB with 4.90 GiB free;
   72.28 GiB allocated by PyTorch. Response-length reduction alone does not
   sufficiently bound the repeated prompt tokens in turn-context replay.

6. 10:25:03 — reduce only the Qwen3.5-2B decision budget `max_steps: 50 -> 32`,
   retaining 128 generated tokens per decision. Effective response capacity is
   now 4096. Retain micro-batch 1, entropy checkpointing, and activation offload.
   This bounds repeated per-turn prompt replay, not just generated answers.
   Full training remains 150 optimizer steps; data batch/group size and loss
   coefficients unchanged. The user-facing progress explicitly disclosed that
   this is no longer the same 50-decision evaluation budget.
   Unit: `alfworld-qwen35-2b-memory-s32-20260907_102503.service`.
   Output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0/memory_mb1_s32_t128_20260907_102503`.
   Failed during the first Actor update around **10:46 on September 7, 2026**:
   GPU 0 requested 16.54 GiB with 16.43 GiB free while 60.72 GiB was already
   allocated by PyTorch.

The composed-config regression expectations now follow the working-tree
Qwen3.5-2B profile (`16` decisions × `256` generated tokens = `4096` storage).
General 50-step/256-token budget tests remain unchanged to preserve coverage of
the original runtime contract.

## Running candidate observed after continuation (14:22 Asia/Shanghai)

A separate full-run candidate was already launched at **11:30 on September 7, 2026**
with the then-current Qwen3.5-2B configuration. It is not one of the isolated
`memory_*` attempts above:

- Main PID: `2000820`; output: `outputs/alfworld/qwen35_2b/v1/2gpu/seed0`.
- Effective run settings printed in the log: `enable_thinking=false`,
  `max_steps=16`, `max_new_tokens_per_turn=256`, `data.max_response_length=4096`,
  KL coefficient `0.004`, entropy coefficient `0.02`, and `ppo_micro_batch_size_per_gpu=1`.
- By **14:20 on September 7, 2026**, it had completed actor updates through
  `training/global_step=30` and saved `global_step_10`, `global_step_20`, and
  `global_step_30`.
- Step 30 had finite actor loss (`0.0985748`), gradient norm (`27.125`),
  reward mean (`-0.0796875`), and validation reward mean (`-0.0171429`).
- No CUDA OOM or traceback has appeared after the first 30 updates. The process
  remained alive at **14:22** and was actively using the actor and inference
  workers, so it was not stopped.

The working-tree source/config edits made later at approximately **13:46–13:47**
were not loaded by this already-running process. Its metrics still include the
legacy protocol-penalty fields, which is expected for this launch. A new run is
required to evaluate the newer no-synthetic-penalty implementation; do not
compare that future run directly with this candidate without recording the
revision and reward semantics.

## Penalty-removal cleanup (14:30 Asia/Shanghai)

The legacy candidate was stopped before starting a replacement run:

- Main process `2000820` was sent `SIGTERM`; no ALFWorld trainer, Ray runner, or
  SGLang scheduler processes remained afterward.
- The cloud SwanLab run `Godknows/ALFWorldRL`, run `nurrq7ej` (experiment
  `alfworld_qwen35_2b_v1_seed0`) was deleted successfully. The project query then
  returned zero runs.
- Local SwanLab run bundles and `.swanlab_experiment.json` were removed from the
  Qwen3.5-2B output tree. Checkpoints, rollout JSONL, and ordinary training logs
  were retained for audit; they are not active SwanLab records.
- The five isolated failed-attempt local SwanLab bundles were also removed.

The source now disables ALFWorld penalty telemetry with
`trainer.enable_penalty_logging: false`. The shared PPO backend keeps its
restoration behavior by default, but skips both penalized-sample logging and
penalty metric aggregation for ALFWorld. Only termination counters are exposed
through the task-specific rollout-metrics hook.

## Clean no-penalty relaunch result (14:41 Asia/Shanghai)

A clean Qwen3.5-2B launch was attempted from the updated working tree under
`outputs/alfworld/qwen35_2b/v1/2gpu/seed0_no_penalty_20260907`, using the same
`ALFWorldRL` project and the no-penalty configuration. Preflight confirmed
`enable_penalty_logging=false`. The launch failed before rollout/update because
SGLang's rank-0 scheduler exited with code 1 during initialization; no Actor
step or SwanLab run was created. The cloud project still reports zero runs.
This startup failure is independent of the ALFWorld reward/penalty removal.

## Subsequent two-category penalty design (September 7, 2026)

After the no-penalty cleanup and the failed SGLang-only relaunch documented above,
ALFWorld reward handling was changed to the current two-category policy:

- an output with no parsed, structurally complete tool call receives penalty 1,
  `-0.1`;
- a parsed call that cannot be executed (unknown tool, invalid parameter schema,
  or inadmissible action) receives penalty 2, `-0.1`;
- the categories are checked in that order and are mutually exclusive per
  decision step, while counts accumulate across a trajectory;
- the task-specific SwanLab hook exposes only
  `alfworld/valid_tool_call_count/{min,max,mean}` for the number of trajectory
  steps that actually called the environment, and
  `alfworld_penalty/no_tool_call_count` and
  `alfworld_penalty/invalid_tool_call_count` in addition to termination counters;
- the legacy `num_turns/*` and `tool_call_counts/*` SwanLab aliases are not
  emitted for ALFWorld;
- generic restoration `penalty_records` and generic penalty metrics remain
  disabled for ALFWorld.

The implementation is in `src/alfworld_baseline/agent_loop.py` and
`src/alfworld_baseline/metrics.py`; the Qwen3.5-2B training restart remains a
separate concern because the earlier failure occurred during SGLang
initialization before rollout.
