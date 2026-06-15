# Hierarchical Multi-Agent Image Restoration

This example implements stages A-H of the design in `分层多智能体图像修复系统设计草案.md`. It provides validated protocols, a deterministic workflow, Agent Lightning tracing, a real restoration/IQA path isolated in the `verl` conda environment, four parallel GLM-4.1V expert interfaces, and trajectory-level GRPO training from the four SFT LoRA adapters.

## Current Scope

- One deterministic degradation diagnosis per trajectory.
- Four parallel expert identities with one shared restoration tool registry.
- Scripted expert actions using the canonical `restore_image(action)` protocol.
- Copy-based mock components for CPU regression tests.
- Real restoration workers invoked through the isolated `verl` environment.
- TOPIQ-NR, MUSIQ, and NIQE evaluation with normalized weighted aggregation.
- A separate GRPO reward calibration for MANIQA, direction-normalized NIQE,
  CLIP-IQA, and TOPIQ-NR.
- Single-call GLM-4.1V diagnosis through a vLLM OpenAI-compatible endpoint.
- Strict VLM parsing with raw-response and token-ID retention.
- Independent `predicted_strict` and `oracle_observe` routing modes.
- Diagnosis API, parsing, classification, confusion-matrix, and latency metrics.
- Strict restoration-expert Hermes parsing with raw response, reasoning, token, latency, and error retention.
- Independent `replay` and `vlm_strict` expert decision modes using the same parser and Controller path.
- Replay-driven multi-turn tool execution for pre-training feasibility validation.
- Four unique expert resource names with independent future policy paths and one shared tool schema.
- Four-category Replay and strict-VLM validation matrices with isolated output directories and traces.
- Best-image rollback, stop handling, failure limits, and trajectory JSON export.
- Agent Lightning operation/object spans and exactly one final rollout reward.
- Image-only, no-GT GRPO seed manifests with 1,000 train and 200 validation images per expert.
- Four-metric IQA reward using MANIQA, direction-normalized NIQE, CLIP-IQA, and TOPIQ-NR.
- Step IQA rewards summed into one trajectory reward, including stop and failure shaping.
- Agent Lightning VERL with `trace_aggregator.level=trajectory_transition`, one latest image per turn, trajectory-level GRPO advantages, GLM-4.1V multimodal compatibility patches, and LoRA-only policy updates.
- Four independent GRPO run configurations and SwanLab experiment names under the shared `image-restoration-multi-agent` project.
- A Stage H persistent tool service with all IQA metrics on `cuda:0` and all restoration models on `cuda:1`.

All four experts receive the same 16 restoration actions: the 12 copied `verl` tools plus NAFNet denoising, FocalNet dehazing/desnowing, and MB-TaylorFormer dehazing. No expert-specific action filtering is applied.

## Requirements

Use the repository environment. In the local setup used for this example:

```bash
conda activate agent-lightning
```

The controller and Agent Lightning runner use `agent-lightning`. Restoration and IQA processes use `verl`; the main environment never imports their PyTorch, BasicSR, or model packages. Stages D-G retain the one-shot subprocess path. Formal Stage H launches a local HTTP service whose model-owning children remain alive throughout the GRPO run.

Stage E serves `/home/LXJ/Python_Projects/Models/GLM-4.1V-9B-Thinking` with vLLM `0.10.2` from the `agent-lightning` environment. The model is exposed through the standard OpenAI Chat Completions API. Guided JSON decoding is intentionally disabled so the smoke test observes the untrained model's actual format-following behavior.

Stage H also requires the `swanlab` Python package in `agent-lightning`; the verified environment uses SwanLab `0.8.2`.

The launch script enables `--enable-auto-tool-choice --tool-call-parser hermes` for both diagnosis and expert inference. Prompts explicitly embed their function schema inside `<tools>` tags because the GLM-4.1V checkpoint's bundled chat template does not render the OpenAI `tools` request field by itself. Diagnosis uses `<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"fog","visual_evidence":[...]}}</tool_call>`; experts use `<tool_call>{"name":"restore_image","arguments":{"action":"..."}}</tool_call>`. The controller derives `route_to` from `primary_type`.

## Smoke Test

From the repository root:

```bash
conda run -n agent-lightning \
  python examples/image_restoration_multi_agent/smoke_test.py
```

The command prints the final reward, selected expert, best image, trajectory path, and recorded span names.

## Real Model Smoke Test

The default real smoke test runs FocalNet dehazing followed by TOPIQ-NR, MUSIQ, and NIQE scoring:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n agent-lightning \
  python examples/image_restoration_multi_agent/real_smoke_test.py \
  --input External_Tools/inputs/fog.png \
  --action focalnet_dehaze
```

The command starts model subprocesses in `verl`. Override the isolated interpreter with `IMAGE_RESTORATION_PYTHON=/path/to/verl/bin/python` when Conda is not on `PATH`.

## Stage E VLM Smoke Test

Start GLM-4.1V on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agent-lightning \
  bash examples/image_restoration_multi_agent/serve_glm4v.sh
```

In another terminal, run the complete path on a different GPU:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/stage_e_smoke_test.py \
  --input External_Tools/inputs/fog.png \
  --degradation-type fog \
  --action focalnet_dehaze \
  --routing-mode oracle_observe
```

`CUDA_VISIBLE_DEVICES=1` makes physical GPU 1 appear as `cuda:0` to the isolated `verl` restoration and IQA subprocesses. Use `--routing-mode predicted_strict` to verify that an invalid VLM response terminates as `diagnosis_failed` without selecting a fallback expert or invoking a restoration worker.

## Stage F Expert Smoke Test

Keep the same vLLM service running. First validate the complete multi-turn execution path with replayed Hermes decisions and real restoration/IQA models:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/stage_f_smoke_test.py \
  --input External_Tools/inputs/fog.png \
  --degradation-type fog \
  --routing-mode oracle_observe \
  --expert-decision-mode replay \
  --actions focalnet_dehaze stop
```

Then validate the untrained model's real expert request and controlled failure path:

```bash
conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/stage_f_smoke_test.py \
  --input External_Tools/inputs/fog.png \
  --degradation-type fog \
  --routing-mode oracle_observe \
  --expert-decision-mode vlm_strict \
  --actions stop
```

Both paths use the same strict parser and Controller. Replay proves that valid Hermes decisions can execute the multi-turn tool/IQA loop; `vlm_strict` records the untrained model's actual response and stops before the Worker when the response is invalid.

## Stage G Four-Expert Matrix

With the vLLM service still running, validate all four Oracle routes with real restoration and IQA models:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/stage_g_smoke_test.py \
  --expert-decision-mode replay \
  --output-root examples/image_restoration_multi_agent/artifacts/stage_g_replay_real
```

The default matrix executes:

```text
fog       -> focalnet_dehaze -> IQA -> stop
snow      -> focalnet_desnow -> IQA -> stop
rain      -> turbo_rain      -> IQA -> stop
low_light -> hvicidnet       -> IQA -> stop
```

Validate the four real expert interfaces separately from task quality:

```bash
conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/stage_g_smoke_test.py \
  --expert-decision-mode vlm_strict \
  --max-tokens 256 \
  --output-root examples/image_restoration_multi_agent/artifacts/stage_g_vlm_strict
```

`--max-tokens 256` is a smoke-test override that shortens the known untrained thinking loop. It does not change the checked-in stage G generation setting.

## Unit Tests

```bash
conda run -n agent-lightning \
  python -m pytest -q examples/image_restoration_multi_agent/tests
```

## Stage H GRPO Data

Build and validate the four no-GT seed datasets:

```bash
conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/grpo/build_grpo_seed_dataset.py

conda run --no-capture-output -n agent-lightning \
  python examples/image_restoration_multi_agent/grpo/validate_grpo_dataset.py \
  examples/image_restoration_multi_agent/grpo/data/{fog,snow,rain,low_light}_{train,val}.jsonl
```

The generated JSONL files and rollout artifacts are ignored by Git. Each sample contains only an input image, its fixed expert assignment, and runtime output location. Target actions, precomputed trajectories, GT images, and IQA answers are not stored.

## Stage H GRPO Smoke Test

The smoke test uses the real GLM-4.1V base model and Fog SFT LoRA, two stochastic trajectories, independent latest-image transitions with trajectory-level advantages, copy restoration, deterministic action-dependent IQA, one actor update, and checkpoint saving:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash examples/image_restoration_multi_agent/grpo/run_expert_grpo.sh fog --smoke
```

Smoke runs use the YAML `swanlab.smoke_mode` setting, which defaults to `offline`; formal runs use `swanlab.mode`, which defaults to `online`. Because the current SFT policy can still omit Hermes calls, smoke mode retains each real VLM response but overrides the environment action with `scunet -> stop`. This smoke-only override guarantees two turns so latest-image feedback, trajectory reward broadcasting, response masks, and actor updates are exercised. Formal training never enables the override.

The verified run retained four VLM turns as four independent visual transitions from two rollouts. Each turn contained only its latest image plus complete text state; GRPO deduplicated the repeated final reward by rollout and shared the resulting trajectory advantage across its turns with `1/T` weighting. A lightweight prompt-image trace annotation preserves the current image path even when the provider omits large multimodal prompt payloads from telemetry. The run reported `n_triplets=4`, `n_trajectory_transition_rollouts=2`, `avg_images_per_transition=1.0`, `max_images_per_transition=1`, no dropped or padded transitions, finite entropy/KL/gradient norm, and a consolidated LoRA adapter under `grpo/outputs/fog/smoke/`. Peak actor allocated memory fell from about 76.6 GB in the cumulative-image trajectory smoke to about 29.8 GB. The deterministic override gives both trajectories the same reward, so its group advantage is intentionally zero. This proves the training path, not restoration policy quality; before relying on a long formal run, monitor Hermes validity and group reward standard deviation during the first batches.

## Stage H Formal GRPO

The four-GPU topology keeps one complete 16-model restoration set and two IQA workers on each visible GPU.
vLLM uses all four GPUs with TP=4; `train_batch_size=32` and `rollout_n=4` produce 128 concurrent trajectories.
The launcher serially trains Fog, Snow, Rain, and Low-light, stopping immediately if any expert fails:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash examples/image_restoration_multi_agent/grpo/run_expert_grpo_4gpu.sh
```

The two-GPU topology has the same per-GPU tool allocation, while vLLM uses both GPUs with TP=2.
Its independent configurations use `train_batch_size=16`, 64 concurrent trajectories, separate checkpoints,
and `-2gpu` SwanLab experiment names:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/image_restoration_multi_agent/grpo/run_expert_grpo_2gpu.sh
```

Before formal training starts, either launcher preloads two MANIQA/NIQE/CLIP-IQA/TOPIQ-NR workers on each GPU.
Each visible GPU also receives one complete restoration set: one toolkit process containing the 12 copied `verl`
models plus one process for each of the four candidate models. Requests are pooled across visible GPUs while each model
process remains internally serialized. The service log is written beside the GRPO log as
`grpo/log/<expert>_<topology>_tool_runtime_<timestamp>.log`. Startup fails before GRPO begins if any model cannot load or
the requested resident set exceeds available memory. `IMAGE_RESTORATION_TOOL_STARTUP_TIMEOUT` changes the
default 1,800-second preload timeout, and `IMAGE_RESTORATION_TOOL_PORT` changes the default port `8767`.

Formal rollout concurrency equals `train_batch_size * rollout_n`. All trajectories remain in flight: vLLM
continuously batches their first-turn requests, and later turns re-enter the same dynamic
batch as soon as their restoration and IQA feedback is ready. This preserves multi-turn trajectory semantics without
introducing a global turn barrier that would make every trajectory wait for the slowest tool call.
The launcher renders training-step and per-batch rollout completion as two stable terminal progress bars. The
saved GRPO log contains compact progress snapshots instead of repeated `Completed x/y tasks` lines, and
expected malformed Hermes outputs are collapsed from a traceback into one `invalid_tool_call` warning line.

Formal runs use SwanLab project `image-restoration-multi-agent` with experiment names `fog-expert-grpo`, `snow-expert-grpo`, `rain-expert-grpo`, and `low-light-expert-grpo`. Each expert YAML exposes `swanlab.enabled`, `project_name`, `experiment_name`, `mode`, `smoke_mode`, and `log_dir`. Supported modes are `online`, `local`, `offline`, and `disabled`; setting `enabled: false` also removes SwanLab from the VERL logger list. Authenticate with `swanlab login` or `SWANLAB_API_KEY` before starting an online run. Credentials are intentionally not stored in YAML.

Each expert YAML is the experiment-facing source of truth rather than a minimal launcher file. It explicitly exposes SwanLab logging, rollout sampling, `ppo_mini_batch_size`, per-GPU micro-batches, PPO epochs, dynamic batching, token budgets, clipping, actor/ref log-prob batches, optimizer scheduling, KL behavior, LoRA settings, FSDP offload, checkpoint retention, and resume behavior. `train_grpo.py` only applies reduced smoke overrides (`train_batch_size=1`, `rollout_n=2`, `ppo_mini_batch_size=2`, shorter responses, eager execution, and one training step).

## Included Files

| File/Directory | Description |
|---|---|
| `config/default.yaml` | Workflow limits, reward coefficients, and four-expert shared-registry configuration |
| `config/real.yaml` | Stage D subprocess paths, timeouts, devices, IQA normalization ranges, and weights |
| `config/iqa_reward_v1.json` | Versioned four-metric normalization statistics and initial GRPO reward weights |
| `config/stage_e.yaml` | vLLM endpoint, generation settings, routing mode, and stage D runtime configuration |
| `config/stage_f.yaml` | Diagnosis/expert vLLM endpoints, expert decision mode, and real tool runtime configuration |
| `config/stage_g.yaml` | Four unique expert resources, shared tool/runtime settings, and future policy-path slots |
| `config/stage_h.yaml` | GRPO reward shaping, frozen expert SFT adapters, real restoration runtime, and four-metric IQA runtime |
| `config/tools.yaml` | Complete real restoration action registry shared by all four experts |
| `schemas.py` | Strict Pydantic task, decision, result, step, and trajectory contracts |
| `config.py` | YAML configuration loading and cross-expert consistency validation |
| `tool_registry.py` | Shared action validation and OpenAI-compatible tool schema generation |
| `exceptions.py` | Workflow-specific exception hierarchy |
| `agents/scripted.py` | Deterministic diagnosis and expert implementations |
| `agents/prompts.py` | Versioned diagnosis/expert prompts and Hermes tool schemas |
| `agents/vlm_diagnosis.py` | Single-call OpenAI-compatible VLM adapter and strict response parser |
| `agents/vlm_expert.py` | Per-turn expert VLM adapter, message-history construction, and strict Hermes parser |
| `agents/replay.py` | Replay decision source that passes predefined Hermes responses through the production parser |
| `diagnosis_metrics.py` | Stage E API, parsing, classification, confusion-matrix, and latency metrics |
| `workers/copy_worker.py` | Copy-based mock restoration worker |
| `workers/subprocess_worker.py` | Validated, timeout-aware adapter for restoration processes in `verl` |
| `evaluators/scripted.py` | Deterministic IQA evaluator |
| `evaluators/pyiqa_evaluator.py` | Real IQA normalization, aggregation, feedback, and failure handling |
| `tool_runtime/restoration_entrypoint.py` | Restoration model entrypoint executed by `verl` |
| `tool_runtime/iqa_entrypoint.py` | IQA-PyTorch entrypoint executed by `verl` |
| `tool_runtime/persistent_tool_server.py` | Stage H local service supervising persistent split-GPU model workers |
| `tool_runtime/service_client.py` | Standard-library HTTP client used by restoration and IQA adapters |
| `controller.py` | Single-diagnosis fixed-expert workflow state machine |
| `factory.py` | Deterministic and real per-task component construction |
| `lit_agent.py` | Deterministic and real Agent Lightning `LitAgent` wrappers |
| `smoke_test.py` | No-GPU traced end-to-end smoke test |
| `real_smoke_test.py` | GPU traced smoke test across `agent-lightning` and `verl` |
| `serve_glm4v.sh` | vLLM launch command for the local GLM-4.1V checkpoint |
| `stage_e_smoke_test.py` | Traced real-VLM smoke test for both routing modes |
| `stage_f_smoke_test.py` | Traced replay/real-VLM expert smoke test with real restoration and IQA components |
| `stage_g_smoke_test.py` | Four-category Replay or strict-VLM validation matrix with isolated outputs |
| `calibrate_iqa_reward.py` | Resumable all-tool training-image calibration for the four-metric GRPO reward |
| `grpo/build_grpo_seed_dataset.py` | Deterministic four-class no-GT train/validation manifest builder |
| `grpo/validate_grpo_dataset.py` | Strict GRPO manifest validator |
| `grpo/agent.py` | Fixed-expert Agent Lightning rollout returning one trajectory reward |
| `grpo/smoke_runtime.py` | Real-policy smoke environment with copy restoration and deterministic IQA |
| `grpo/train_grpo.py` | Agent Lightning VERL latest-image/trajectory-advantage GRPO entrypoint with SwanLab logging |
| `grpo/run_expert_grpo.sh` | Shared internal smoke/formal launcher used by both GPU topologies |
| `grpo/run_expert_grpo_4gpu.sh` | Serial four-expert launcher with TP=4 and fail-fast behavior |
| `grpo/run_expert_grpo_2gpu.sh` | Serial four-expert launcher with TP=2 and fail-fast behavior |
| `grpo/render_training_log.py` | Compact terminal progress bars and single-line Hermes error rendering |
| `grpo/configs/` | Independent Fog, Snow, Rain, and Low-light GRPO configurations |
| `grpo/configs_2gpu/` | Independent two-GPU configurations and output locations for all four experts |
| `grpo/templates/glm4v_no_thinking.jinja` | GLM-4.1V chat template aligned with the expert SFT prefix |
| `tests/` | Protocol, controller, subprocess, IQA, failure-path, VLM parsing, and trace tests |

## Output Contract

Each stage E run writes `stage_e_result.json`, including the raw diagnosis response, full API response payload, response/model IDs, finish reason, latency, token IDs, parse status, routing source, and optional downstream `WorkflowResult`.

Each stage F run writes `stage_f_result.json`. Every expert turn is retained inside `trajectory.json` with its decision source, parse status, raw response, reasoning content, full response payload, token IDs, selected action, and error. Successful replay runs also create intermediate images. Image content is referenced by path and is never embedded in a trajectory.

Each stage G category writes `stage_g_result.json` and `trajectory.json` under its own `fog/`, `snow/`, `rain/`, or `low_light/` directory. The matrix root also contains `stage_g_replay_matrix.json` or `stage_g_vlm_strict_matrix.json`, including the four expert resource names and per-route results.
