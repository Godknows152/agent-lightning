# Hierarchical Multi-Agent Image Restoration

This example implements stages A-F of the design in `分层多智能体图像修复系统设计草案.md`. It provides validated protocols, a deterministic workflow, Agent Lightning tracing, a real restoration/IQA path isolated in the `verl` conda environment, and GLM-4.1V diagnosis/expert inference served by vLLM.

## Current Scope

- One deterministic degradation diagnosis per trajectory.
- Four parallel expert identities with one shared restoration tool registry.
- Scripted expert actions using the canonical `restore_image(action)` protocol.
- Copy-based mock components for CPU regression tests.
- Real restoration workers invoked through an isolated `verl` subprocess.
- TOPIQ-NR, MUSIQ, and NIQE evaluation with normalized weighted aggregation.
- Single-call GLM-4.1V diagnosis through a vLLM OpenAI-compatible endpoint.
- Strict VLM parsing with raw-response and token-ID retention.
- Independent `predicted_strict` and `oracle_observe` routing modes.
- Diagnosis API, parsing, classification, confusion-matrix, and latency metrics.
- Strict restoration-expert Hermes parsing with raw response, reasoning, token, latency, and error retention.
- Independent `replay` and `vlm_strict` expert decision modes using the same parser and Controller path.
- Replay-driven multi-turn tool execution for pre-training feasibility validation.
- Best-image rollback, stop handling, failure limits, and trajectory JSON export.
- Agent Lightning operation/object spans and exactly one final rollout reward.

SFT, RL, and trained-model task-quality validation remain deferred to later stages. The untrained expert VLM is not expected to select or sequence tools correctly during stage F.

All four experts receive the same 16 restoration actions: the 12 copied `verl` tools plus NAFNet denoising, FocalNet dehazing/desnowing, and MB-TaylorFormer dehazing. No expert-specific action filtering is applied.

## Requirements

Use the repository environment. In the local setup used for this example:

```bash
conda activate agent-lightning
```

The controller and Agent Lightning runner use `agent-lightning`. Restoration and IQA subprocesses use `verl`; the main environment never imports their PyTorch, BasicSR, or model packages.

Stage E serves `/home/LXJ/Python_Projects/Models/GLM-4.1V-9B-Thinking` with vLLM `0.10.2` from the `agent-lightning` environment. The model is exposed through the standard OpenAI Chat Completions API. Guided JSON decoding is intentionally disabled so the smoke test observes the untrained model's actual format-following behavior.

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

## Unit Tests

```bash
conda run -n agent-lightning \
  python -m pytest -q examples/image_restoration_multi_agent/tests
```

## Included Files

| File/Directory | Description |
|---|---|
| `config/default.yaml` | Workflow limits, reward coefficients, and four-expert shared-registry configuration |
| `config/real.yaml` | Stage D subprocess paths, timeouts, devices, IQA normalization ranges, and weights |
| `config/stage_e.yaml` | vLLM endpoint, generation settings, routing mode, and stage D runtime configuration |
| `config/stage_f.yaml` | Diagnosis/expert vLLM endpoints, expert decision mode, and real tool runtime configuration |
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
| `controller.py` | Single-diagnosis fixed-expert workflow state machine |
| `factory.py` | Deterministic and real per-task component construction |
| `lit_agent.py` | Deterministic and real Agent Lightning `LitAgent` wrappers |
| `smoke_test.py` | No-GPU traced end-to-end smoke test |
| `real_smoke_test.py` | GPU traced smoke test across `agent-lightning` and `verl` |
| `serve_glm4v.sh` | vLLM launch command for the local GLM-4.1V checkpoint |
| `stage_e_smoke_test.py` | Traced real-VLM smoke test for both routing modes |
| `stage_f_smoke_test.py` | Traced replay/real-VLM expert smoke test with real restoration and IQA components |
| `tests/` | Protocol, controller, subprocess, IQA, failure-path, VLM parsing, and trace tests |

## Output Contract

Each stage E run writes `stage_e_result.json`, including the raw diagnosis response, full API response payload, response/model IDs, finish reason, latency, token IDs, parse status, routing source, and optional downstream `WorkflowResult`.

Each stage F run writes `stage_f_result.json`. Every expert turn is retained inside `trajectory.json` with its decision source, parse status, raw response, reasoning content, full response payload, token IDs, selected action, and error. Successful replay runs also create intermediate images. Image content is referenced by path and is never embedded in a trajectory.
