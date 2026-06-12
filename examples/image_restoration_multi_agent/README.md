# Hierarchical Multi-Agent Image Restoration

This example implements stages A-C of the design in `分层多智能体图像修复系统设计草案.md`. It provides validated protocols, a deterministic end-to-end restoration workflow, and Agent Lightning tracing without requiring a GPU, an LLM, a real restoration model, or a real IQA model.

## Current Scope

- One deterministic degradation diagnosis per trajectory.
- Four parallel expert identities with one shared restoration tool registry.
- Scripted expert actions using the canonical `restore_image(action)` protocol.
- Copy-based mock restoration workers and deterministic IQA scores.
- Best-image rollback, stop handling, failure limits, and trajectory JSON export.
- Agent Lightning operation/object spans and exactly one final rollout reward.

Real GLM-V inference, restoration models, IQA models, SFT, and RL are intentionally deferred to later stages.

## Requirements

Use the repository environment. In the local setup used for this example:

```bash
conda activate agent-lightning
```

The example only uses dependencies already required by Agent Lightning.

## Smoke Test

From the repository root:

```bash
conda run -n agent-lightning \
  python examples/image_restoration_multi_agent/smoke_test.py
```

The command prints the final reward, selected expert, best image, trajectory path, and recorded span names.

## Unit Tests

```bash
conda run -n agent-lightning \
  python -m pytest -q examples/image_restoration_multi_agent/tests
```

## Included Files

| File/Directory | Description |
|---|---|
| `config/default.yaml` | Workflow limits, reward coefficients, and four-expert shared-registry configuration |
| `config/tools.yaml` | Placeholder restoration actions used by all four experts |
| `schemas.py` | Strict Pydantic task, decision, result, step, and trajectory contracts |
| `config.py` | YAML configuration loading and cross-expert consistency validation |
| `tool_registry.py` | Shared action validation and OpenAI-compatible tool schema generation |
| `exceptions.py` | Workflow-specific exception hierarchy |
| `agents/scripted.py` | Deterministic diagnosis and expert implementations |
| `workers/copy_worker.py` | Copy-based mock restoration worker |
| `evaluators/scripted.py` | Deterministic IQA evaluator |
| `controller.py` | Single-diagnosis fixed-expert workflow state machine |
| `factory.py` | Per-task deterministic component construction |
| `lit_agent.py` | Agent Lightning `LitAgent` wrapper |
| `smoke_test.py` | No-GPU traced end-to-end smoke test |
| `tests/` | Protocol, controller, failure-path, and trace tests |

## Output Contract

Each run creates an output directory containing copied intermediate images and `trajectory.json`. The JSON document is the serialized `RestorationTrajectoryState`; image content is referenced by path and is never embedded in the trajectory.
