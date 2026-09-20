# ALFWorld Model Profiles

Each profile owns its model checkpoint, tokenizer tool protocol, prompt source,
materialized parquet, Hydra entry configuration, launcher, outputs, logs and SwanLab
experiment name. Shared ALFWorld environment code and rewards remain outside profiles;
profile-specific runtime/performance overrides are kept in the profile's Hydra file.

| Profile | Model | Parser | Prompt source | Data | Launcher |
| --- | --- | --- | --- | --- | --- |
| `qwen25_1_5b` | `Qwen2.5-1.5B-Instruct` | Qwen3 XML | `prompts_gigpo.py` (`alfworld_gigpo_qwen3_xml_v1`) | `data/qwen25_1_5b/` | `scripts/alfworld/qwen25_1_5b_v1.sh` |
| `qwen35_2b` | `Qwen3.5-2B` | Qwen3 XML | `prompts_gigpo.py` (`alfworld_gigpo_qwen3_xml_v1`) | `data/qwen35_2b/` | `scripts/alfworld/qwen35_2b_v1.sh` |
| `qwen35_9b` | `Qwen3.5-9B` | Qwen3 XML | `prompts_gigpo.py` (`alfworld_gigpo_qwen3_xml_v1`) | `data/qwen35_2b/` | `scripts/alfworld/qwen35_9b_v1.sh` |

Run only one launcher per experiment. Do not override `variables.BASE_MODEL`,
`variables.DATA_DIR`, or `actor_rollout_ref.rollout.multi_turn.format` from the
command line, because those three fields are a single protocol contract.

```bash
# Qwen2.5-1.5B preflight
ALFWORLD_SKIP_SWANLAB_VERIFY=1 ALFWORLD_SWANLAB_MODE=offline \
  bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen25_1_5b_v1.sh --preflight

# Qwen3.5-9B preflight
ALFWORLD_SKIP_SWANLAB_VERIFY=1 ALFWORLD_SWANLAB_MODE=offline \
  bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_9b_v1.sh --preflight

# Qwen3.5-2B preflight (current default)
ALFWORLD_SKIP_SWANLAB_VERIFY=1 ALFWORLD_SWANLAB_MODE=offline \
  bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh --preflight
```

The respective full-run outputs are `outputs/alfworld/<profile>/v1/2gpu/` and
`log/alfworld/<profile>/v1/2gpu/`; their SwanLab experiments use
`alfworld_<profile>_v1_seedN`.

All three profiles use the same GiGPO prompt contract. The Qwen3.5-2B profile
also enables its isolated performance configuration:
eight AgentLoop workers, a bounded ALFWorld TextWorld environment pool, no Actor
parameter or optimizer offload, enabled gradient checkpointing, and PPO
mini/micro batches (`4`/`4`). Ref offload retains its composed setting. Qwen3.5-9B and Qwen2.5-1.5B continue to use the
shared baseline settings and `alfworld_tool_config.yaml`.

SGLang JIT kernels require C++20. The ALFWorld launcher uses `/usr/bin/gcc-10`
and `/usr/bin/g++-10` through `CC`, `CXX`, `CUDAHOSTCXX` and `NVCC_CCBIN`, and
propagates the same values to Ray workers. This does not replace the system
default compiler and does not modify image-restoration launchers.

Native V1 migration details, per-step storage, KL-reference semantics and CPU-only
validation scope are documented in [NATIVE_VERL_MIGRATION.md](docs/NATIVE_VERL_MIGRATION.md).
