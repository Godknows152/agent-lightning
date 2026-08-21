# Unified expert GRPO configurations

All configurations use physical GPUs 0 and 1 and initialize the actor and the
frozen KL reference from the unified SFT LoRA at:

`LlamaFactory/image_restoration_experts/outputs/qwen3_5_0813/format_cold_start/unified`

Versions mirror the existing per-expert experiments:

- `v1`: current-IQA reward baseline.
- `v2`: marginal-efficiency reward and reduced token entropy coefficient.
- `v3`: v2 reward plus action-rarity exploration reward.
- `v4`: thinking decision-point entropy regularization.
- `v4.1.2`: scheduled legal-action first-token entropy regularization.
- `v4.1.3`: positive-advantage-gated legal-action first-token entropy regularization.
- `v4.1.4`: quality-and-validity-gated legal-action first-token entropy with cosine decay.

Each version writes checkpoints to `outputs/unified/<version>/2gpu`, SwanLab
files below its output directory, and process/tool logs to
`log/unified/<version>/2gpu`.
