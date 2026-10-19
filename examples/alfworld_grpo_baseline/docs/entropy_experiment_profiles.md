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
checkpoint 150. Branch entropy coefficient is 0.0005; ordinary entropy and KL
coefficients are both 0.001. Both thinking/action consistency bonuses are zero.
Training uses three epochs, eight rollouts, and the existing trajectory GRPO
reward. Output directory and SwanLab experiment name are `新动作熵1019`.
For the later list-filtering resume, see `action_listing_filter.md`.
