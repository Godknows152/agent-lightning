# ALFWorld GRPO 训练文件规范

本实验沿用 `examples/image_restoration_multi_agent/old_verl_grpo` 的版本化配置、后台主日志、
独立 checkpoint 输出与 SwanLab 本地记录方式，但所有文件保持在 ALFWorld 隔离目录内。

```text
examples/alfworld_grpo_baseline/
├── config/
│   ├── alfworld_common_config_2gpu.yaml
│   └── alfworld/v1/alfworld_config_2gpu.yaml
├── scripts/
│   ├── run_alfworld_grpo_2gpu.sh
│   └── alfworld/alfworld_v1.sh
├── data/{train.parquet,test.parquet}
├── log/alfworld/v1/2gpu/
│   └── seed0/alfworld_v1_seed0_<timestamp>.log
└── outputs/alfworld/v1/2gpu/
    ├── seed0/
    │   ├── global_step_*/
    │   ├── latest_checkpointed_iteration.txt
    │   ├── rollouts/
    │   ├── swanlab/
    │   └── .swanlab_experiment.json
    └── # seed1/seed2 仅在后续需要多次独立重复时创建
```

## 命令

```bash
# 不启动 Ray/GPU，只验证 seed 0 的有效配置和路径
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_v1.sh --preflight

# 1 步 smoke，默认后台运行，SwanLab offline
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_v1.sh --smoke

# 5 步 pilot，默认后台运行，SwanLab offline
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_v1.sh --pilot

# 唯一的 baseline 正式训练（seed 0），默认后台运行，SwanLab cloud
SEED=0 bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_v1.sh

# 可选：论文需要均值/方差时，再运行三个 seed 串行训练
bash examples/alfworld_grpo_baseline/run_three_seeds_serial_2gpu.sh
```

当前 baseline 默认只训练 `seed0`，正式训练默认 `resume_mode=disable`。不同 seed 绝不复用输出目录；
只有在后续需要论文统计稳健性时才启用 seed1/seed2。若后续需要恢复，应显式提供
Hydra override `trainer.resume_mode=resume_path trainer.resume_from_path=/path/to/global_step_N`，并保持
原 seed 的 `trainer.experiment_name` 与输出目录不变，以便 old-VERL 的 SwanLab marker 续接同一 run。

可用环境覆盖：`ALFWORLD_OUTPUT_DIR`、`ALFWORLD_LOG_DIR`、`ALFWORLD_SWANLAB_LOG_DIR`、
`ALFWORLD_SWANLAB_MODE`、`ALFWORLD_TOTAL_STEPS`、`CUDA_VISIBLE_DEVICES` 和 `PYTHON_BIN`。
