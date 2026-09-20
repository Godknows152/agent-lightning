# fog v4.1.4：四卡 step 140 转双卡续训

2026-09-20 已完成迁移。原四卡任务因 OOM 退出；检查时无该任务训练进程。原始 checkpoint 和日志保留。

在仓库根目录启动：

```bash
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v4_1_4.sh
```

默认使用物理 GPU 0、1，后台运行。从绝对 step 141 继续至 248，剩余 108 步。
配置文件为 `config/fog/v4.1.4/fog_config_2gpu.yaml`。

SwanLab 接续原实验 `Godknows/FogRL/g7vpp7ps`（`fog_v4.1.4_0.4熵正则`），
以 `id=g7vpp7ps, resume="must"` 初始化。下一次记录从 step 141 开始，原 1–140 步保留。
本地双卡输出仍使用独立目录；云端实验身份由 `trainer.swanlab_resume_run_id` 固定，不依赖新目录中的标记文件。
2026-09-20 误建的 `h6brsgy6` 实验及该次双卡本地产出已清理，尚未重新启动训练。

## 来源与产物

以下路径均相对于 `examples/image_restoration_multi_agent/old_verl_grpo/`：

| 用途 | 路径 |
| --- | --- |
| 原始四卡 checkpoint | `outputs/fog/v4.1.4/2gpu/0.4熵正则/global_step_140` |
| 原运行保存的有效配置 | `outputs/fog/v4.1.4/2gpu/0.4熵正则/swanlab/run-20260920_012421-g7vpp7ps/files/config.yaml` |
| 导出的标准 PEFT LoRA | `outputs/fog/LoRA/v4.1.4/entropy_0.4_4gpu_step140/fog` |
| 双卡优化器、调度器、RNG、数据进度 | `outputs/fog/LoRA/v4.1.4/entropy_0.4_4gpu_step140/resume_2gpu/global_step_140` |
| 新训练输出 | `outputs/fog/v4.1.4/2gpu/0.4熵正则_from4gpu_step140` |
| 新训练日志 | `log/fog/v4.1.4/2gpu/0.4熵正则_from4gpu_step140` |

源目录虽然名为 `2gpu`，其 `actor/fsdp_config.json` 和四组分片确认实际 world size 为 4。
LoRA 为 FP32、rank 16、alpha 32，共 496 个张量；使用原始 adapter 配置保留 dropout 和 target modules。

## 连续性

- Actor 从导出的 step 140 LoRA 初始化。Adam 一阶、二阶矩由四份等长参数分片合并后重分为两份；优化器 step 仍为 280。
- 恢复 scheduler 的 `last_epoch=140`，开始续训时实际学习率为 `1.4708205381669757e-5`。
  原始余弦调度保留：初始学习率 `3e-5`、warmup 比例 `0.05`、最低比例 `0.1`、总步数 248。
- 普通熵系数 `0.0012`；首 token 动作熵系数 `0.008`，保持 `delayed_constant`、起点比例 `0.4` 及原质量门控。
- Actor KL 系数及 algorithm KL 系数均为 `0.06`。
- KL reference 继续加载 `LlamaFactory/image_restoration_experts/outputs/qwen3_5_0731/format_cold_start/fog`，避免参考策略随 Actor 的 RL 权重改变。
- 保留 batch size 16、每条提示 3 条 rollout、PPO mini batch 8，以及原采样、奖励、验证参数。
- 复制 `data.pt`，保留第 3 个 epoch 内已消费 16 个 batch 的位置；启动器复用现有 parquet，避免改变续训样本顺序。
- 拷贝原 rank 0、1 的 RNG 状态。world size 改变后随机数消费和浮点归约顺序会改变，不保证与继续四卡训练逐位一致。

与原四卡有效配置的实质差异限于恢复来源、日志/输出名称及双卡资源设置：训练与工具设备数 4→2，每卡训练 micro batch 2→1，SGLang 显存比例 0.20→0.15。全局 PPO mini batch 保持不变。

迁移 checkpoint 不包含 FSDP model 分片，因此本配置的 `load_contents` 为 `[optimizer, extra]`；权重通过 LoRA 路径初始化。
之后若改为恢复新生成的原生双卡 checkpoint，需要同时将 `load_contents` 改回 `[model, optimizer, extra]` 并更新恢复路径。
当前命令固定从迁移后的 step 140 开始。

## 验证

- 启动脚本 `--preflight`、Shell 语法及 `git diff --check` 通过。
- 实际启动双卡 FSDP Actor，仅初始化和恢复后退出：逐一验证两个 rank 上全部 496 个 LoRA 参数的值、优化器参数顺序、Adam 状态及恢复后的学习率与调度步数，均通过。
- 转换和独立 KL reference 回归测试，加动作熵调度测试，共 10 项通过。
- 未执行正式 rollout 或训练更新；正式训练由上述命令启动。

转换脚本：`scripts/export_LoRA/export_fsdp_checkpoint_to_lora.py`（导出权重）及
`scripts/export_LoRA/prepare_lora_resume_state.py`（重分 Adam 状态并复制调度与数据进度）。
