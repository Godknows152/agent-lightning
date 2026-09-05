# ALFWorld Qwen3.5-2B KL/Entropy 调参记录（Prompt schema v4 新基线）

实验对象：ALFWorld GRPO v1，Qwen3.5-2B，2 GPU。

## 2026-09-05/06：Prompt schema v4 基线检查

- 运行：`attempt_006_schema_fix_kl0.00320_e0.02000`
- 完成到：`step=26`
- `actor/entropy`：前 5 步均值 `0.2253`，后 5 步均值 `0.3092`，范围 `0.2149–0.3245`
- `actor/kl_loss`：前 5 步均值 `0.0247`，后 5 步均值 `0.1387`，末步 `0.1546`
- `critic/rewards/mean`：前 5 步均值 `-0.0244`，后 5 步均值 `0.0138`，但波动较大，未确认持续上升
- `actor/tool_choice_entropy`：后 5 步均值约 `3.48`，没有归零
- 结论：没有熵爆炸，也未触发 `<0.1` 的硬熵坍缩阈值；但策略熵持续偏低，KL 漂移升高，奖励趋势不稳定，未满足“奖励持续升高”。按新 Prompt 基线重新开始调参。

## 本轮调整

| 时间 | 原因 | KL coefficient | Entropy coefficient | 说明 |
|---|---|---:|---:|---|
| 2026-09-05/06 | 新 Prompt 基线奖励不稳定、策略熵偏低 | 0.0032 → 0.0032 | 0.020 → 0.030 | 只提高 Entropy 正则，保留 KL 不变，扩大探索余量；不改变其他参数 |

## 清理与重启

- 已停止精确 systemd 任务 `alfworld-qwen35-2b-schema-v5.service`。
- 已删除旧实验目录及其中的 checkpoint、rollout、日志、本地 SwanLab：
  `outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_006_schema_fix_kl0.00320_e0.02000`
- 已删除旧调参文档后重新创建本文件；保留实验计划和 Prompt schema 文档。
- 已删除属于该实验的 Ray session：`session_2026-09-06_00-02-10_964674_2462654`。
- 已使用已安装 SwanLab SDK 对旧云端 run 执行删除并返回成功；云端 run 的最终删除状态应以随后查询为准。

## 新实验

- 输出目录：`outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_007_prompt_v4_kl0.00320_e0.03000`
- 启动脚本：`scripts/alfworld/qwen35_2b_v1.sh`
- 目标：先观察至少 5 个完整训练 step，再判断熵、KL、工具探索和奖励趋势。
