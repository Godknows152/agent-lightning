# ALFWorld Qwen3.5-2B 自适应 KL/Entropy 记录

监控对象：ALFWorld GRPO v1，Qwen3.5-2B，2 GPU。每 30 分钟检查一次；只允许修改 `actor.kl_loss_coef` 与 `actor.entropy_coeff`。

判据：连续窗口中 `actor/entropy < 0.1` 或工具探索归零视为熵坍缩；连续窗口 `actor/entropy > 12` 视为熵爆炸；奖励要求最近窗口中位数高于前一窗口。检测到目标未满足时，删除失败尝试的本地产出和 SwanLab 记录，更新参数，提交 git 后重新启动。

| 时间 | 原因 | 调整前（KL / Entropy） | 调整后（KL / Entropy） | 上一轮效果 |
|---|---|---|---|---|
| 2026-09-05 17:04:49 +0800 | bootstrap | 0.005 / 0.006 | 0.004 / 0.0069 | step=None, reward=None, entropy=None, tool_entropy=None |
| 2026-09-05 17:12:31 +0800 | bootstrap | 0.004 / 0.0069 | 0.0032 / 0.007935 | step=None, reward=None, entropy=None, tool_entropy=None |
| 2026-09-05 19:38:00 +0800 | entropy_collapse | 0.0032 / 0.007935 | 0.0032 / 0.0119025 | stopped at step 36；actor/entropy≈0.0333，actor/kl_loss≈0.0122，reward≈-0.0141，tool_entropy≈1.3762；Ray 临时目录超过 95% 使用率 |
| 2026-09-05 20:13:41 +0800 | policy_entropy_adjustment | 0.0032 / 0.0119025 | 0.0032 / 0.02 | stopped at step 9；actor/entropy 在 0.663→0.147 后回升至 0.418，未形成稳定窗口；critic/rewards/mean 在 -0.272~-0.070 间波动，未确认持续上升；tool_choice_entropy 仍非零但下降至 1.57；已删除本轮本地产出并清空 /tmp/ray |

## 2026-09-05：第 13 轮启动前调整

### 判定

第 12 轮（`attempt_003_kl0.00320_e0.01190`）运行到 `step=9` 后停止。`actor/entropy` 从 `0.6628` 下降到 `0.1467`，随后回升到 `0.4181`，尚未达到 `<0.1` 的硬坍缩阈值，但呈现明显的低熵下探与波动；`critic/rewards/mean` 未形成持续上升窗口，且日志出现未知工具名/格式错误。因此不继续沿用该轮。

### 参数更新

- `actor.entropy_coeff`: `0.0119025 → 0.02`，增强策略熵正则，扩大探索余量，优先避免早期低熵下探。
- `actor.use_kl_loss`: `true`（保持）。
- `actor.kl_loss_coef`: `0.0032`（保持；本次只调整策略熵）。
- `actor.kl_loss_type`: `low_var_kl`（保持）。
- `algorithm.use_kl_in_reward`: `false`（保持）。

### 清理

- 已停止旧实验进程组。
- 已删除旧实验 `attempt_003_kl0.00320_e0.01190` 的本地产出、rollouts、日志和本地 SwanLab 文件。
- 已清空 `/tmp/ray`。
- 云端 SwanLab run `aujutpqx`：当前 CLI 未提供删除命令，未伪造删除结果；如需云端删除，请在 SwanLab UI/API 中执行。

### 新实验

- 新实验目录：`attempt_004_kl0.00320_e0.02000`。
- 启动后至少观察 5 个完成 step，再判断熵稳定性、工具探索和奖励趋势。
