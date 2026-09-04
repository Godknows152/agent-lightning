# ALFWorld KL/Entropy 调参记录

本文件记录 Qwen3.5-2B ALFWorld GRPO 的 KL 与 Entropy 参数更新。每次调整只改变
KL/Entropy 相关字段，并在启动前完成 Hydra/训练 preflight。

## 2026-09-05：第 5 轮 → 第 6 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.02`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.0005`
- step 1--5 的 `actor/entropy` 从 `0.575` 上升至 `9.887`，后期出现明显熵爆炸趋势。
- step 3--5 连续 `tool_call_counts/mean=0`、`actor/tool_choice_entropy=0`，奖励固定约 `-0.05`，轨迹探索归零。
- `actor/kl_loss` 达到 `0.677`、`0.359`、`0.568`，但 `0.0005` 系数仍未抑制策略漂移。
- 训练在 step 5 后停止并删除本轮 output、rollouts、日志和 local SwanLab；云端 SwanLab run 尚需授权删除。

### 本轮新参数

- `actor.entropy_coeff = 0.01`：将熵激励减半，避免 token-level 熵继续上冲。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.001`：提高显式 KL 约束，限制策略远离 reference；该值与早期配置一致，但本轮熵系数更低。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

## 2026-09-05：第 4 轮 → 第 5 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.05`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.0001`
- `actor/entropy` 在 step 4、5 升至 `2.51`、`4.94`，出现熵爆炸趋势。
- `actor/tool_choice_entropy` 与 `tool_call_counts/mean` 在 step 4、5 均为 `0`，奖励固定约 `-0.05`，优势为 `0`，工具探索坍缩。
- `actor/kl_loss` 在 step 4 达 `2.22`、step 5 为 `0.99`；低 KL 系数未能约束策略漂移。
- 训练在 step 5 后停止并删除本轮产出；云端 SwanLab run `0go5u59j` 仍需授权 UI/API 删除（当前 CLI 无删除命令）。

### 本轮新参数

- `actor.entropy_coeff = 0.02`：从 `0.05` 下调，保留探索激励并抑制 token-level 熵爆炸。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.0005`：从 `0.0001` 提高到中等强度，约束 KL 漂移但低于曾导致熵坍缩的 `0.001`。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

## 2026-09-05：第 3 轮 → 第 4 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.012`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.0002`
- `actor.kl_loss_type = low_var_kl`
- `algorithm.use_kl_in_reward = false`
- 训练在 `global_step=2` 后停止；本轮产出已删除。
- `actor/entropy` 从 step 1 的约 `0.60` 降至 step 2 的约 `0.025`，仍触发早期熵坍缩判据。
- `actor/tool_choice_entropy` 从约 `2.51` 降至约 `0.24`，仍有工具调用，但探索强度快速下降。
- `critic/rewards/mean` 从约 `-0.22` 改善到 `-0.014`，但只有两个点，不能证明持续提升。
- `actor/kl_loss` 约为 `0.0004`、`0.0285`，降低 KL 后仍未阻止 token-level 熵快速下落。
- 结论：主要限制已转为熵正则过弱；停止并清理该轮，继续提高 Entropy、轻微降低 KL。
- 本轮 SwanLab 云端记录：`nwrnit3c`（当前 CLI 无删除命令，需授权 UI/API 删除）。

### 本轮新参数

- `actor.entropy_coeff = 0.05`：提高到共享配置的常用强度，优先阻止训练初期熵坍缩。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.0001`：保留轻量 KL 锚定，避免重新出现较大 KL 项压制探索。
- `actor.kl_loss_type = low_var_kl`：保持低方差估计。
- `algorithm.use_kl_in_reward = false`：保持关闭，避免双重 KL 惩罚。

## 2026-09-05：第 2 轮 → 第 3 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.007`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.001`
- `actor.kl_loss_type = low_var_kl`
- `algorithm.use_kl_in_reward = false`
- 训练在 `global_step=8` 后停止；本轮产出已删除。
- `actor/entropy`：step 1 为约 `0.69`，step 2–8 多数降至 `0.01–0.25`，显示熵坍缩风险。
- `actor/tool_choice_entropy`：step 1 约 `2.21`、step 2 约 `0.07`，step 3–8 为 `0`；轨迹探索性消失。
- `actor/kl_loss` 峰值约 `9.24`（step 5），在 `0.001` 系数下 KL 项压过熵奖励。
- `critic/rewards/mean` 主要在 `-0.05` 附近，step 3 约 `-0.75`，未见持续上升。
- 结论：当前组同时出现低熵、零工具选择熵、奖励停滞和 KL 峰值，继续训练不足以达到目标；停止并清理该轮。
- 本轮 SwanLab 云端记录：`djo7yp6c`（当前 CLI 无删除命令，需授权 UI/API 删除）。

### 本轮新参数

- `actor.entropy_coeff = 0.012`：温和提高 token-level 熵正则，恢复探索但避免一次性大幅提高。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy 约束漂移。
- `actor.kl_loss_coef = 0.0002`：将上一轮 KL 权重降至五分之一，避免 KL 峰值把策略推入低熵状态。
- `actor.kl_loss_type = low_var_kl`：保持低方差估计。
- `algorithm.use_kl_in_reward = false`：保持关闭，避免与 actor KL 重复惩罚。

### 监控判据

每次低频检查最近 3–5 个已完成 step，仅读取日志末尾的标量：

- 熵健康区间：`actor/entropy` 大部分保持在约 `0.2–5`，连续两步 `<0.1` 判为坍缩，连续两步 `>12` 判为爆炸。
- 探索性：`actor/tool_choice_entropy > 0` 且 `tool_call_counts/mean > 0`；连续三步二者同时为零则判为探索坍缩。
- 奖励：比较最近窗口 `critic/rewards/mean` 与前一窗口，要求没有持续恶化并出现缓慢改善迹象。
- KL：记录 `actor/kl_loss` 与 `actor/kl_coef`；若 KL 连续升高且熵同步下降，优先降低 `kl_loss_coef`；若熵爆炸且 KL 很低，优先降低 `entropy_coeff` 或适度提高 KL 系数。
