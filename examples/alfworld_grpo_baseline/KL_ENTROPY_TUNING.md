# ALFWorld KL/Entropy 调参记录

本文件记录 Qwen3.5-2B ALFWorld GRPO 的 KL 与 Entropy 参数更新。每次调整只改变
KL/Entropy 相关字段，并在启动前完成 Hydra/训练 preflight。

### 2026-09-05 检查与记录约定

- 每轮必须 cloud/online 启动并验证云端 run；不再擅自切换 offline。
- 启动检查只验证进程、配置、登录与实验创建，不当作训练效果检查。常规指标检查间隔不少于约 20 分钟；相邻窗口至少各 5 个完成 step，不根据单步涨跌判断持续增奖。启动失败或非有限指标可以提前处置。
- token 熵与工具探索指标联合观察；`<0.1` 或 `>12` 是现有报警阈值而非充分判据，接近均匀输出且工具探索归零也不能判健康。奖励短暂改善不代表目标完成。
- 第 9、10、11 轮的部分调整基于未训练或仅两步观察，属于未充分验证的试探，不作为已证明有效的结论；后续不得将“尚无足够证据”直接视为必须再次改参。

## 2026-09-05：第 10 轮 → 第 11 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.0025`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.003`
- 可核实的标量记录覆盖 step 1–2（停止前至少已有 `rollouts/3.jsonl`，不把 step 2 当作最终步数）：`actor/entropy` 从 `0.493` 降至 `0.261`，尚未连续低于 `0.1`，但呈下降趋势；`actor/kl_loss` 稳定在约 `0.00035`；`actor/tool_choice_entropy` 为 `2.387`、`2.514`；`critic/rewards/mean` 从 `-0.25` 改善至 `-0.20`。
- 由于随后按要求停止并删除产出，无法证明完整训练周期满足目标。

### 本轮新参数

- `actor.entropy_coeff = 0.004`：小幅提高熵正则，针对前两步熵快速下落的风险，增加抗坍缩能力；仍远低于此前出现熵爆炸的高系数。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.003`：保持上一轮已观察到的稳定 KL 强度，避免同时改变两个方向而无法归因。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

### 本轮效果

- 待在线训练验证；本轮必须使用 `ALFWORLD_SWANLAB_MODE=cloud`，以便实时观察曲线。

## 2026-09-05：第 9 轮 → 第 10 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.003`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.005`
- 第 9 轮尚未启动，因而没有可用于证明目标达成的新指标；当前状态不能证明熵稳定或奖励持续升高。

### 本轮新参数

- `actor.entropy_coeff = 0.0025`：小幅降低熵激励，优先压制此前出现的高熵趋势，同时保留探索空间。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.003`：适度降低 KL 约束，避免过强 KL 压制策略更新和奖励增长。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

### 本轮效果

- 已重启第 10 轮训练；待积累足够 step 后低频检查。重点观察最近 5 个已完成 step 的 `actor/entropy`、`actor/kl_loss`、`actor/tool_choice_entropy` 和 `critic/rewards/mean`。

## 2026-09-05：第 8 轮 → 第 9 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.001`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.01`
- 第 8 轮完成至 step 150；清理前日志显示最终验证 reward 约 `-0.05214`。更正：`actor/entropy = 11.16–11.35` 属于第 7 轮 step 62–64，不能作为第 8 轮指标。第 8 轮完整熵/训练奖励曲线已随产出删除，不能据此证明熵稳定或奖励持续增长。
- 本轮训练产出已清理；截至本次修改尚无新训练效果可记录。

### 本轮新参数

- `actor.entropy_coeff = 0.003`：在 `0.001` 与此前较高熵系数之间取中间值，保留必要探索并降低熵继续上冲风险。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.005`：从 `0.01` 下调至中等强度，避免 KL 约束过强导致策略过度贴近 reference、奖励停滞或熵坍缩。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

### 本轮效果

- 待训练验证；重点观察最近 5 个已完成 step：`actor/entropy` 是否保持在约 `0.2–5`、`actor/tool_choice_entropy` 是否持续大于 `0`，以及 `critic/rewards/mean` 是否较前窗口缓慢上升。

## 2026-09-05：第 7 轮 → 第 8 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.005`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.002`
- 训练运行至 step 64 后退出；step 62--64 的 `actor/entropy` 为 `11.16`、`11.25`、`11.35`，接近词表熵上限。
- 后期 `actor/tool_choice_entropy` 与 `actor/action_path_entropy` 持续为 `0`，平均工具调用约 `1`，奖励约 `-0.00` 至 `-0.05`，探索目标未达成。
- `actor/kl_loss` 约 `1.66`--`1.84`，仍存在显著策略漂移；随后触发 `get_rope_index` 的 `NoneType` 运行时错误并退出。
- 已删除本轮 output、rollouts、checkpoint、日志和 local SwanLab；云端 SwanLab 记录仍需授权删除。

### 本轮新参数

- `actor.entropy_coeff = 0.001`：显著降低熵奖励，避免策略趋向近似均匀分布。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.01`：提高 KL 约束一个数量级，使 KL 项足以抵消高熵漂移。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

## 2026-09-05：第 6 轮 → 第 7 轮

### 上一轮参数与训练情况

- `actor.entropy_coeff = 0.01`
- `actor.use_kl_loss = true`
- `actor.kl_loss_coef = 0.001`
- step 1--5 中，`actor/entropy` 在 `0.0278`、`0.0629`、`0.3063`、`0.0964` 间剧烈振荡，step 5 接近坍缩。
- step 3--5 连续 `tool_call_counts/mean=0`、`actor/tool_choice_entropy=0`，奖励固定约 `-0.05`；step 4--5 出现 32/31 次 malformed tool call。
- `actor/kl_loss` 从 `0.246`、`1.408` 升至 `6.516`，说明策略漂移过大且训练不稳定。
- 训练在 step 5 后停止并删除本轮 output、rollouts、日志和 local SwanLab；云端 SwanLab run 需授权删除。

### 本轮新参数

- `actor.entropy_coeff = 0.005`：进一步降低熵激励，减少熵振荡和异常格式输出。
- `actor.use_kl_loss = true`：继续使用冻结 reference policy。
- `actor.kl_loss_coef = 0.002`：提高 KL 约束，抑制 step 间策略漂移和 KL 爆升。
- `actor.kl_loss_type = low_var_kl`，`algorithm.use_kl_in_reward = false`：保持原设置，避免重复 KL 惩罚。

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


## 2026-09-05：第 12 轮启动前调整

### 判定

当前无存活目标训练进程，最新 rollout 只覆盖启动早期，未形成可验证的连续奖励上升或熵稳定窗口；目标未满足。

### 参数更新

- `actor.entropy_coeff`: `0.004 → 0.006`，小幅增强探索，降低早期熵坍缩风险。
- `actor.use_kl_loss`: `true`（保持）。
- `actor.kl_loss_coef`: `0.003 → 0.005`，增强 reference 锚定，抑制策略漂移和熵爆炸。
- `actor.kl_loss_type`: `low_var_kl`（保持）。
- `algorithm.use_kl_in_reward`: `false`（保持，避免重复 KL 惩罚）。

### 效果

调整前没有足够指标窗口，无法评价效果；启动后按最近至少 5 个完成 step 检查。
