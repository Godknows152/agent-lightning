# Action 稀有度奖励实施说明

## 1. 实验边界

- 基线：v2 的 `marginal_efficiency_v1` 奖励、采样参数、KL 和 SFT adapter。
- v1/v2：`algorithm.action_rarity_reward_coeff: 0.0`，行为不变。
- 新 v3：只把 `algorithm.action_rarity_reward_coeff` 覆盖为 `0.02`。
- 旧 v3 的 action-token/trie 熵正则、额外 actor forward 和相关配置不参与新实验。

## 2. 奖励定义

在一个全局 rollout batch 中，只统计成功执行的 16 种非停止修复 action：

```text
c(a) = action a 在当前 batch 中的执行次数
N = sum_a c(a)
p_hat(a) = c(a) / N
```

轨迹 `i` 有 `m_i` 次非停止修复调用时，其稀有度分数为：

```text
s_i = clip(mean_t(-log(p_hat(a_i,t))) / log(16), 0, 1)
```

轨迹内取平均而不是求和，避免通过增加工具调用次数获取更高奖励。`stop` 不参与计数，也不获得稀有度奖励。

## 3. 质量门控和 GRPO 接入

训练 driver 先只用原始任务奖励和现有 KL 计算一次基础 advantage：

```text
g_i = 1[sum_t(base_advantage_i,t * response_mask_i,t) > 0]
r_i_rarity = 0.02 * g_i * s_i
```

该标量被加到轨迹最后一个模型生成 token 的 `token_level_rewards`，然后重新计算最终 advantage。这样经验频率不需要可微，策略梯度通过最终 reward/advantage 更新 actor；基础 advantage 非正的稀有但低质量轨迹不会得到奖励。

## 4. 训练指标

| 指标 | 含义 |
|---|---|
| `actor/tool_choice_entropy` | 当前 batch 非停止修复 action 的经验熵 |
| `actor/tool_choice_sample_count` | 参与统计的修复调用数 |
| `actor/tool_choice_unique_action_count` | 当前 batch 出现的修复 action 数 |
| `actor/action_rarity_score_mean` | 有效轨迹的平均归一化稀有度 |
| `actor/action_rarity_reward_mean` | 所有轨迹实际获得的平均稀有度奖励 |
| `actor/action_rarity_reward_max` | 当前 batch 最大稀有度奖励 |
| `actor/action_rarity_reward_gate_ratio` | 有效轨迹中基础 advantage 为正的比例 |
| `actor/action_rarity_valid_trajectory_rate` | 至少执行一次非停止修复的轨迹比例 |
| `num_turns/{min,max,mean}` | 为兼容既有 SwanLab 图表而保留的名称；记录每条轨迹实际进入执行队列的工具调用次数 |
| `tool_call_counts/{min,max,mean}` | 与 `num_turns/*` 相同的工具调用次数，使用语义明确的新名称 |

## 5. 启动方式

```bash
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh --preflight
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh
```

可以通过环境变量覆盖最大奖励：

```bash
OLD_VERL_ACTION_RARITY_REWARD_COEFF=0.01 \
  bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh
```

## 6. 日志与训练输出

所有专家的 v1/v2/v3 入口按专家和版本隔离运行产物：

```text
log/<expert>/vN/
  <expert>_vN_<timestamp>.log
  restoration_tool_info.log
  restoration_tools.log

outputs/<expert>/vN/
  global_step_*/
  penalized_samples/
  swanlab/
```

`VERL_LOG_DIR` 指向对应的 `log/<expert>/vN`，因此主训练日志和两类工具调用日志不会跨版本混写。checkpoint、惩罚样本导出和 SwanLab 本地记录统一位于对应的 `outputs/<expert>/vN`。

## 7. 已知边界

- 经验奖励只能强化 rollout 中实际采样到的稀有 action；完全坍缩且没有替代 action 样本时，它本身没有恢复信号。
- 当前第一版按全局 batch 统计，不区分图像状态。正 advantage 门控用于抑制不合适的稀有 action，但它不是严格的状态条件熵。
- `rollout.n=3` 不足以稳定估计 16 类同状态熵，因此第一版不在三样本 prompt group 内单独估计频率。
