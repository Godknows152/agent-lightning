# 工具级受限分布熵实施方案

## 1. 文档状态

- 当前状态：阶段 B 已完成。
- 适用训练链路：`examples/image_restoration_multi_agent/old_verl_grpo/`。
- 实际训练后端：`examples/image_restoration_multi_agent/verl_backend/`。
- 目标协议：Qwen3 Coder XML 工具调用格式。
- 目标函数：`restore_image`。
- v3 工具选择熵系数：`tool_choice_entropy_coeff: 0.05`。
- v1/v2 工具选择熵系数：`tool_choice_entropy_coeff: 0.0`。

本版只增强模型在合法修复工具之间的探索性，不把自然语言、XML 标签、函数名或参数名的变化误当成工具探索。

## 2. 改造原因

此前的定向熵仍在 action 文本对应的词表 token 上计算。该量主要反映模型生成工具名称字符串时的 token 置信度，不能直接表示模型在可用工具集合上的不确定性。例如，模型可以生成许多不同的分析文本，但最终始终选择同一个工具；这时文本层面仍有多样性，工具轨迹却已经收敛。

新方案将正则对象改为当前回合实际开放的工具选择树：

1. 只读取 `restore_image.action` 的 active schema enum。
2. 在准确的 XML action span 内构建合法工具名称的 token trie。
3. 只在 trie 的真实分叉位置对合法下一 token 做受限 softmax。
4. 对一次工具调用的根分支熵和实际采样路径上的条件分支熵求和。
5. 轨迹内按有效工具调用次数平均，再对 PPO batch 聚合。

这样，增加自然语言变化不会提高工具熵奖励，单纯增加工具调用轮数也不会线性放大奖励。

## 3. 真实 Tokenizer 约束

Qwen3.5-9B tokenizer 下，17 个 action 只有 15 个唯一首 token，不能把每个工具简化成一个互不冲突的首 token 类别。

已确认的首 token 碰撞为：

| 工具组 | 共享首 token ID | 后续分叉 |
|---|---:|---|
| `turbo_rain`、`turbo_snow` | `63161` | 在后续 `_r` / `_s` 路径分开 |
| `focalnet_dehaze`、`focalnet_desnow` | `69` | 在后续 `_de...` 路径分开 |

因此实现不保存静态首 token ID 列表，而是根据每个回合的 active schema，在生成 action 的原始 XML 上下文中重新编码所有合法候选并构建 trie。这样同时解决：

- 多 token 工具名；
- 首 token 碰撞；
- 某些回合隐藏 `stop` 或其他 action；
- tokenizer 在不同前缀上下文中可能产生的切分变化。

## 4. 数学定义

设一次工具调用的合法工具集合构成 token trie。对采样工具路径上的第 `d` 个真实分叉，合法下一 token 集合为 `C_d`，模型在该位置的 logits 为 `z_d`。受限概率为：

```text
p_d(k) = exp(z_d[k]) / sum(exp(z_d[j]) for j in C_d)
```

该分叉的受限熵为：

```text
H_d = -sum(p_d(k) * log(p_d(k)) for k in C_d)
```

一次调用的工具选择熵估计为：

```text
H_call = sum(H_d for d in sampled trie branch positions)
```

根分支熵加上采样路径上的条件分支熵，是完整工具分布链式熵的 Monte Carlo 估计。对于没有 token 前缀碰撞的工具，只包含根分支；对于两组碰撞工具，还包含对应的后续条件分支。

设轨迹 `b` 有 `N_b` 次成功对齐的工具调用，则：

```text
H_trajectory(b) = sum(H_call) / N_b
```

无有效工具决策的轨迹贡献 0。Actor loss 中增加：

```text
L_actor = L_original - tool_choice_entropy_coeff * mean(H_trajectory)
```

阶段 B 使用：

```yaml
tool_choice_entropy_coeff: 0.05
```

## 5. 数据协议

Rollout 为每个 response 位置维护以下字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `tool_choice_call_start_mask` | `[B, R]` | 每次成功对齐工具调用的第一个 trie 决策位置为 1 |
| `tool_decision_position_mask` | `[B, R]` | 所有真实 trie 分叉位置为 1 |
| `tool_choice_candidate_token_ids` | `[B, R, 17]` | 每个分叉位置的合法下一 token ID，0 padding |
| `tool_choice_candidate_leaf_counts` | `[B, R, 17]` | 每个候选分支覆盖的最终工具数，0 表示无效槽位 |

候选宽度固定为 17，便于 TensorDict 拼接和微批次传输。实际候选数由 active schema 和当前 trie 节点决定。

`tool_choice_call_start_mask` 是显式调用计数协议，不能根据候选数量反推。否则 active schema 仅有两个工具时，根决策可能被错误识别为不存在。

工具 observation、非工具 agent 输出和 response padding 对应的四个字段全部补零。

## 6. 严格对齐规则

工具决策提取采用 fail-closed 策略。只有同时满足下列条件才写入 mask 和候选：

1. 当前 assistant turn 恰好解析出一个工具调用。
2. 函数名为 `restore_image`。
3. 参数只有 `action`，且值为非空字符串。
4. action 属于当前回合 active schema enum。
5. 响应中恰好有一个结构化 `<parameter=action>...</parameter>` span。
6. XML 文本中的 action 与 parser 结果一致。
7. fast tokenizer 提供的 offset mapping 能完整覆盖 action，且 token 不跨越边界。
8. 原始 `response_ids` 与 decode/re-encode roundtrip 一致。
9. 实际 action token 序列存在于动态候选 trie。

自然语言中出现同名工具不会被标记；XML 不完整、多个调用、schema 隐藏、边界跨 token 或 tokenizer roundtrip 失败时，该回合不产生工具熵奖励，并记录诊断原因。

## 7. Actor Forward

Actor 只在候选位置 gather logits，不为 loss 额外保留完整 `[B, R, V]` 张量。

rollout batch 中 prompt 使用左 padding、response 使用右 padding；进入 actor 前，`left_right_2_no_padding()` 会删除两侧 padding，FSDP 再按当前微批次的最长真实序列做右 padding。因此 response token `r` 的 causal predictor 必须按每个样本的真实 prompt 长度定位：

```text
real_prompt_length = sum(attention_mask[:prompt_width])
predictor_position = real_prompt_length - 1 + r
```

这里的 `predictor_position` 是去 padding 后、当前 FSDP 微批次局部右 padding 序列中的位置。不能继续使用原始 padded `prompt_width` 作为统一切片起点，否则真实 prompt 较短的样本会读取偏后的错误 logits。实现同时校验：候选只能出现在有效 response 位置、prompt 非空、predictor 未越过当前 logits 序列宽度、候选 token ID 未越过词表。

当前阶段 B 支持的 actor 路径为：

```yaml
use_remove_padding: false
use_fused_kernels: false
ulysses_sequence_parallel_size: 1
```

启用工具选择熵但不满足这些条件时，forward 会明确拒绝运行，防止静默计算错误。

## 8. PPO 聚合与指标

PPO 使用显式调用起点计数：

```text
sequence_entropy = sum(entropy at decision positions)
call_count = sum(tool_choice_call_start_mask)
trajectory_entropy = sequence_entropy / max(call_count, 1)
```

随后按全局 batch size 和 data-parallel size 使用现有 loss 聚合约定计算均值。

SwanLab/训练日志新增：

| 指标 | 含义 |
|---|---|
| `actor/tool_choice_restricted_entropy` | 每轨迹按调用次数归一化后的工具受限熵 |
| `actor/tool_choice_entropy_bonus` | 系数乘以工具受限熵后的 loss 奖励量 |
| `actor/tool_choice_entropy_coeff` | 当前工具选择熵系数 |

Rollout 对齐诊断使用 `tool_choice_decision/*` 指标，包括匹配率、每轨迹决策位置数以及 roundtrip、schema、边界失败率。

## 9. 版本配置

| 版本 | IQA 奖励 | 普通 response 熵 | 工具选择熵 | 用途 |
|---|---|---:|---:|---|
| v1 | 原奖励 | `0.005` | `0.0` | 原奖励基线 |
| v2 | 边际效率奖励 | `0.005` | `0.0` | 新奖励基线 |
| v3 | 与 v2 相同 | `0.005` | `0.05` | 工具级探索实验 |

v2 与 v3 的 IQA 奖励及工具调用成本保持一致，主要实验变量是工具选择熵正则。

共享配置：

```yaml
# restoration_common_config_2gpu.yaml (v1/v2)
actor_rollout_ref:
  actor:
    tool_choice_entropy_coeff: 0.0

# restoration_common_config_v3_2gpu.yaml (v3)
actor_rollout_ref:
  actor:
    tool_choice_entropy_coeff: 0.05
```

公共启动器的 preflight 必须验证：

- v1/v2 系数严格为 `0.0`；
- v3 系数严格为 `0.05`；
- v3 仍使用边际效率奖励；
- 实验名、输出目录、主日志和工具日志包含正确的版本目录；
- actor 运行参数满足阶段 B 的 forward 限制。

## 10. 阶段 B 完成项

- [x] 从 active schema 动态提取合法 action。
- [x] 在原始 XML 上下文中编码候选并构建 token trie。
- [x] 支持两组真实首 token 碰撞及后续条件分支。
- [x] 增加显式调用起点、决策位置、候选 token 和叶子数字段。
- [x] 为 observation、padding 和非工具 agent 提供稳定全零 schema。
- [x] 修正 `left_right_2_no_padding` 与微批次局部右 padding 后的逐样本 causal logits 对齐。
- [x] actor 仅 gather 合法候选 logits。
- [x] 实现轨迹内按工具调用次数归一化的 PPO 熵奖励。
- [x] 配置 v3 系数为 `0.05`，v1/v2 保持 `0.0`。
- [x] 删除旧的 action 文本 token 熵运行时分支、配置字段和指标。
- [x] 增加真实 tokenizer 碰撞、均匀/塌缩分布、零 mask、双工具 schema、归一化和 backward 测试。

## 11. 验收标准

阶段 B 完成后应满足：

1. v1/v2 运行结果不依赖任何工具选择字段，系数为零时 loss 行为保持原样。
2. v3 的工具熵只由合法工具分支 logits 决定，不受自然语言文本多样性直接影响。
3. 17 个等概率工具的理论最大熵为 `log(17)`。
4. 单一工具概率占优时受限熵接近 0。
5. 两次同尺度调用与一次调用得到相同量级的轨迹熵，不鼓励调用满轮数。
6. active schema 只有两个工具时仍正确计为一次调用。
7. 无有效决策、所有候选 padding 或 malformed XML 时 loss 有限且不产生 bonus。
8. 工具熵 backward 后候选 logits 梯度存在且有限。
9. 12 个专家版本启动脚本全部通过 `--preflight`。
10. 活动代码、配置和测试中不存在旧 token 级实现的字段、指标或测试命名。

## 12. 后续实验建议

阶段 B 只完成代码与配置，不替代实际训练对照。训练 v3 时建议同时观察：

- `actor/tool_choice_restricted_entropy` 是否在早期下降后保持非零；
- `actor/tool_choice_entropy_bonus` 相对 `actor/pg_loss` 的量级；
- 每个训练阶段的工具频率、首选工具占比和 action path entropy；
- 平均工具调用次数是否因归一化设计保持稳定，而不是被熵奖励推高；
- IQA 增益、无效调用率和最终轨迹奖励是否优于 v2。

若 `0.05` 导致无效调用明显上升或 IQA 下降，可对比 `0.01`；若工具分布仍过快塌缩且 bonus 远小于 policy loss，可对比 `0.1`。调整时保持 v2 奖励和其余采样参数不变。
