# 2026 年使用大模型进行 ALFWorld 实验的 Baseline 调研

> 整理日期：2026-09-05
>
> 本文整理 2026 年公开的、明确包含 ALFWorld 实验的论文与开源实现，重点比较模型、奖励、Prompt、步数及训练超参数。论文中没有披露的实现细节，尽量通过对应代码仓库补充。不同论文的实验规模、后端和数据划分并不完全一致，表中的参数不能未经验证直接横向比较。

## 1. 论文概览

| 工作 | 发表状态 | ALFWorld 模型 | 方法重点 | 代码/资料 |
|---|---|---|---|---|
| **STEP-HRL** | ACL 2026 Long Paper | Mistral-7B-Instruct-v0.2、Gemma-7B、Llama3-8B；扩展实验含 Llama3.2-1B/3B | 层次化 RL、子任务级和 step-level credit assignment | [论文](https://aclanthology.org/2026.acl-long.318/)，[代码](https://github.com/TonyStark042/STEP-HRL) |
| **SALT** | Findings of EACL 2026 | Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct | 基于轨迹图的 step-level advantage assignment | [论文](https://aclanthology.org/2026.findings-eacl.247/) |
| **STAPO** | ACL 2026 Long Paper | Qwen2.5-1.5B/7B/14B、Llama3.1-8B | normalized entropy、selective trajectory-aware optimization | [论文](https://aclanthology.org/2026.acl-long.1308/) |
| **SkillRise** | 2026 arXiv 预印本 | Qwen3-1.7B、Qwen3-4B | 跨任务 skill document 迁移 | [论文](https://arxiv.org/abs/2607.26784)，[代码](https://github.com/Within-yao/SkillRise) |
| **From History to State** | 2026 公开工作 | Qwen3-8B | 将完整历史压缩成显式 task state，再做 RL refinement | 论文与实现需以项目最新版本为准 |
| **RWML** | 2026 arXiv 预印本 | LLM policy + action-conditioned world model | world model 辅助 policy learning | [论文](https://arxiv.org/abs/2602.05842) |
| **Agent2 RL-Bench** | 2026 arXiv 预印本 | Qwen2.5-7B-Instruct、Qwen3-8B | 自动设计 RL pipeline 的工程化案例 | [论文](https://arxiv.org/abs/2604.10547)，[代码](https://github.com/microsoft/RD-Agent/) |

最适合直接参考 ALFWorld RL baseline 的是 **STEP-HRL、SALT 和 STAPO**；与当前 Qwen3.5-2B 规模最接近的是 **SALT 的 Qwen2.5-1.5B** 和 **SkillRise 的 Qwen3-1.7B**。

---

## 2. STEP-HRL

### 2.1 模型

- Mistral-7B-Instruct-v0.2
- Gemma-7B
- Llama3-8B-Instruct
- 扩展实验：Llama3.2-1B/3B-Instruct

### 2.2 奖励

外部任务奖励保持简单：

```text
任务成功：1
任务失败：0
```

同时增加与子任务完成/局部进度有关的 intrinsic reward，并将最终结果传递给：

- high-level planner；
- low-level executor；
- local-progress policy。

STEP-HRL 的主要贡献不是构造复杂的环境 reward，而是通过层次结构改善 sparse reward 的 credit assignment。

### 2.3 Prompt

STEP-HRL 将决策拆为三类 Prompt：

**High-level planner** 输入：

- 原始任务描述；
- 已完成的子任务；
- 当前 local progress；
- 当前 observation。

输出下一个 subtask。

**Low-level executor** 输入：

- 当前 subtask；
- 当前 observation；
- local progress。

输出可执行动作，并判断当前 subtask 是否完成。

**Local-progress policy** 输入：

- 上一步 local progress；
- 当前 subtask；
- 上一步 action；
- 新 observation。

输出更新后的局部进度。

这种设计避免将完整历史无限追加到上下文中。

### 2.4 训练与采样参数

| 参数 | 设置 |
|---|---:|
| BC epochs | 5 |
| Offline RL epochs | 3 |
| BC batch size | 128 |
| RL batch size | 256 |
| Actor learning rate | `1e-5` |
| Critic learning rate | `1e-4` |
| Optimizer | AdamW |
| Discount factor | `gamma=0.99` |
| Critic warmup | 100 steps |
| 最大 episode 环境步数 | 50 |
| High-level/local-progress temperature | 0.7 |
| Low-level temperature | 0 |
| High-level/low-level max new tokens | 32 |
| Local-progress max new tokens | 150 |
| 硬件 | 8 × A100 80GB |

论文指出，ALFWorld 上 BC 已能取得很高成绩，因此 RL 带来的额外提升相对有限。实际瓶颈通常是 credit assignment、历史表示和动作执行质量。

---

## 3. SALT

### 3.1 模型

- Qwen2.5-1.5B-Instruct
- Qwen2.5-7B-Instruct

### 3.2 奖励

基础奖励：

```text
成功：+1
失败：0
invalid action：-0.1
```

SALT 不改变 rollout 和环境 reward，而是在 GRPO advantage 计算之后构造 trajectory graph：

1. 合并重复或等价状态；
2. 比较相同状态下不同后续 action 的结果；
3. 为不同 step 重新分配 advantage；
4. 减少整条轨迹共享同一 advantage 带来的 credit assignment 问题。

### 3.3 Prompt

典型结构：

1. system instruction；
2. task description；
3. 当前 step count；
4. 最近 3 步 action-observation history；
5. 当前 observation；
6. admissible actions；
7. `<think>...</think>`；
8. 最终动作放在 `<action>...</action>` 中。

### 3.4 训练参数

| 参数 | 设置 |
|---|---:|
| Group size | 8 |
| 最大交互步数 | 50 |
| 最大 prompt length | 2048 |
| 最大 response length | 512 |
| History length | 3 |
| Rollout temperature | 1.0 |
| Evaluation temperature | 0.4 |
| Learning rate | `1e-6` |
| KL loss coefficient | 0.01 |
| Training steps | 300 |
| Clip ratio | 0.2 |
| Qwen2.5-1.5B mini-batch | 256 |
| Qwen2.5-7B mini-batch | 128 |
| 随机种子 | 3 个 |
| 结果汇报 | 最后 5 个 checkpoint 平均 |

**可借鉴点：** 如果当前实验只允许使用最终成功/失败 reward，可以优先改进 step-level advantage，而不是马上修改环境 reward。

---

## 4. STAPO

### 4.1 模型

- Qwen2.5-1.5B-Instruct
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Llama3.1-8B-Instruct

### 4.2 奖励

基础 reward 与 SALT 类似：

```text
成功：+1
失败：0
invalid action：-0.1
```

STAPO 的区别是先计算 normalized entropy，再识别可能被 trajectory neglect 的异常 step，仅对这些 step 施加 trajectory-aware reward 或 penalty。

论文强调，原始 Shannon entropy 受状态 action space 大小影响：可选动作越多，天然 entropy 可能越高。因此不能只用一个全局 entropy 阈值判断策略异常。

### 4.3 Prompt

- task description；
- 当前 step；
- 最近 history；
- current observation；
- admissible actions；
- `<think>` 推理段；
- `<action>` 动作段。

### 4.4 训练参数

| 参数 | 设置 |
|---|---:|
| 最大 episode steps | 50 |
| 最大 prompt length | 2048 |
| 最大 response length | 512 |
| Actor learning rate | `1e-6` |
| Group size | 8 |
| 每次 update 的 groups | 16 |
| 并行环境数 | 128 |
| Mini-batch | 256 |
| Rollout temperature | 1.0 |
| Validation temperature | 0.4 |
| Training iterations | 150 |
| IQR coefficient `lambda` | 1.5 |
| `alpha/beta/gamma` | 0.01 |
| `omega` | 1 |
| Global KL coefficient | 0.01 |

**可借鉴点：** 对当前实验，应联合观察 token-level entropy、tool-choice entropy、action-space 大小和 reward，而不是只看 `actor/entropy`。

---

## 5. SkillRise

### 5.1 模型与训练结构

- Qwen3-1.7B
- Qwen3-4B

训练将相关任务组成 task sequence：

- 每个 batch：16 条 task sequence；
- 每条 sequence：3 个任务；
- 每个任务：8 次独立 rollout；
- 每次 update：`16 × 3 × 8 = 384` 次 task plays；
- Actor learning rate：`1e-6`；
- Mini-batch：128；
- `gamma=0.6`；
- 使用 FSDP；
- Evaluation temperature：0.7。

### 5.2 奖励

**Task solving reward** 使用当前任务的最终 success reward。

**Skill curation reward** 使用后续任务的 discounted reward，衡量当前整理出的 skill document 对未来任务是否有帮助。

### 5.3 Prompt

- expert agent 身份；
- 当前 skill document；
- 初始 observation；
- 当前 trajectory；
- admissible actions；
- 逐步 reasoning 要求；
- 最终动作放在 `<action>...</action>` 中。

SkillRise 的额外信息是动态 skill document，可记录某类任务的导航经验、操作顺序和失败规避策略。

---

## 6. From History to State

该工作使用 Qwen3-8B，先进行 SFT，再从 SFT adapter 初始化在线 RL。其核心是用 deterministic task tracker 将完整历史压缩成 compact state block。

### 6.1 RL 参数

| 参数 | 设置 |
|---|---:|
| Rollouts/group | 4 |
| Rollout temperature | 0.8 |
| Top-p | 0.95 |
| Learning rate | `5e-6` |
| Gamma | 0.98 |
| KL coefficient | 0.02 |
| Gradient clip | 1.0 |
| 每个 task family 训练 games | 200 |
| 最大 episode steps | 30 |
| Max new tokens | 64 |
| Evaluation temperature | 0.4 |

### 6.2 Shaped reward

正向奖励：

```text
成功终止：+3.0
推进 tracked subgoal：+1.0
访问新的 hinted location type：+0.02
打开新的 hinted container：+0.05
到达目标位置：+0.2
打开目标 receptacle：+0.2
正确放置：+0.5
```

负向奖励：

```text
invalid action：-0.3
no-effect：-0.2
repeated no-progress：-0.2
late-stage regression：-0.3
错误目标实例：-0.3
重复访问 location：-0.1
重开已搜索 container：-0.1
search 后 wandering：-0.15
每步 cost：-0.01
```

reward 由固定规则计算，RL 过程中不调用 LLM judge。

> 注意：这套 reward 不能直接改入当前受“只允许修改 KL/Entropy”约束的 Qwen3.5-2B 实验；这里只作为后续 reward ablation 参考。

---

## 7. Agent2 RL-Bench

Agent2 RL-Bench 不是单一 ALFWorld RL 算法，而是研究 LLM 自动设计 RL pipeline 的工程化案例。

### 7.1 配置

- Base model：Qwen2.5-7B-Instruct；
- controlled study：Qwen3-8B Base；
- ALFWorld evaluation split：134 episodes；
- 单次时间预算：12 小时；
- 代表性 pipeline：offline expert SFT → online beta-mixed DAgger → replay-weighted SFT；
- 最大步数：12；
- temperature：0.4。

### 7.2 Action score

```text
admissible action：+2
exact expert action：+1.5
新 action：+0.25
重复 action：-0.40
think：-0.50
look/inventory：-0.20
invalid action：大幅负分
```

它说明，在 ALFWorld 中，动作合法性、专家动作一致性、重复动作和 invalid action 的处理，可能比单纯增加 entropy bonus 更直接影响结果。

---

## 8. 跨论文共同结论

### 8.1 模型规模

常见模型包括：

- 小模型：Qwen2.5-1.5B、Qwen3-1.7B；
- 中等模型：Qwen2.5-7B、Qwen3-4B、Llama3.1-8B；
- 其他模型：Mistral-7B、Gemma-7B、Llama3-8B、Qwen2.5-14B。

Qwen3.5-2B 最接近 Qwen2.5-1.5B 和 Qwen3-1.7B 的实验设置，不应直接照搬 7B/8B 的 batch 或并行规模。

### 8.2 Reward

最常见的 baseline 是：

```text
success=1
failure=0
invalid action=-0.1
```

主要改进路线：

- SALT：改 step-level advantage；
- STEP-HRL：增加 subtask/local-progress 信号；
- STAPO：只对异常 step 施加 selective penalty/reward；
- From History to State：使用固定 progress-shaped reward。

### 8.3 Prompt

推荐的通用输入结构：

```text
system instruction
+ task description
+ current step count
+ 最近 3 步 history
+ current observation
+ admissible actions
+ 严格输出格式
```

### 8.4 步数和温度

- 标准最大 episode steps：30～50；
- 50 步是更常见的通用设置；
- 12 步更像受限工程 baseline；
- 训练 rollout temperature：0.8～1.0；
- 验证 temperature：0.4～0.7；
- 层次化 low-level executor 有时使用 temperature=0。

### 8.5 常见训练超参数范围

| 参数 | 常见范围/值 |
|---|---|
| Actor learning rate | `1e-6`～`5e-6`；STEP-HRL 使用 `1e-5` |
| KL coefficient | 0.01 或 0.02 |
| Rollouts/group | 4 或 8 |
| Group size | 8 |
| Clip ratio | 0.2 |
| Gamma | 0.6、0.98、0.99 |
| Max prompt length | 2048 |
| Max response length | 512；部分任务使用更长输出 |

---

## 9. 对当前 Qwen3.5-2B 实验的建议

### 9.1 联合监控指标

不要只看 token-level `actor/entropy`，建议同时记录：

```text
actor/entropy
actor/tool_choice_entropy
actor/action_path_entropy
critic/rewards/mean
actor/kl_loss
tool_call_counts/mean
invalid_action_count
unknown_tool_count
format_error_count
```

更可靠的熵坍缩判据是：

```text
actor/entropy 持续下降
+ tool_choice_entropy 接近 0
+ tool_call_counts/mean 接近 0
+ reward 停滞或恶化
```

STAPO 的启示是，还应结合 admissible action 数量进行 normalized entropy 分析。

### 9.2 区分三类问题

1. **策略分布问题**：entropy 过低/过高、KL 持续升高、动作选择单一；
2. **工具协议问题**：unknown tool、malformed tool call、parser 无法识别动作；
3. **任务策略问题**：重复 look/inventory、no-effect、不推进 subgoal、reward 不升高。

如果工具协议问题占主导，只调 KL/Entropy 不能彻底解决训练失败。

### 9.3 推荐的后续 baseline

#### Baseline A：标准 GRPO

```text
模型：Qwen3.5-2B
reward：success=1，failure=0，invalid=-0.1
max_steps：50
group_size：8
rollout_temperature：1.0
evaluation_temperature：0.4
learning_rate：1e-6
KL：0.01
```

Prompt 使用 task、step count、最近 3 步 history、observation 和 admissible actions。

#### Baseline B：SALT-style

保持环境 reward 不变，只修改 advantage：

- 同一 prompt 采样多条 rollout；
- 合并重复状态；
- 比较 state-action 的后续结果；
- 重新分配 step-level advantage。

#### Baseline C：progress-shaped reward

作为独立 reward ablation，使用固定的 progress/no-progress 规则验证 sparse reward 是否过稀疏。该实验不能与当前只允许修改 KL/Entropy 的实验混用。

---

## 10. 最终结论

2026 年 ALFWorld 大模型实验大致形成了以下范式：

```text
instruction-tuned 小/中型模型
+ admissible-action Prompt
+ 严格 tool-call 格式
+ 30～50 步 episode
+ 1e-6 级别 actor learning rate
+ 4～8 条 rollout/group
+ success/failure sparse reward 作为 baseline
+ step-level credit assignment 或 progress signal
```

对 Qwen3.5-2B，最应优先借鉴：

1. **SALT**：不改变 reward，改善 step-level advantage；
2. **STEP-HRL**：使用 subtask/local-progress 表示；
3. **STAPO**：使用 normalized entropy 和 selective trajectory 分析；
4. **SkillRise**：参考 Qwen3 系列的 Prompt 和跨任务 skill document。

当前实验阶段建议优先保证：

- 工具名和动作格式严格一致；
- admissible actions 正确注入 Prompt；
- 统计 invalid/unknown/malformed action；
- 联合观察 reward、tool entropy 和 KL；
- 不将其他论文的 reward、batch 或 rollout 参数未经验证地直接复制到现有实验。
