# ALFWorld 合法动作分叉熵（实验版本）

该功能在 **loss 层面**增加动作探索正则，保留原版轨迹 GRPO 的奖励、优势、PPO 和 KL。
原版 `qwen35_2b_v1.sh` 默认行为不变。实现全部放在本目录，未修改共享的原生 veRL checkout。

## 正则作用的位置

1. 每个环境步骤保存执行动作前的 admissible actions。
2. 严格解析 XML 中最终执行的动作，寻找它在本轮 thinking 中的第一次完整文本出现。
3. 在相同文本前缀下 tokenize 所有合法命令，并校验所选命令对应的 token 与真实生成 IDs 一致。
4. 沿所选命令路径，仅保留具有至少两个不同合法下一 token 的分叉。共享 token 去重。
5. 从 Actor 已有前向 logits 中提取这些候选 token，计算可微熵，随原 PPO loss 一次反向传播。

例如合法命令为 `look`、`go to cabinet 1`、`go to cabinet 2`、`go to drawer 1`，
thinking 首次写出 `go to cabinet 1` 时，参与正则的是 `go/look`、`cabinet/drawer`、`1/2`。
共同的 `to`、工具 schema 标签等不参与正则。编号也是合法动作区别的一部分。

候选路径包含动作之后的第一个边界 token，使用该次输出的实际续接文本进行上下文分词。
因此遇到一个命令是另一个命令的前缀时，可区别“在此结束”与“继续更长命令”。
没有为每个候选动作额外执行模型 forward，也不在线计算梯度归因。

## 目标与归一化

分叉节点的合法下一 token 集合为 `C`，训练采样温度为 `T`：

```text
q(v) = softmax(logits[C] / T)[v]
H_node = -sum(q * log(q))
H_decision = mean(H_node over this decision's branching nodes)
H_trajectory = sum(H_decision over all decisions) / trajectory_length
H_batch = mean(H_trajectory over real trajectories)
L_actor = L_original_GRPO_and_KL - coefficient * H_batch
```

使用自然对数；实际优化原始熵，`H_node / log(|C|)` 仅用于诊断。
没有匹配位置或只有一个候选时，该步熵为零，仍保留原来的 GRPO 样本和轨迹分母。
位置索引指向生成 token，提取 logits 时使用 `prompt_length + response_position - 1`。

原生 V1 会把整批步骤切为多个优化 mini-batch。代码在切分前给第 `i` 条轨迹每步设置
`1 / trajectory_length_i` 的权重；每个 mini-batch 使用固定分母
`trajectory_count * mini_batch_size / padded_row_count`，并乘原生 DP size 来抵消梯度平均。
因此在固定模型参数下，各 mini-batch 目标的平均值等于上述整批轨迹均值；动态 micro-batch
拆分或重排不改变梯度。真实训练仍按原生流程逐 mini-batch 更新参数。
合成 padding 权重为零，继承的旧分叉元数据不会参与索引；被 loss mask 拒绝的 token 也不参与。

这里优化的是**已采样思考前缀上的局部条件熵**，不等于完整命令分布的精确熵。
分叉均匀也不等于所有完整命令均匀，不能据此保证任务成功率提高。
条件熵升高时，全词表中落在合法候选集合的概率仍可能降低，因此同时记录合法概率质量。

## 配置与入口

- 配置：`config/alfworld/qwen35_2b/action_branch_entropy/alfworld_config_2gpu.yaml`
- 启动脚本：`scripts/alfworld/qwen35_2b_branch_entropy.sh`
- 开关：`actor_rollout_ref.branch_entropy.enabled`
- 系数：`actor_rollout_ref.branch_entropy.coefficient=0.001`。此前 `0.003` 的冻结样本校准
  仅用于初始量级参考，不能代表实际训练梯度比或最优系数。
- 原全回答熵：`actor.entropy_coeff=0.001`、`actor.calculate_entropy=true`，与原版 v1 一致。
  原生 PPO/KL/全回答熵损失计算后，再减去分叉熵项；两种熵同时参与反向传播。
- 基底：2026-10-09 归因实验使用的 action-explicit SFT `checkpoint-100` 的合并全模型。
  可用 `ALFWORLD_SFT_MODEL` 指定另一份完整模型；Actor 与冻结 reference 仍从同一模型初始化。
- 实验名：`branch_entropy_1009`。
- 输出：`outputs/qwen3.5_2B/branch_entropy_1009`。
- `resume_mode=disable`，不承接之前的 GRPO checkpoint；3 epochs、自动计算步数。
- LR 上限 `3e-6`、下限 `1e-6`、20% warmup、余弦衰减；训练/验证采样继承 v1。
- 奖励：成功 10；无动作/非法动作每次 −0.2；失败轨迹每次重复 −0.1，成功轨迹豁免重复扣分。

实现阶段已完成 CPU 测试和配置校验。2026-10-09 调整后的训练使用物理 GPU 0、1，可通过以下命令启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_branch_entropy.sh
```

2026-10-12 新增独立的无分叉熵对照配置
`config/alfworld/qwen35_2b/no_branch_entropy/alfworld_config_2gpu.yaml`，
相对 `branch_entropy_1009` 仅将训练开关 `actor_rollout_ref.branch_entropy.enabled` 设为 `false`。
普通熵系数仍为 `0.001`，其余模型、奖励、优化器、训练采样和种子配置相同。
后续按要求将此对照配置的验证改为每个任务随机采样 1 条完整轨迹（`val_kwargs.n=1`，
共 140 条），训练仍为每个任务 8 条；验证温度、top-p、top-k 和每 10 步验证的频率保持不变。
禁用后不构造分叉元数据，也不使用分叉熵 Actor 引擎；保留的 `coefficient` 值不参与损失。
实验名为 `no_branch_entropy_1012`，产出独立保存于
`outputs/qwen3.5_2B/no_branch_entropy_1012`。首次运行从同一份 SFT 模型开始；
当前 `resume_mode=auto`，后续启动会恢复该目录最新的完整检查点，并接续原 SwanLab 实验。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_no_branch_entropy.sh
```

ALFWorld worker 在 Ray 保留公共 CUDA 可见列表时，将分配到的物理 GPU ID 映射为
进程内 CUDA 序号，因此 `0,3` 对应 `cuda:0,cuda:1`，同时保留 SGLang 权重传输所需的公共视图。
该适配不修改共享 veRL 文件或训练目标。

2026-10-12 续训时，参考模型的 FSDP CPU 锁页分配因主机内存碎片失败。
无分叉熵配置因此启用 `actor_rollout_ref.reference_resident=true`，并设置
`actor_rollout_ref.ref.fsdp_config.param_offload=false`，让冻结参考权重常驻 GPU。
原生 FSDP 会强制卸载 forward-only 模型，故由独立的 ALFWorld 参考引擎仅在包装模型时
关闭此行为；模型加载精度、分片、参考概率计算和无优化器的推理语义保持不变。
其他配置默认不启用该适配。比较两组实验的运行时间和显存时，需要计入这一存放方式差异。
SwanLab 收尾也已修复，Python 训练异常会上报 `crashed`，而非默认的 `success`。
相关吞吐试验、失败原因和完整续训步验证记录见
[`docs/throughput_tuning_step60_20261012.json`](docs/throughput_tuning_step60_20261012.json)。

当前版本明确限定原生 V1 同步、trajectory GRPO、FSDP/FSDP2、packed logits、Actor SP=1、
未融合 LM head，以及 rollout log-prob bypass。Rollout 自身仍可 TP=2。
不支持的配置在创建 Ray/模型之前报错，不会静默回退成另一套熵算法。

## 诊断指标

| 指标 | 含义 |
| --- | --- |
| `actor/branch_entropy` | 轨迹等权的原始分叉熵 |
| `actor/branch_entropy_normalized` | 用每个节点的 `log(候选数)` 归一化后的熵 |
| `actor/branch_legal_mass` | 被监测节点在全词表中分配给合法下一 token 集合的概率质量 |
| `actor/branch_entropy_loss` | 实际加入 Actor loss 的负熵项 |
| `actor/branch_entropy_coefficient` | 当前正则系数 |
| `alfworld_branch/match_fraction` | 最终动作在 thinking 中成功对齐的步骤比例 |
| `alfworld_branch/eligible_fraction` | 实际有分叉节点的步骤比例 |
| `alfworld_branch/node_count` | 当前 rollout 的分叉节点总数，不含 padding |
| `alfworld_branch/*_count` | matched、no_action、not_admissible、no_thinking、no_mention、token_mismatch、unstable_prefix、ambiguous_boundary |

前三项按所有轨迹和步骤归一化，无节点的步骤记零；解读时须结合 coverage。
`token_mismatch` 表示解码再编码与实际生成 token 不一致；`unstable_prefix` 表示候选替换
改变了动作之前的分词。两者均跳过本步正则，保留原训练 tokens，不能偷偷改写或缩小合法候选集。
`ambiguous_boundary` 表示实际续接 token 仍不能区别短命令与更长命令，也跳过该步正则。

## 验证与计算开销

`tests/test_branch_entropy.py` 覆盖真实 tokenizer、首次出现、命令前缀结束分叉、原始 ID 漂移、
因果偏移、合法 token 梯度与熵增方向、全词表合法质量、padding/no-match、动态行拆分重排、
DP/轨迹归一化、原生 PPO+KL 联合梯度、Flash CE 原地反向的兼容性、环境状态快照、
原生 AgentLoop 队列序列化、配置隔离和不支持配置拒绝。

额外工作包括 CPU 候选命令分词、少量合法候选 logits 的 gather/softmax。
一个 micro-batch 的所有候选共用一次 gather，避免逐节点反向散射整张 logits 大小的梯度。
另外，每个选中节点会计算用于诊断的 detached 全词表 logsumexp；此项不反向传播。
没有额外模型 forward，实际吞吐与显存增量仍需另行进行 GPU 训练测量。

此前系数 `0.003`、关闭普通熵的试验在 GPU 0、3 完成了首个完整训练 step。
该试验（SwanLab `royw4m12`）已按要求停止并删除本地和云端产出，以下仅保留历史验证结果。
首批 128 条轨迹、6156 个真实决策中，5810 步有合法分叉（94.38%）；
`actor/branch_entropy=0.650023`，`actor/branch_entropy_loss=-0.00195007`，系数为 `0.003`。
Actor 最大 allocated/reserved 显存为 49.66/58.78 GiB，首步总耗时约 718 秒；
这不是与旧正则控制变量对比的性能基准，不能据此推断算法的额外开销或训练收益。
启动期间修复了非连续物理 GPU 编号映射；另由内核日志定位到主机内存碎片导致锁页分配失败，
完成一次主机内存整理后恢复运行，未修改学习率、奖励、采样规模或共享 veRL 文件。

代码位置：`src/alfworld_baseline/branch_entropy.py`（定位/校验/权重）、
`branch_entropy_engine.py`（Actor 引擎、loss、V1 batch 接入）、`agent_loop.py`（逐步状态快照）。
