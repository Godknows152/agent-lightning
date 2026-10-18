# 原生无熵正则_1017

本实验从 `/home/LXJ/Python_Projects/Models/Qwen3.5-2B` 原始发布权重开始，
Actor 和冻结 reference 使用相同初始权重，不加载项目 SFT；初次启动不恢复旧训练。
其余配置复制 `no_branch_entropy_1012`，仅新增思考/action 一致性奖励。

## 奖励定义

每次模型输出经现有 XML 解析器确认为合法单次工具调用后，取
`<parameter=action>` 内的完整 action（例如 `go to cabinet 1`）。
在该次生成的 `</think>` 之前的思考文本中，若此 action **出现至少两次（>=2）**，
该次决策奖励固定加 0.1；出现 0 次或 1 次不加分，更多次也不额外加分。可选的 `<think>` 开始标签不参与匹配。

匹配大小写敏感、保留 action 内部空白、使用完整词边界；
`go to cabinet 10` 不算 `go to cabinet 1`，`looking` 不算 `look`。
不对思考内容做语义推断。最终 XML 中的 action 不计入次数。
未闭合思考、无 XML、多个调用或无效 XML 均不加分。
这里的工具名称是完整 action，不是所有调用共有的 `alfworld_action`。

奖励不额外要求任务成功或 action 在当前 admissible commands 中。
合法 XML 中不可执行的 action 仍可能满足文字一致性条件，同时保留原有无效动作惩罚。
每步奖金累积到 episode reward；仍沿用 baseline 的 trajectory GRPO，
最终轨迹奖励和 advantage 广播给该轨迹所有实际生成步。
没有改变 response token、mask、logprob 或原有任务成功/无动作/无效动作/重复动作奖励。

## 配置与监测

- `THINKING_ACTION_REPEAT_REWARD=0.1`；其他实验默认 0，不启用此奖励。
- branch entropy 关闭；**普通 token entropy 仍为 0.001**，KL 仍为 0.001，保持 baseline 设置。
- 训练 batch 16、每 prompt 8 条轨迹、学习率 3e-6、3 epochs，使用 GPU 0、1。
- 输出：`outputs/qwen3.5_2B/原生无熵正则_1017/`。
- 日志：`log/alfworld/qwen35_2b/原生无熵正则_1017/full/`。
- SwanLab 项目 `ALFWorldRL`，实验名 `原生无熵正则_1017`。
- 新增指标：`alfworld_reward/thinking_action_repeat_total`、`thinking_action_repeat_mean`、
  `thinking_action_repeat_count`、`base_reward_mean`；保留任务成功计数。
- 验证的 reward_extra_info 分开记录 `score`、`base_reward`、`thinking_action_reward`、`success`。

由于最多 50 次环境决策，新增奖金最高可达 5，任务成功奖励为 10。
因此比较实验时同时观察原任务成功率和 base reward，不能仅凭总奖励判断任务能力提升。

## 启动与验证

```bash
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_raw_once_1017.sh --preflight
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_raw_once_1017.sh
```

默认后台运行。配置使用 resume_mode=auto，默认恢复最后一个完整检查点及当前 SwanLab run。
测试覆盖精确匹配、重复提及、缺失/多次 XML、原有惩罚、原生 step 数据完整性、
奖金每轨迹只累计一次及配置与 baseline 的完整差异核验。

## 2026-10-18 续训调整

初始实验使用 +1 奖励；按用户要求停止后，从 `global_step_40` 恢复，奖励改为 +0.1。
原始模型/reference、其他超参数、奖励规则和输出目录不变；恢复 Actor、优化器、调度器、随机数及数据状态。
原 SwanLab run `6bcln7mt` 保留，本次用一次性参数
`variables.SWANLAB_NEW_RUN_ON_RESUME=true` 新建同名 run；该参数默认 false。
新 run 在 step 40 记录恢复标记，首个优化后的训练指标为 step 41，不复制之前的曲线。
原/新 run ID 及来源检查点保存在输出目录的 `swanlab_identity_history/`，
后续普通重启继续连接新 run。

## 2026-10-18 重复提及奖励续训

按用户要求停止原训练，从最新完整 `global_step_70` 恢复。将 thinking 中
完整最终 action 恰好出现一次加 0.1，改为出现至少两次加 0.1（每步固定，不按次数累乘）。
系数改用 `THINKING_ACTION_REPEAT_REWARD=0.1`，原 `THINKING_ACTION_ONCE_REWARD` 不启用。
实现保留旧 once 配置的语义，但两种奖励不可同时启用。指标同步改名为 `thinking_action_repeat_*`。
其他模型、优化器、奖励和训练超参数不变。旧 run `02ql4okz` 及其训练日志保留；
检查点之后的旧 rollout 71–73 已移入输出目录的 `restart_repeat_0p1_20261018_*` 归档，
防止恢复后同名 rollout 覆盖旧样本。使用 `variables.SWANLAB_NEW_RUN_ON_RESUME=true`
新建同名实验，从 step 70 记录恢复标记，首个优化步为 71。
