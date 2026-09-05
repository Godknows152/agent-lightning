# ALFWorld Qwen3.5-2B：Prompt/schema 修复与重启

## 改动

- 第一轮先创建实际环境，再从实例读取 observation 和 admissible actions；不再依赖 parquet 里可能过期的动作表。
- 每轮从该轨迹的最新可用动作生成 `action.enum`，传给 chat template；解析器使用相同 schema 的 typed 副本，不修改共享工具对象。
- 工具执行仍严格检查动作是否属于实际环境的可用集合。**enum 是模型可见约束，不代表启用了语法约束解码，也不保证模型不产生非法动作。**
- 每一步重新生成 task、当前 observation、最近 3 轮 action/反馈摘要及当前动作表。保留动作表可读性；它与 enum 来自同一状态。
- 删除重复任务行和自定义 XML 工具格式指令，工具格式由原生 chat template 提供。
- terminal/empty-action 状态不再尝试生成下一轮 schema。

## 额外修正：推理与训练上下文一致

前一次未提交的 Prompt 修复仅替换推理上下文，训练概率仍以旧完整轨迹为条件，这是不正确的。
本次记录每轮实际 prompt token、response token 和原轨迹中的偏移；FSDP 将每轮 prompt+response 作为独立 packed sequence 前向，位置从零开始，然后把生成 token 的 logprob/entropy 可微映射回原轨迹的 loss slots。
Actor 更新、old-policy 概率和 reference KL 都走相同 replay 路径；原 GRPO 分组、奖励、response mask 和归一化保持不变。
仅包含 `alfworld_turn_contexts` 的样本启用此路径；不改变图像恢复实验。

## 参数保持不变

- 模型：`/home/LXJ/Python_Projects/Models/Qwen3.5-2B`
- KL：`use_kl_loss=true, kl_loss_coef=0.0032, kl_loss_type=low_var_kl`
- Entropy：`entropy_coeff=0.02`
- seed=0，batch=8，rollout n=4，2 GPU，150 updates；不加载旧实验 checkpoint。

## 清理

- 旧主进程 PID 2064533 已停止，GPU 0/1 已释放。
- 旧目录 `attempt_004_kl0.00320_e0.02000`（checkpoint、rollout、log、本地 SwanLab）已删除。
- `/home/LXJ/tmp/ray` 已清空；`/tmp/ray` 无残留。
- 旧云端 run `Godknows/ALFWorldRL/2jsvjpc7` 使用已安装 SDK `Api().run(...).delete(commit=True)` 删除成功，后续查询确认。
- 保留其他用户在 GPU 2/3 上的工作以及其他实验、模型、数据和源代码。

## 验证与新实验

- 回归覆盖：初始与后续 schema 同步、并发轨迹隔离、3 轮历史、终止状态、训练 replay token 映射与梯度、错配时拒绝训练。
- 新输出目录：`outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_005_prompt_schema_v4_kl0.00320_e0.02000`
- 重启状态和首次更新指标在启动验证后补记；启动进程不等于已完成训练更新。
