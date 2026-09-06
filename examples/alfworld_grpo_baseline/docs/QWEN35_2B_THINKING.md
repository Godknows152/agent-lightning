# Qwen3.5-2B：先思考，再调用 ALFWorld 工具

更新：2026-09-06。

## 配置与生效范围

`config/alfworld/qwen35_2b/v1/alfworld_config_2gpu.yaml` 覆盖：

```yaml
data:
  apply_chat_template_kwargs:
    enable_thinking: true
```

只改变 Qwen3.5-2B 的默认行为；共享配置及其他模型不变。
现有进程不会热更新配置。此次未停止、删除或重启当前实验；下一次通过
`scripts/alfworld/qwen35_2b_v1.sh` 新启动训练时生效。

## 协议

- 每一步依然只提供目标、当前观察、当前合法动作及动态 schema。
- 模板以开放的 `<think>` 开始生成；模型结束思考后输出 `</think>`，然后输出一个 XML 工具调用。
- 格式约束只检查思考结束后的输出，不把思考文本当作额外前缀惩罚。
- 解析器忽略思考中的工具示例；未闭合思考时不执行任何工具。
- 思考结束后的多调用、解释、后缀仍沿用原有格式惩罚；动作合法性校验不变。
- 原始思考 token、工具 token 和相应 logprob 全部保留在训练序列中；解析器的临时文本视图不替换训练 token。
- 思考不作为下一轮历史上下文；每一步重新思考当前状态。

## 未改变的设置与限制

KL = 0.004，Entropy = 0.02；学习率、数据、batch、奖励及生成预算不变。
当前 Qwen3.5-2B 整条轨迹生成预算已提高为 **4096 token**，思考也占用该预算。
开启思考仍可能减少有效交互轮数，或导致思考未结束就耗尽预算。
模板不再使用“briefly/简短”限制，只要求模型在当前状态下完成必要推理后调用工具；不能仅靠开关保证任务表现提高。

## 验证

- ALFWorld 回归测试：30 passed。
- 标准启动脚本 `--preflight` 通过，模板分别测试 thinking 开关两种前缀。
- 独立协议脚本 `--no-generate --enable-thinking` 渲染通过。
- 新测试覆盖思考中的 XML 不执行、未闭合思考不执行、保留 token/logprob、思考后的非法附加文本仍惩罚。
- 未额外占用训练 GPU 进行真实模型生成；尚未验证 thinking 模式完整训练 update 或任务收益。
- preflight/test 退出时仍有既有 `multiprocess.ResourceTracker` 析构警告；进程退出码为 0，不属于本次修改。

诊断脚本支持 `--enable-thinking` / `--no-enable-thinking`；Qwen3.5-2B 默认启用。
真实生成诊断时显式指定合理的 `--max-new-tokens`，原默认 128 不保证足够覆盖思考和调用。
