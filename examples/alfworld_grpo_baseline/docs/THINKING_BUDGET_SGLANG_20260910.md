# Qwen3.5-9B / SGLang thinking budget 实测（2026-09-10）

## 实验范围

- 本地 Qwen3.5-9B 基础权重，TP=2，两张 A800 80GB。
- 使用训练的 `.pydeps` SGLang 0.5.13.post1 和 VERL 的 `_set_envs_and_config` 兼容初始化。
- 启用 LoRA runtime（rank=16、与训练一致的 target modules），**未加载已训练 LoRA 权重**。
- bf16、FA3、eager、`mem_fraction_static=0.2`；使用 v5 提示词、thinking 开启、无 schema。
- `temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=256`；使用训练的 EOS ID。
- 从实际 TextWorld 环境获取 4 个任务，各取初始状态及执行一步合法动作后的状态（带历史）：8 个 prompt。
- 主实验 8 种条件、64 个响应；追加 3 种换行结束条件、24 个响应，总计 88 个响应。
  每条件8个响应，这是长度控制和接口验证，不是任务成功率评测。
- `Engine.generate(input_ids=...)` 构造原生 `GenerateReqInput`，进入与训练相同的
  `tokenizer_manager.generate_request`；**没有运行完整 Ray/GRPO loop**。

## 接口核对（推荐独立采样使用 newline 结束版本）

直接使用 `SamplingParams(thinking_budget=32)` 会报 `unexpected keyword argument`。
单独在 `sampling_params.custom_params` 填预算，也不会自动开启预算控制。
SGLang 原生机制需要：

```python
# Engine 启动参数
engine = sgl.Engine(..., enable_custom_logit_processor=True)

# 独立采样请求
engine.generate(
    input_ids=prompt_ids,
    sampling_params={
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "custom_params": {"thinking_budget": 64},
    },
    custom_logit_processor=ALFWorldNewlineThinkingBudgetLogitProcessor.to_str(),
    return_logprob=True,
)
```

`custom_logit_processor` 是 GenerateReqInput 的顶层字段，不是 SamplingParams 字段。
当前 VERL SGLang server 的 generate 方法尚未转发这个自定义字段；不能把上面的
`thinking_budget` 原样当成已有训练 Hydra 参数使用。

## 为什么原生 Qwen3 processor 无效

1. 内置 Qwen3 ID 为 151667 / 151668；本地 Qwen3.5 的 ID 是 **248068 / 248069**。
2. 即使替换 ID，内置实现扫描整个 prompt，只要发现 `</think>` 就跳过预算控制。
   v5 system instruction 本身含 `<think>...</think>`，所以会误判。
3. 实验 processor 仅识别 prompt 中最后一个未闭合的 `<think>`，按新生成 token 数计数。
   达到预算且尚未自然关闭时，强制下一个 token 为 `</think>`。

直接关闭版本的预算含义：生成的思考 token 上限，不含 prompt 预填的换行、不含关闭标签。
newline 版本按 SGLang 原生行为，在达到预算后必要时额外补1个换行，再关闭思考。
因此该版本实测 `</think>` 前有 B+1 个 token，其中最后一个是额外的分隔换行。
模型可以提前自然结束；预算不是必须生成的固定长度。`budget=0` 在生成第一步就关闭思考。
总输出上限仍是 256，预算限制不会自动扩大总输出容量。

## 训练安全边界

自定义 processor 把预算边界处的 logits 变成只允许 `</think>`，因此返回该 token 的
采样 log-prob 为 0。原始模型在同一上下文下的 `</think>` 概率一般远小于 1。

当前训练使用 `bypass_mode=true`，直接把 rollout log-prob 作为 old log-prob；
而 FSDP replay 的 forward 没有这个采样 processor。若不做额外处理，强制 token 的
old/new log-prob 会来自不同分布，影响 PPO ratio、KL/entropy 和梯度解释。

**本次仅启用独立诊断配置，没有直接在生产 PPO 中启用。** 要正式接入，应保留
强制关闭 token 作为上下文，但明确标记该确定性边界，并让 loss mask、KL/entropy、
rollout correction 和 turn-context replay 对这些位置一致处理；不能仅在采样时硬截断
然后将所有 token 都按自由采样 token 训练。

## 文件与复现

- 配置：`config/diagnostics/thinking_budget_qwen35_9b.json`
- 适配 processor：`src/alfworld_baseline/thinking_budget.py`
- GPU 脚本：`scripts/test_sglang_thinking_budget.py`
- CPU 测试：`tests/test_thinking_budget.py`
- 一键复现（先确认 GPU 0/1 空闲）：

```bash
bash examples/alfworld_grpo_baseline/scripts/test_sglang_thinking_budget.sh
```

测试输出包含 `configuration.json`、`api_probe.json`、`prompts.json`、`results.jsonl`、
`summary.json`、`logprob_comparison.json` 和 `run.log`。保留原始 prompt IDs、输出 IDs、
逐 token log-prob、动作有效性与结束原因；同一输出目录不能重复运行混入旧结果。

## 实测结果：初始状态 + 带历史状态

| 条件 | 预算 | 闭合思考 | 有效动作 | 达到上限要求 |
|---|---:|---:|---:|---|
| baseline | 无 | 4/8 | 4/8 | None |
| custom_params_only | 32 | 2/8 | 2/8 | False |
| stock_qwen3 | 32 | 3/8 | 3/8 | False |
| qwen35_ids_only | 32 | 4/8 | 2/8 | False |
| last_open_budget_0 | 0 | 8/8 | 7/8 | True |
| last_open_budget_32 | 32 | 8/8 | 2/8 | True |
| last_open_budget_64 | 64 | 8/8 | 2/8 | True |
| last_open_budget_128 | 128 | 8/8 | 2/8 | True |

32/64 预算下各 8 条响应的关闭位置均为第 32/64 个生成思考 token 之后。
128 预算下关闭位置为 `[128,128,128,108,128,128,128,128]`，有一条自然提前闭合。

**预算命中不等于动作格式正确或任务成功。** 强制闭合后，模型可能在标签外继续输出解释、
计划，甚至继续生成到总上限256 tokens。v5 要求闭合后仅一行动作，因此这些输出仍会
被判为无动作。上表的有效动作只表示命令属于当前合法动作集合，不代表成功完成任务。
小样本每条件8条、随机采样，不能据此声称预算0最佳、预算能提高成功率或稳定加速。

### 强制 token 的实际 log-prob 差异

| 样本 | SGLang 强制采样的 log-prob | 同一模型无约束重评分的 log-prob |
|---|---:|---:|
| 0 | 0.0 | -22.484465 |
| 1 | 0.0 | -26.448462 |
| 2 | 0.0 | -24.717463 |
| 3 | 0.0 | -27.007845 |

这不是“只改一个计数参数”的无副作用优化，正式训练必须处理上述分布差异。

原始实验目录（绝对路径）：`/home/LXJ/Python_Projects/Agent_Lightning/examples/alfworld_grpo_baseline/outputs/diagnostics/qwen35_9b/thinking_budget_20260910_165344/with_history_and_logprob`。

## 补充对照：先补换行，再关闭思考（更推荐）

直接在一个词或句子中间强制 `</think>`，容易让模型在标签外继续解释。
为排除自定义结束格式引入的干扰，补测了与 SGLang 原生 processor 一致的
“达到预算 → 必要时强制换行 → 强制关闭标签”行为；仅修正 token ID 和最后思考块检测。
模型、环境状态、提示词、采样参数不变。

| 条件 | `</think>` 前 token 数（包含额外分隔换行） | 闭合数 | 有效动作数 |
|---|---|---:|---:|
| newline_budget_32 | [33, 33, 33, 33, 33, 33, 33, 33] | 8/8 | 7/8 |
| newline_budget_64 | [65, 65, 65, 65, 65, 65, 65, 65] | 8/8 | 8/8 |
| newline_budget_128 | [129, 129, 129, 129, 129, 129, 129, 129] | 8/8 | 8/8 |

补换行版本的32/64/128三档全部命中上限；64和128两档在本次小样本中有效动作均为8/8。
这支持在独立采样中优先采用 **newline 版本、64-token 内容预算**作为进一步实验起点。
不能从8条样本推断长期训练成功率，也不代表所有未来输出都合法。

24条补测中，强制换行和关闭标签两个位置的采样 log-prob 均为0，所以上述训练安全限制
仍然适用，不能仅因为有效动作增加就跳过 PPO 概率/掩码适配。

补测原始目录：`/home/LXJ/Python_Projects/Agent_Lightning/examples/alfworld_grpo_baseline/outputs/diagnostics/qwen35_9b/thinking_budget_20260910_165344/newline_termination_control`。

CPU 回归：106 项通过；包含预算边界、提前闭合、混合batch、序列化及换行结束行为。
