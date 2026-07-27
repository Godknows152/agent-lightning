# `tool_action_token_entropy` 熵正则化实施方案

## 1. 文档状态

- 状态：阶段 0、阶段 A、阶段 B 已实施；按本轮要求已同时启用阶段 C 的首轮系数。
- 当前 v3 系数：`entropy_coeff: 0.005`，`tool_action_entropy_coeff: 0.001`。
- 在线探索性指标：已增加 `actor/action_path_entropy` 和 `actor/tool_choice_entropy`。
- 适用训练链路：`examples/image_restoration_multi_agent/old_verl_grpo/` 启动的 Qwen3.5 多轮工具调用 GRPO。
- 实际 VERL 后端：`examples/image_restoration_multi_agent/verl_backend/`。
- 目标模型协议：`qwen3_coder` XML 工具调用格式。
- 目标工具：`restore_image`。
- 目标决策字段：`<parameter=action>...</parameter>` 中的 action 值。

本文所说的 `tool_action_token_entropy` 是“模型生成 action 值时，对这些 token 位置上的词表分布计算的熵”，不是在全部合法工具之间严格归一化得到的离散工具选择熵。后者需要计算每个多 token action 候选序列的完整概率，不在第一版实施范围内。

## 2. v1/v2/v3 版本定位与一键切换

### 2.1 版本语义

本次改造后的训练版本命名为 v3。三个版本的语义固定如下：

| 版本 | 奖励配置 | 全 response 熵 | `tool_action_token_entropy` | 用途 |
|---|---|---:|---:|---|
| v1 | 原始 `step_mixed_v1` 奖励 | `0.005` | `0.0` | 保留原始奖励基线 |
| v2 | `marginal_efficiency_v1`，工具调用成本 `0.12` | `0.005` | `0.0` | 保留当前效率奖励基线 |
| v3 | 与 v2 相同的 `marginal_efficiency_v1` 奖励 | `0.005` | 初始建议 `0.001` | 测试定向 action token 熵 |

因此，v3 不是第三套 IQA 奖励函数，而是在 v2 奖励基础上加入 `tool_action_token_entropy` 熵正则。这样 v2 与 v3 的主要实验变量只有定向熵项，便于判断探索性变化是否确实来自新正则。

如果后续需要测试 Tool-only 消融，应另外使用命令行覆盖或新建明确命名的消融脚本，不改变正式 v3 的定义：

```bash
bash train_sh/fog/fog_v3.sh \
  actor_rollout_ref.actor.entropy_coeff=0.0
```

### 2.2 版本隔离原则

实施后必须满足：

1. v1、v2、v3 都有独立启动脚本。
2. 启动不同脚本即可完成版本切换，不要求用户手工追加 Hydra 参数。
3. v1、v2、v3 的实验名、输出目录、主日志和工具日志互不覆盖。
4. v3 只能自动恢复 v3 输出目录中的 checkpoint，不能错误恢复 v1/v2。
5. 底层 VERL mask 和 loss 实现可以共享，但新配置默认关闭，不能改变 v1/v2 数值行为。
6. v1/v2 配置应显式写出 `tool_action_entropy_coeff: 0.0`，不能只依赖 dataclass 默认值。
7. v3 配置应显式写出奖励配置路径、普通熵系数和定向熵系数。
8. 启动日志必须打印版本名、最终 Hydra 配置名、奖励配置路径和两个熵系数。
9. `--preflight` 必须校验脚本声称的版本与最终组合配置一致。

### 2.3 目标启动脚本

在现有目录中新增四个脚本：

```text
examples/image_restoration_multi_agent/old_verl_grpo/train_sh/
├── fog/
│   ├── fog_v1.sh
│   ├── fog_v2.sh
│   └── fog_v3.sh
├── low_light/
│   ├── low_light_v1.sh
│   ├── low_light_v2.sh
│   └── low_light_v3.sh
├── rain/
│   ├── rain_v1.sh
│   ├── rain_v2.sh
│   └── rain_v3.sh
└── snow/
    ├── snow_v1.sh
    ├── snow_v2.sh
    └── snow_v3.sh
```

使用方式保持一致：

```bash
# 雾专家 v1：原始奖励
bash examples/image_restoration_multi_agent/old_verl_grpo/train_sh/fog/fog_v1.sh

# 雾专家 v2：边际效率奖励
bash examples/image_restoration_multi_agent/old_verl_grpo/train_sh/fog/fog_v2.sh

# 雾专家 v3：边际效率奖励 + tool_action_token_entropy
bash examples/image_restoration_multi_agent/old_verl_grpo/train_sh/fog/fog_v3.sh
```

其他专家只替换目录和专家名。

### 2.4 v3 配置文件

已增加一份共享 v3 覆盖配置和四份专家 v3 配置：

```text
examples/image_restoration_multi_agent/old_verl_grpo/config/
├── restoration_common_config_v3_2gpu.yaml
├── fog_v3_config_2gpu.yaml
├── low_light_v3_config_2gpu.yaml
├── rain_v3_config_2gpu.yaml
└── snow_v3_config_2gpu.yaml
```

共享 v3 配置只维护奖励路径和熵系数差异：

```yaml
# restoration_common_config_v3_2gpu.yaml
defaults:
  - restoration_common_config_2gpu
  - _self_

actor_rollout_ref:
  actor:
    entropy_coeff: 0.005
    tool_action_entropy_coeff: 0.001
    calculate_entropy: true
  rollout:
    multi_turn:
      tool_config_path: examples/image_restoration_multi_agent/old_verl_grpo/config/tool_config/restoration_tool_config_marginal_efficiency_2gpu.yaml
```

四份专家 v3 配置作为 Hydra 主配置，继承该共享 v3 配置，并分别设置专家数据、SFT LoRA、实验名和隔离目录。例如：

```yaml
# fog_v3_config_2gpu.yaml
defaults:
  - restoration_common_config_v3_2gpu
  - _self_

trainer:
  experiment_name: fog_v3
  default_local_dir: examples/image_restoration_multi_agent/old_verl_grpo/outputs/fog_v3
  ray_kwargs:
    ray_init:
      runtime_env:
        env_vars:
          SWANLAB_LOG_DIR: /home/LXJ/Python_Projects/Agent_Lightning/examples/image_restoration_multi_agent/old_verl_grpo/outputs/fog_v3/swanlab
          VERL_LOG_DIR: /home/LXJ/Python_Projects/Agent_Lightning/examples/image_restoration_multi_agent/old_verl_grpo/log/fog/v3
```

采用这一层级是因为 `hydra.searchpath` 必须出现在主配置中；直接把现有专家主配置嵌套为 defaults 会违反 Hydra 的该约束。共享 v3 覆盖仍保证四专家的奖励与熵设置只有一个维护点。

奖励配置不需要复制为一份内容完全相同的 v3 文件。v2 和 v3 都引用：

```text
config/tool_config/restoration_tool_config_marginal_efficiency_2gpu.yaml
```

这样能保证 v2/v3 奖励逻辑严格相同，避免两份 YAML 在后续维护中发生无意差异。版本区别由 v3 专家配置中的 `tool_action_entropy_coeff` 表达。

### 2.5 v1/v2 配置的显式兼容设置

在现有共享配置中增加：

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.005
    tool_action_entropy_coeff: 0.0
    calculate_entropy: true
```

v1/v2 继续继承该值，因此新 loss 代码存在但不生效。v3 薄配置只覆盖：

```yaml
tool_action_entropy_coeff: 0.001
```

v1 与 v2 仍通过各自脚本选择不同的奖励配置：

```text
v1 -> restoration_tool_config_current_iqa_2gpu.yaml
v2 -> restoration_tool_config_marginal_efficiency_2gpu.yaml
v3 -> restoration_tool_config_marginal_efficiency_2gpu.yaml
```

### 2.6 v3 启动脚本约定

由于启动脚本位于 `train_sh/<expert>/`，脚本不能把自身目录误当成 `old_verl_grpo` 根目录。每个 v1/v2/v3 脚本都应统一使用：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
OLD_VERL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
```

所有公共路径都从 `OLD_VERL_DIR` 构造：

```bash
TRAINING_VARIANT="v3"
LOG_DIR="${OLD_VERL_DIR}/log/${EXPERT}/${TRAINING_VARIANT}"
TOOL_CONFIG_PATH="${OLD_VERL_DIR}/config/tool_config/restoration_tool_config_marginal_efficiency_2gpu.yaml"

export OLD_VERL_CONFIG_NAME="${EXPERT}_v3_config_2gpu"
export OLD_VERL_EXPERIMENT_NAME="${EXPERT}_${TRAINING_VARIANT}"
export OLD_VERL_OUTPUT_DIR="${OLD_VERL_DIR}/outputs/${EXPERT}_${TRAINING_VARIANT}"
export OLD_VERL_LOG_DIR="${LOG_DIR}"
```

调用公共启动器时使用：

```bash
exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@"
```

不建议继续使用含义较窄的 `REWARD_VARIANT`，因为 v3 的差异不仅是奖励。三个版本统一改用：

```bash
TRAINING_VARIANT="v1"  # 或 v2/v3
```

### 2.7 日志、输出和断点隔离

建议使用以下目录：

```text
old_verl_grpo/
├── log/
│   ├── fog/
│   │   ├── v1/
│   │   ├── v2/
│   │   └── v3/
│   ├── low_light/{v1,v2,v3}/
│   ├── rain/{v1,v2,v3}/
│   └── snow/{v1,v2,v3}/
└── outputs/
    ├── fog_v1/
    ├── fog_v2/
    ├── fog_v3/
    ├── low_light_v1/
    ├── low_light_v2/
    ├── low_light_v3/
    ├── rain_v1/
    ├── rain_v2/
    ├── rain_v3/
    ├── snow_v1/
    ├── snow_v2/
    └── snow_v3/
```

每个版本目录分别维护：

- 主训练日志；
- `restoration_tool_info.log`；
- `restoration_tools.log`；
- SwanLab 本地目录；
- checkpoint；
- penalized samples。

公共启动器需要支持：

```bash
LOG_DIR="${OLD_VERL_LOG_DIR:-${LOG_ROOT}/${EXPERT}}"
```

从而由 v1/v2/v3 包装脚本将日志定向到各自版本目录。已有历史日志不迁移，新启动的实验使用新目录。

### 2.8 preflight 版本一致性检查

公共启动器的 preflight 应接收 `OLD_VERL_TRAINING_VARIANT`，并检查：

| 版本 | 预期 reward mode | 预期 `entropy_coeff` | 预期 `tool_action_entropy_coeff` |
|---|---|---:|---:|
| v1 | `step_mixed_v1` | `0.005` | `0.0` |
| v2 | `marginal_efficiency_v1` | `0.005` | `0.0` |
| v3 | `marginal_efficiency_v1` | `0.005` | 大于 `0.0` |

还应检查：

1. v3 的配置名以 `_v3_config_2gpu` 结尾。
2. `trainer.experiment_name` 以 `_v3` 结尾。
3. output、SwanLab 和日志目录都包含 `v3`。
4. v3 使用边际效率奖励配置。
5. `calculate_entropy=true`。
6. v1/v2 的 `tool_action_entropy_coeff` 必须为零。

如果任何一项不一致，启动器在创建数据、启动 Ray 或加载模型前失败。

## 3. 背景与问题

当前配置：

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.005
    calculate_entropy: true
```

现有 actor loss 使用完整 `response_mask` 聚合 entropy，因此自然语言分析、`<tool_call>` XML 标签、固定函数名 `restore_image`、参数名 `action` 和 action 值都会获得同样的熵奖励。

这存在两个问题：

1. 大部分熵奖励消耗在自然语言和固定格式 token 上，不能直接维持修复工具选择的探索性。
2. 如果直接提高全局 `entropy_coeff`，可能增加冗余推理、格式错误和无效输出，而不一定提高工具分布多样性。

第一版改造将新增一套独立 mask，只在合法 `restore_image` 调用的 action 值 token 上计算额外 entropy bonus。

## 4. 设计目标

### 4.1 功能目标

1. 精确标记合法 action 值对应的原始生成 token。
2. 不标记自然语言分析、XML 标签、函数名、参数名、工具反馈和 padding。
3. 支持多轮轨迹和多 token action，例如：
   - `scunet`；
   - `retinexformer_fivek`；
   - `mb_taylorformer_dehaze`；
   - `stop`。
4. action 必须同时满足：
   - XML 结构合法；
   - 函数名为 `restore_image`；
   - 参数名为 `action`；
   - 参数值属于当前回合实际开放的 schema enum；
   - tokenizer 重编码结果与原始 `response_ids` 完全一致。
5. 匹配失败时采用 fail-closed 策略：该回合 mask 全零，并记录失败原因，不允许模糊匹配。
6. 新增熵项必须通过独立系数启停，默认值为 `0.0`，保证旧训练行为不变。

### 4.2 优化目标

1. 工具调用次数增加不能自动带来更大的工具熵奖励。
2. 不因某个 action 被 tokenizer 切成更多 token 就显著放大整条轨迹的奖励。
3. 无合法工具调用的轨迹不能获得工具 action 熵奖励。
4. 与现有 GRPO、KL loss 和全 response entropy 兼容。

### 4.3 非目标

第一版不处理以下内容：

1. 不修改 IQA 奖励和工具调用成本。
2. 不改变 `prompt_min_stop_tool_calls`。
3. 不对工具调用进行 constrained decoding。
4. 不计算合法 action 集合上的严格 categorical entropy。
5. 不为 action 名增加专用词表 token。
6. 不从训练日志反向推断 token mask。

## 5. 熵正则定义

设：

- $B$ 为当前全局 PPO batch 中的轨迹数量；
- $H_{b,t}$ 为轨迹 $b$ 的第 $t$ 个 response token 位置上的词表熵；
- $m_{b,t}\in\{0,1\}$ 为 `tool_action_token_mask`；
- $n_b=\sum_t m_{b,t}$ 为轨迹 $b$ 中合法 action token 的数量。

定义：

$$
H_{\text{tool-action}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{1}[n_b>0]
\frac{\sum_t m_{b,t}H_{b,t}}{n_b}.
$$

actor 总损失改为：

$$
\mathcal L_{\text{actor}}
=
\mathcal L_{\text{PG}}
+\beta_{\text{KL}}\mathcal L_{\text{KL}}
-\beta_{\text{response}}H_{\text{response}}
-\beta_{\text{tool}}H_{\text{tool-action}}.
$$

其中：

- `entropy_coeff` 对应 $\beta_{\text{response}}$，保留现有语义；
- 新增 `tool_action_entropy_coeff` 对应 $\beta_{\text{tool}}$；
- `tool_action_entropy_coeff=0.0` 时完全关闭新熵项。

采用“轨迹内 token 均值、全局 batch 均值”的原因：

1. 一条轨迹调用六次工具不会仅因为 mask token 更多而获得六倍熵奖励。
2. 工具熵不会主动抵消 v2 奖励中的 `0.12` 工具调用成本。
3. 没有合法 action 的轨迹贡献零，格式错误或完全不调用工具不会得到该项奖励。
4. 可以复用 VERL 的 `seq-mean-token-mean` 聚合模式及已有的全局 batch 归一化。

## 6. action token 的严格匹配

### 6.1 只标记 action 值

对于：

```xml
<tool_call>
<function=restore_image>
<parameter=action>
scunet
</parameter>
</function>
</tool_call>
```

只标记 `scunet` 对应的 token。以下内容全部为零：

- `<tool_call>`；
- `restore_image`；
- `<parameter=action>`；
- action 前后的换行和空白；
- `</parameter>` 等结束标签。

`restore_image` 是固定函数名，不代表不同图像修复操作之间的选择，不能纳入工具选择熵。

### 6.2 匹配输入

匹配必须使用 SGLang 在当前 assistant turn 返回的原始 `response_ids`。禁止使用：

- console 日志中的解码文本；
- SwanLab 展示文本；
- penalized sample 中经过处理的 `raw_text`；
- 拼接完整轨迹后再猜测 action 出现位置。

原因是同一个 action 名可能出现在自然语言分析中，例如：

```text
RIDCP did not improve the image, so I will use SCUNet.
...
<parameter=action>
scunet
</parameter>
```

简单搜索 `scunet` 会误标自然语言中的同名文本。

### 6.3 匹配步骤

建议在 `ToolAgentLoop._handle_generating_state()` 中完成以下步骤：

1. 对本轮 `response_ids` 应用现有格式 guardrail。
2. 使用现有 `Qwen3XMLToolParser` 解析工具调用。
3. 要求本轮恰好解析出一个调用。
4. 要求 `FunctionCall.name == "restore_image"`。
5. 解析 `FunctionCall.arguments`，要求仅存在合法 `action` 值。
6. 从当前回合的 active tool schema 中读取 action enum。
7. 解码完整本轮 response，保留 special token：

   ```python
   text = tokenizer.decode(response_ids, skip_special_tokens=False)
   ```

8. 使用结构边界定位唯一的 action 原始字符区间：

   ```regex
   <parameter=action>(?P<raw_value>.*?)</parameter>
   ```

9. 去除 `raw_value` 两端空白，得到 action 字符区间，不得搜索全局同名字符串。
10. 使用 fast tokenizer 对完整文本重编码：

    ```python
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ```

11. 严格要求：

    ```python
    encoding["input_ids"] == response_ids
    ```

12. 选择 offset 完全包含在 action 字符区间中的 token。
13. 如果任何 token 横跨 action 与空白/XML 边界，匹配失败。
14. 对选中的连续 token 再次 decode，要求等于 action 原值。
15. 要求至少选中一个 token。

### 6.4 建议的 helper 返回值

建议新增纯函数，方便 CPU 单元测试：

```python
@dataclass(frozen=True)
class ToolActionMaskResult:
    mask: list[int]
    matched: bool
    action: str | None
    failure_reason: str | None


def build_tool_action_token_mask(
    response_ids: list[int],
    tokenizer: Any,
    parsed_tool_calls: list[FunctionCall],
    allowed_actions: set[str],
) -> ToolActionMaskResult:
    ...
```

建议的失败原因枚举：

- `no_parsed_tool_call`；
- `multiple_tool_calls`；
- `unexpected_function_name`；
- `missing_action_argument`；
- `action_not_in_active_schema`；
- `missing_action_xml_span`；
- `multiple_action_xml_spans`；
- `parsed_action_text_mismatch`；
- `tokenizer_not_fast`；
- `token_id_roundtrip_mismatch`；
- `token_crosses_action_boundary`；
- `empty_action_token_span`；
- `decoded_action_token_mismatch`。

不允许在严格匹配失败后退化为 `encode(action)` 并在整段 response 中搜索，因为这会重新引入同名 action 出现在自然语言中的误标风险。

## 7. 多轮 mask 的构造与传递

### 7.1 `AgentData` 中的累计 mask

在 `ToolAgentLoop` 的 `AgentData` 中新增：

```python
self.tool_action_token_mask: list[int] = []
```

必须始终满足：

```python
len(agent_data.tool_action_token_mask) == len(agent_data.response_mask)
```

模型生成本轮 response 后：

```python
agent_data.response_mask += [1] * len(turn_response_ids)
agent_data.tool_action_token_mask += turn_action_mask
```

插入工具 observation、重构 prompt 或其他零损失内容时：

```python
agent_data.response_mask += [0] * len(observation_ids)
agent_data.tool_action_token_mask += [0] * len(observation_ids)
```

### 7.2 最终轨迹示例

```text
轨迹区段：              assistant 1 | tool result 1 | assistant 2 | padding
response_mask：         11111111111 | 0000000000000 | 11111111111 | 000000
tool_action_token_mask：00000111000 | 0000000000000 | 00000011000 | 000000
```

### 7.3 `AgentLoopOutput` 数据字段

新增：

```python
tool_action_token_mask: list[int] | None = None
```

兼容性策略：

1. `ToolAgentLoop` 返回实际 mask。
2. 其他 agent loop 未提供该字段时，统一补成与 `response_ids` 等长的全零 mask。
3. `_InternalAgentLoopOutput` 使用 padding 后的 `torch.Tensor`。
4. `_postprocess()` 始终在 `TensorDict` 中输出：

   ```python
   "tool_action_token_mask": tool_action_token_mask
   ```

这样 loss 层在启用新系数时不需要判断 batch 是否来自某种 agent loop。

## 8. loss 接入方案

### 8.1 配置字段

在 `ActorConfig` 和 actor YAML 中新增：

```yaml
# Entropy regularization coefficient applied only to valid tool action value tokens.
tool_action_entropy_coeff: 0.0
```

配置约束：

1. `tool_action_entropy_coeff >= 0.0`。
2. 当 `tool_action_entropy_coeff > 0.0` 时，`calculate_entropy` 必须为 `true`。
3. 默认值必须为 `0.0`，避免改变其他 VERL 示例行为。

旧版图像修复共享配置后续可显式设置：

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.005
    tool_action_entropy_coeff: 0.001
    calculate_entropy: true
```

上述 `0.001` 仅作为首轮 smoke/短程标定起点，不应在没有观察 loss 比例前直接作为最终训练值。

### 8.2 loss 计算

在 `ppo_loss()` 的字段列表中加入：

```python
fields = [
    "response_mask",
    "tool_action_token_mask",
    "old_log_probs",
    "advantages",
]
```

构造有效 mask：

```python
tool_action_mask = (
    data["tool_action_token_mask"].to(bool)
    & data["response_mask"].to(bool)
)
```

交集操作是第二道防线，防止工具 observation 或 padding 因上游 bug 被错误标记。

新增聚合：

```python
tool_action_entropy = agg_loss(
    loss_mat=entropy,
    loss_mask=tool_action_mask,
    loss_agg_mode="seq-mean-token-mean",
    **config.global_batch_info,
)

policy_loss -= config.tool_action_entropy_coeff * tool_action_entropy
```

注意：

1. 不应复用当前 actor 的 `loss_agg_mode="token-mean"`。
2. 如果使用全局 action token 数做 token mean，调用更多工具会产生更多 entropy bonus，与减少调用次数的奖励目标冲突。
3. 使用 `seq-mean-token-mean` 时，每条轨迹的工具熵贡献上限不随工具调用次数线性增长。
4. `global_batch_size` 继续使用现有 PPO 全局 batch 大小；无合法 action 的轨迹贡献零。

### 8.3 零 mask 安全

`seq-mean-token-mean` 已通过 `1e-8` 防止单条序列除零，但仍需测试：

- 整个 batch 的 mask 全零时，`tool_action_entropy == 0`；
- loss 不是 NaN/Inf；
- backward 正常；
- 记录 warning 或 coverage 指标，但不终止训练。

## 9. 需要修改的文件

### 9.1 版本配置、启动脚本与公共启动器

1. `examples/image_restoration_multi_agent/old_verl_grpo/config/restoration_common_config_2gpu.yaml`
   - 显式增加 `tool_action_entropy_coeff: 0.0`；
   - 保证 v1/v2 在底层代码升级后仍保持原行为。

2. 四个 v3 专家配置：
   - `config/fog_v3_config_2gpu.yaml`；
   - `config/low_light_v3_config_2gpu.yaml`；
   - `config/rain_v3_config_2gpu.yaml`；
   - `config/snow_v3_config_2gpu.yaml`。
   - 分别继承现有专家配置；
   - 使用 v2 的边际效率奖励配置；
   - 显式设置 `entropy_coeff`、`tool_action_entropy_coeff` 和 `calculate_entropy`；
   - 使用独立 v3 实验名和输出目录。

3. 四个 v3 启动脚本：
   - `train_sh/fog/fog_v3.sh`；
   - `train_sh/low_light/low_light_v3.sh`；
   - `train_sh/rain/rain_v3.sh`；
   - `train_sh/snow/snow_v3.sh`。
   - 一键选择 v3 专家配置；
   - 写入独立日志、输出和 SwanLab 目录；
   - 将 `OLD_VERL_TRAINING_VARIANT=v3` 传给公共启动器。

4. 现有八个 v1/v2 启动脚本
   - 统一从 `train_sh/<expert>/` 正确解析 `OLD_VERL_DIR`；
   - 将 `REWARD_VARIANT` 改为 `TRAINING_VARIANT`；
   - 分别设置 `OLD_VERL_TRAINING_VARIANT=v1` 或 `v2`；
   - 日志迁移到 `log/<expert>/<variant>/`；
   - 保持各自奖励配置不变。

5. `examples/image_restoration_multi_agent/old_verl_grpo/run_expert_old_verl_grpo_2gpu.sh`
   - 支持 `OLD_VERL_LOG_DIR`；
   - 接收并打印 `OLD_VERL_TRAINING_VARIANT`；
   - preflight 校验版本、奖励模式、配置名、两个熵系数及输出路径；
   - 防止跨版本自动恢复 checkpoint。

6. `examples/image_restoration_multi_agent/old_verl_grpo/run_four_experts_serial_old_verl_grpo_2gpu.sh`
   - 如继续保留串行四专家入口，应增加 `v1|v2|v3` 参数；
   - 根据版本调用 `train_sh/<expert>/<expert>_<variant>.sh`；
   - 默认版本必须显式记录，禁止静默选择。

### 9.2 运行时与 mask

7. `examples/image_restoration_multi_agent/verl_backend/verl/experimental/agent_loop/tool_agent_loop.py`
   - 新增 `ToolActionMaskResult`；
   - 新增严格 action mask helper；
   - `AgentData` 累积 mask；
   - assistant response、tool observation 两条路径同步扩展 mask；
   - 记录匹配成功率和失败原因。

8. `examples/image_restoration_multi_agent/verl_backend/verl/experimental/agent_loop/agent_loop.py`
   - 扩展 `AgentLoopOutput`；
   - 扩展 `_InternalAgentLoopOutput`；
   - padding `tool_action_token_mask`；
   - `_postprocess()` 写入 batch；
   - 非工具 agent 默认输出全零 mask。

### 9.3 VERL 配置与 loss

9. `examples/image_restoration_multi_agent/verl_backend/verl/workers/config/actor.py`
   - 新增 `tool_action_entropy_coeff: float = 0.0`；
   - 添加非负校验及文档。

10. `examples/image_restoration_multi_agent/verl_backend/verl/trainer/config/actor/actor.yaml`
   - 增加配置项和说明。

11. VERL 生成的 trainer 配置
   - 按项目现有配置生成流程同步刷新 `_generated_ppo_trainer.yaml` 等受影响文件；
   - 不手工遗漏生成配置中的 actor 字段；
   - 运行 config sanity tests 确认配置文档一致。

12. `examples/image_restoration_multi_agent/verl_backend/verl/workers/utils/losses.py`
   - 从 batch 读取新 mask；
   - 计算 `tool_action_token_entropy`；
   - 加入 actor loss；
   - 增加训练指标。

v3 专家配置应在 mask 和 loss 验证后启用非零系数。初始实验建议保留全局 `entropy_coeff=0.005`，新增项先从较小系数开始；如需验证纯工具熵效果，再通过明确的消融覆盖设置 `entropy_coeff=0.0`。

## 10. 监控指标

### 10.1 actor loss 指标

新增：

- `actor/tool_action_token_entropy`：按本文公式聚合的工具 action token 熵；
- `actor/tool_action_entropy_bonus`：`tool_action_entropy_coeff * tool_action_token_entropy`；
- `actor/tool_action_entropy_coeff`：实际生效系数；
- `actor/response_entropy_loss`：建议将现有 `actor/entropy_loss` 保留或补充更明确别名。

### 10.2 mask 质量指标

新增：

- `tool_action_mask/matched_turns`；
- `tool_action_mask/attempted_turns`；
- `tool_action_mask/match_rate`；
- `tool_action_mask/tokens_per_trajectory_mean`；
- `tool_action_mask/trajectories_with_action_rate`；
- `tool_action_mask/roundtrip_failure_rate`；
- `tool_action_mask/invalid_schema_action_rate`；
- `tool_action_mask/boundary_failure_rate`。

失败原因应作为计数器记录，不要为每个失败样本输出完整 INFO 日志，以免训练日志膨胀。首次异常可用 WARNING，逐样本细节使用 DEBUG。

### 10.3 训练效果指标

继续联合观察：

- `timing_s/agent_loop/tool_calls/mean`；
- `num_turns/mean`；
- action 分布；
- `actor/action_path_entropy`；
- `actor/tool_choice_entropy`；
- invalid action / malformed XML 比例；
- 无工具轨迹比例；
- 最终 IQA 增益；
- 平均工具调用成本；
- v2 轨迹总奖励。

不能仅凭 `actor/tool_action_token_entropy` 上升就判断探索性改善，因为该指标仍是 action 位置上的全词表熵。

其中两个经验分布熵按当前 actor 训练 batch 计算，均使用自然对数，单位为 nat：

1. `actor/action_path_entropy` 将每条非空轨迹的完整 action 序列视为一个类别，例如
   `("ridcp", "scunet", "stop")`。
2. `actor/tool_choice_entropy` 汇总所有非空轨迹中实际执行的 action；`stop` 也作为合法 action 参与统计。
3. 空 action 轨迹不参与两个熵的概率分布，另以
   `actor/action_path_valid_trajectory_rate` 记录非空轨迹覆盖率。
4. 同时记录路径类别数、action 类别数和 action 样本数，避免仅看熵值时忽略有效样本规模。

## 11. 测试方案

### 11.1 action mask 单元测试

建议新增：

```text
examples/image_restoration_multi_agent/verl_backend/tests/experimental/agent_loop/
test_tool_action_token_mask_on_cpu.py
```

至少覆盖：

1. `scunet` 被切成多个 token，全部且仅 action token 为 1。
2. `retinexformer_fivek` 等长 action 正确匹配。
3. 自然语言中先出现 `scunet`，只标记 XML action 值。
4. 多轮分别生成不同 action，累计 mask 与 `response_mask` 等长。
5. 工具 observation 和 padding 全零。
6. 合法 `stop` 在 schema 开放时被标记。
7. `stop` 未出现在当前 schema 时不标记。
8. 非法 action 不标记。
9. 函数名不是 `restore_image` 时不标记。
10. XML 缺少结束标签时不标记。
11. 同一回合多个 action 参数时不标记。
12. 同一回合多个 tool call 时不标记。
13. tokenizer roundtrip 不一致时不标记。
14. token 横跨 action/XML 边界时不标记。
15. action 前后存在多个空格或换行时仍能得到正确字符区间。
16. special EOS token 被 guardrail 裁剪后 mask 长度一致。

测试应使用当前 Qwen3.5 tokenizer 的代表性 tokenization fixture；同时为失败分支提供轻量 fake tokenizer，避免所有测试都依赖本地模型目录。

### 11.2 数据传递测试

扩展 agent loop CPU 测试，断言：

```python
assert output.tool_action_token_mask.shape == output.response_mask.shape
assert (output.tool_action_token_mask <= output.response_mask).all()
```

并验证：

- padding 前后 action token 的相对位置不变；
- `_postprocess()` 后 batch 中字段存在；
- 多个不同长度轨迹拼接正常；
- 非工具 agent 获得全零 mask。

### 11.3 loss 单元测试

为 `ppo_loss()` 增加以下测试：

1. `tool_action_entropy_coeff=0.0` 时 loss 与改造前完全一致。
2. 系数大于零时：

   ```text
   policy_loss_new =
   policy_loss_old - coeff * expected_tool_action_entropy
   ```

3. mask 全零时工具熵和 bonus 均为零。
4. action mask 与 response mask 取交集。
5. 相同逐 token entropy 下，增加工具调用次数但不改变每轨迹平均 entropy。
6. 两条轨迹中一条无合法 action 时，无 action 轨迹贡献零。
7. 多 DP 归一化结果与单进程等价。
8. backward 后梯度有限且无 NaN/Inf。

### 11.4 配置测试

扩展：

```text
examples/image_restoration_multi_agent/verl_backend/tests/workers/config/
test_actor_config_on_cpu.py
```

覆盖：

- 默认系数为 `0.0`；
- 负系数被拒绝；
- 正系数与 `calculate_entropy=false` 的组合被拒绝或给出明确错误；
- YAML 能正确转换为 `ActorConfig`；
- 生成配置和配置文档检查通过。

另外增加 v1/v2/v3 组合测试，逐专家验证：

| 启动脚本 | 最终专家配置 | 奖励配置 | `tool_action_entropy_coeff` |
|---|---|---|---:|
| `fog_v1.sh` | `fog_config_2gpu` | current IQA / `step_mixed_v1` | `0.0` |
| `fog_v2.sh` | `fog_config_2gpu` | marginal efficiency | `0.0` |
| `fog_v3.sh` | `fog_v3_config_2gpu` | marginal efficiency | 大于 `0.0` |

`low_light`、`rain` 和 `snow` 执行相同矩阵。测试还必须断言三个版本的 experiment、output、SwanLab 和 log 路径互不相同。

### 11.5 建议验证命令

在 `examples/image_restoration_multi_agent/verl_backend` 下执行：

```bash
uv run --no-sync pytest -v \
  tests/experimental/agent_loop/test_tool_action_token_mask_on_cpu.py \
  tests/experimental/agent_loop/test_tool_agent_loop_restoration_penalties_on_cpu.py \
  tests/workers/config/test_actor_config_on_cpu.py \
  tests/utils/test_padding_on_cpu.py
```

然后执行仓库级静态检查：

```bash
uv run --no-sync pyright
uv run --no-sync pre-commit run --all-files --show-diff-on-failure
```

最后通过旧版启动器进行：

1. 对 12 个 v1/v2/v3 专家脚本分别执行 `--preflight`；
2. 确认 v1/v2 的新熵系数为零，v3 为非零；
3. 确认三个版本的奖励模式符合版本矩阵；
4. 对一个专家执行一步 v3 `--smoke` 训练；
5. 检查 SwanLab 和 console 是否出现新指标；
6. 人工抽查若干轨迹的 token、offset 和 mask 可视化；
7. 确认 v3 没有读取同专家 v1/v2 的 checkpoint 或日志。

## 12. 分阶段实施

### 阶段 0：版本入口和隔离

实施状态：已完成。

1. 修正现有 v1/v2 脚本从嵌套目录解析公共路径的方式。
2. 增加四个 v3 专家配置和四个 v3 启动脚本。
3. 将训练日志和工具日志隔离到 `log/<expert>/<variant>/`。
4. 扩展公共启动器的版本 preflight。
5. 对全部 12 个脚本运行 preflight。

在阶段 0 结束时，v3 的系数仍保持 `0.0`，先验证版本路由和目录隔离，避免将启动错误与 loss 实现错误混在一起。

### 阶段 A：仅增加 mask 与观测

实施状态：已完成。

1. 完成严格 action mask。
2. 将 mask 传入 batch。
3. 新系数保持 `0.0`。
4. 记录匹配率、失败原因和离线 entropy 指标。
5. 要求正常轨迹的 mask 匹配率接近 100%，所有失败都有明确原因。

阶段 A 不改变训练梯度，是风险最低的验证阶段。

### 阶段 B：接入 loss，系数仍为零

实施状态：已完成。v1/v2 仍显式使用 `tool_action_entropy_coeff=0.0`，其 loss 数值兼容性已通过单元测试。

1. 完成 actor config 和 loss 路径。
2. 验证 `tool_action_entropy_coeff=0.0` 与基线 loss 数值一致。
3. 验证断点加载和旧配置兼容。

### 阶段 C：短程系数标定

实施状态：已按用户要求配置首轮参数，尚未启动实际训练：

```yaml
entropy_coeff: 0.005
tool_action_entropy_coeff: 0.001
```

建议先使用：

```yaml
entropy_coeff: 0.005
tool_action_entropy_coeff: 0.001
```

进行短程训练，重点观察：

1. `tool_action_entropy_bonus` 与 `pg_loss`、`kl_loss` 的数量级；
2. invalid action / malformed XML 是否上升；
3. 工具调用次数是否异常增加；
4. action 分布是否从后期塌缩状态恢复；
5. IQA 增益是否下降。

如果工具熵项过弱，可按 `0.001 -> 0.002 -> 0.005` 逐步扩大；如果格式错误或低质量工具探索明显增加，应降低系数，而不是继续提高。

### 阶段 D：消融实验

至少比较：

| 实验 | `entropy_coeff` | `tool_action_entropy_coeff` | 目的 |
|---|---:|---:|---|
| Baseline | 0.005 | 0.0 | 当前全 response entropy |
| Targeted add-on | 0.005 | 标定值 | 在当前方案上增加定向探索 |
| Tool-only | 0.0 | 标定值 | 判断自然语言/XML 熵是否必要 |
| No entropy | 0.0 | 0.0 | 判断两种熵正则的总体贡献 |

四组实验必须使用相同：

- SFT 初始 LoRA；
- 数据顺序和随机种子；
- GRPO rollout 数；
- IQA 奖励版本；
- `prompt_min_stop_tool_calls`；
- 训练步数。

## 13. 风险与应对

### 13.1 全词表熵可能鼓励非法 action

action token 位置上的 entropy 对整个词表计算，理论上可能把概率质量分配给非法字符串。

应对：

1. 从小系数开始；
2. 保留 schema、格式惩罚和 KL 约束；
3. 监控 invalid action 和 malformed XML；
4. 后续如仍有问题，再升级为合法 action 候选集合上的序列熵。

### 13.2 tokenizer decode/encode 不完全可逆

特殊 token、规范化或异常字节可能导致 roundtrip 不一致。

应对：

1. 严格校验；
2. 失败时全零；
3. 记录 failure reason；
4. 不使用近似搜索作为 fallback。

### 13.3 新熵项间接鼓励长轨迹

如果使用全局 action token sum，更多工具调用会带来更大 bonus。

应对：

- 固定使用 `seq-mean-token-mean`；
- 测试“增加调用次数不增加常量 entropy 下的轨迹 bonus”；
- 不允许该项复用普通 `token-mean` 聚合。

### 13.4 与现有全 response entropy 重复

action token 已经包含在 `response_mask` 中，因此新增项会在这些位置叠加权重。

应对：

1. 明确这是定向加权，不是互斥 mask；
2. 首轮新增系数显著小于或等于现有系数；
3. 通过 Tool-only 消融判断是否需要保留全 response entropy；
4. 记录两个 entropy bonus 的独立数值。

### 13.5 action 名的 token 数不同

不同 action 的 BPE 长度不同。轨迹内 token mean 能消除总 token 数的线性放大，但 action 内部仍按 token 决策计算，不是每个 action 严格等权。

应对：

- 第一版接受这一近似；
- 报告中使用 `tool_action_token_entropy` 名称；
- 不将其直接解释为严格的 `tool_choice_entropy`；
- 如消融显示 tokenization 偏差显著，再实施合法 action 序列概率方案。

## 14. 验收标准

实施完成需同时满足：

1. 默认配置不改变任何现有训练 loss。
2. `tool_action_token_mask.shape == response_mask.shape`。
3. `tool_action_token_mask <= response_mask` 对所有元素成立。
4. 工具反馈、padding、自然语言和 XML token 均不被标记。
5. 合法 action 的所有且仅对应 token 被标记。
6. 匹配失败时不进行模糊 fallback。
7. mask 全零时 loss 无 NaN/Inf。
8. 工具调用次数增加不会在线性意义上增加每轨迹工具熵 bonus。
9. 新旧 entropy 指标可在 console 和 SwanLab 中分别观察。
10. CPU 单元测试、配置测试、静态检查和一步 smoke 训练通过。
11. 基线配置、定向叠加配置和 Tool-only 配置均可通过启动脚本覆盖参数切换。
12. `train_sh` 下四个专家都具有 v1、v2、v3 三个可执行脚本。
13. v1、v2、v3 的奖励模式和熵系数严格符合版本矩阵。
14. 三个版本的输出、日志、SwanLab 和 checkpoint 路径互不覆盖。
15. v3 一键启动不需要用户手工传入奖励或熵相关 Hydra 参数。
16. v1/v2 在相同输入和随机种子下与加入底层新代码前保持数值兼容。

## 15. 后续升级方向

如果第一版证明 action token 定向熵可以改善探索，但非法 action 概率也随之上升，可考虑第二版严格工具选择熵：

1. 为每个合法 action 计算完整多 token 序列条件 log probability；
2. 在当前 schema 开放的 action 集合中归一化；
3. 计算 action categorical entropy；
4. 对共享 token 前缀使用 trie 和 batched prefix forward 降低计算量；
5. 对 `stop` 的开放状态按每回合动态 schema 处理。

该方案更接近真正的 `tool_choice_entropy`，但需要为每个工具决策点评分多个候选序列，计算和显存开销明显高于第一版，因此应在第一版完成消融后再决定是否实施。
