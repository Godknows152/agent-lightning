# ALFWorld baseline（Qwen3.5 v5 文本动作 / Qwen2.5 工具协议）

本目录隔离 ALFWorld 文本环境与 old-VERL baseline。Qwen2.5-1.5B、Qwen3.5-2B 与 Qwen3.5-9B 的模型、parser、提示词、parquet、Hydra 入口、启动脚本、日志、checkpoint 和 SwanLab 实验名均按 profile 分离；当前默认 profile 为 `qwen35_2b`。公共的 ALFWorld 环境、奖励和 Validator 保持共用，模型专属的运行时/性能覆盖保存在各自的 Hydra 配置中。当前 Qwen3.5-2B 使用独立的8个 AgentLoop worker、环境池、关闭 FSDP offload/梯度检查点和更大的 PPO batch；Qwen3.5-9B 与 Qwen2.5-1.5B 不受本次优化影响。详情见 `MODEL_PROFILES.md`。ALFWorld 使用隔离的 `alfworld_tool_agent`；图像修复继续使用共享的 `tool_agent`，不修改其行为。

## Qwen2.5 工具提示词

当前数据集和运行时提示词版本为 `alfworld_qwen25_json_strict_v1`（定义在
`src/alfworld_baseline/prompts.py`）。它在 Qwen 原生 chat template 的通用工具说明之后再次
声明任务专用协议，明确覆盖“调用前可输出 reasoning”“无工具时可正常回答”等通用分支：

- 每个 assistant turn 必须调用且只能调用一次 `alfworld_action`；
- 首字符必须是 `<tool_call>`，末字符必须是 `</tool_call>`；
- 工具块内只能有一个 JSON 对象，`name` 为 `alfworld_action` 且 `arguments` 只含 `action`；
- `action` 必须从当前 admissible 列表逐字复制，禁止同义词、裸文本、Markdown、
  `<think>` 及工具块前后缀；
- `data.apply_chat_template_kwargs.enable_thinking=false`，避免模板自动开启可见思考块。

Qwen tokenizer 可能在返回 token 中追加 `<|im_end|>` 或 `<|endoftext|>`；它们是传输层 EOS/padding，
不是模型可见回答。`scripts/verify_tool_protocol.py` 会单独剥离这些标记，再分别报告严格工具块、
Parser 可解析和 Validator 动作合法率；普通前缀/后缀不会被剥离，而会保留为格式错误并触发单步惩罚。

验证命令（只加载本地模型，不启动 Ray/GRPO）：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src:examples/image_restoration_multi_agent/verl_backend \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python \
  examples/alfworld_grpo_baseline/scripts/verify_tool_protocol.py \
  --samples 32 --max-new-tokens 128 --device cuda:0
```

结果写入 `outputs/diagnostics/tool_protocol.json`，其中 `strict_xml_rate` 统计严格 Qwen 工具块；
`validation_status_counts` 另外区分动作拼写错误（例如把列表中的 `go to fridge 1` 改成
`go to refrigerator 1`）。

切换模型后应重新运行诊断；该抽样只作为格式回归证据，不等同于对任意随机采样或正式训练
成功率的保证。

训练文件已按图像修复 old-VERL 的约定规范化：共享参数位于 `config/alfworld_common_config_2gpu.yaml`，模型参数位于 `config/model_profiles/`，模型入口位于 `config/alfworld/<profile>/v1/`，启动脚本位于 `scripts/alfworld/`。checkpoint、rollout、主日志和 SwanLab 数据分别写入 `outputs/alfworld/<profile>/v1/2gpu/` 与 `log/alfworld/<profile>/v1/2gpu/`。完整目录规范见 `TRAINING_LAYOUT.md`。

SwanLab 正式训练默认启用 cloud 模式，项目名为 `ALFWorldRL`，实验名按 `alfworld_<profile>_v1_seedN` 命名；smoke/pilot 默认使用 offline，避免测试 run 污染云端正式实验。可用 `ALFWORLD_SWANLAB_MODE=cloud` 显式上传 pilot。

```bash
# Qwen2.5-1.5B：仅预检，不占 GPU
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen25_1_5b_v1.sh --preflight

# Qwen2.5-1.5B：5 步 pilot，后台运行
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen25_1_5b_v1.sh --pilot

# Qwen3.5-2B：seed 0 正式训练，后台运行并上传 SwanLab（当前默认）
SEED=0 bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh
```

## 当前验证命令

```bash
# ALFWorld 环境（不启动 Ray/GPU）
conda run -n alfworld-verl python scripts/preflight_alfworld.py
conda run -n alfworld-verl python scripts/smoke_test.py --steps 3

# 组件、Qwen tokenizer 与 old-VERL/ALFWorld 统一环境
PYTHONPATH=examples/alfworld_grpo_baseline/src:examples/image_restoration_multi_agent/verl_backend \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest -q examples/alfworld_grpo_baseline/tests
/home/LXJ/anaconda3/envs/alfworld-verl/bin/python examples/alfworld_grpo_baseline/scripts/preflight.py
```

`alfworld-verl` 是从 `verl` 克隆并补齐 ALFWorld 依赖的统一环境。`preflight.py` 检查正式 VERL runtime 所需的 `alfworld`、`gymnasium`、`stable_baselines3`、`transformers`、`pandas`、`pyarrow` 和 `omegaconf`，要求 Qwen2.5 tokenizer 存在原生 chat template，并要求隔离目录下已存在 `data/train.parquet` 与 `data/test.parquet`；`preflight_alfworld.py` 只检查独立 ALFWorld 数据环境。原 `verl` 和 `alfworld` 环境保持不变。

先生成 old-VERL 数据（该步骤会为每条任务 reset 一次，以构造首轮 observation 和动态 admissible actions；脚本按 64 个游戏分块加载，避免逐条重复初始化 TextWorld）：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src \
  conda run -n alfworld-verl python examples/alfworld_grpo_baseline/scripts/prepare_verl_dataset.py \
  --split train --limit 64 --output examples/alfworld_grpo_baseline/data/train.parquet
PYTHONPATH=examples/alfworld_grpo_baseline/src \
  conda run -n alfworld-verl python examples/alfworld_grpo_baseline/scripts/prepare_verl_dataset.py \
  --split test --limit 32 --output examples/alfworld_grpo_baseline/data/test.parquet
```

生成的数据包含 `extra_info.tools_kwargs.alfworld_action.create_kwargs.game_file`，供 old-VERL
工具实例创建时加载对应游戏；训练配置在 pilot 前需将 `variables.DATA_DIR` 覆盖为该目录。

环境接受的最终动作始终是 ALFWorld 原生文本字符串，例如 `go to cabinet 1`，不是完整 JSON/XML，也不是 `alfworld_action` 函数名。

该 baseline 是多步轨迹级 GRPO：一次合法工具调用对应一次环境 `step`，随后把新的 observation 和 admissible actions 回传给模型继续决策。`single prompt` 只描述每轮 prompt 组织方式。

## Qwen3.5 v5：历史动作 + 文本动作 + thinking（2026-09-10）

提示词版本：`alfworld_qwen35_v5_action_history_text_thinking`，同时应用于 2B/9B。
定义在 `src/alfworld_baseline/prompts_qwen35.py`。配置 `variables.PROMPT_VERSION`
和 rollout 的 `alfworld_prompt_version` 记录运行时版本，实验目录中的 `v1` 不代表提示词版本。

每轮输入包含任务目标、**本条轨迹所有已完成决策的动作历史**（按时间排序）、最新观察及
当前合法动作列表。历史附带 `executed` / `rejected` / `no action` 状态；`executed`
仅表示环境接受了命令，不保证任务推进。不会重复发送旧 observation、旧合法动作列表或旧思考内容。
历史在环境重置时清空，每条轨迹独立；最多有 `max_steps - 1` 条历史进入下一轮 prompt。

```text
Task goal (not an executable action):
{mission}

Previous actions (chronological; not actions to execute again):
1. go to cabinet 1 [executed]
2. open cabinet 1 [executed]

Current observation:
{latest observation}

Current admissible actions (the action value must be copied exactly from this list):
- {command}

After thinking, output exactly one line: Action: <one exact current admissible action>.
```

- 不再注入 Qwen 原生 tools/schema/enum/XML 示例。当前合法动作只作为文本列表提供。
- 两个 Qwen3.5 profile 均开启 `enable_thinking: true`；v5 loop 也保证开启。
  模板预填 `<think>`，模型关闭 `</think>` 后输出一行 `Action: <命令>`；也接受单行裸命令。
- 只解析 `</think>` 后的输出；未闭合 thinking、空输出、多行/多动作、JSON/XML 作为
  **无可解析动作**处理。单行命令解析后由环境检查当前合法性，不做模糊匹配或动作修复。
- `alfworld_action`/`FunctionCall` 仍是 VERL 内部的环境适配接口，**不是模型需要生成的协议**。
  `MULTI_TURN_FORMAT=qwen3_coder` 仅用于满足旧 VERL 基类初始化，v5 决策绕过该 parser。
- `scripts/prepare_verl_dataset.py --profile qwen35` 新生成的数据自动使用 v5。
  现有 v4 parquet 的任务顺序与路径无需改变：第一步从真实环境状态重建 v5 prompt，
  不使用其中旧的系统指令/观察；原 parquet 元数据仍是其制作时的版本。
  `verify_tool_protocol.py` 对 Qwen3.5 会将保存的首步状态重建成 v5，并按文本动作分类。

### 三类互斥惩罚

| 类别 | v5 判定 | 单次奖励 |
|---|---|---:|
| 无动作 | thinking 未闭合或没有可解析的单行文本动作（含多动作、XML/JSON 等） | -0.1 |
| 非法动作 | 文本命令已解析，但不在当前 admissible 列表中，或环境执行报错 | -0.1 |
| 连续重复动作 | 连续有效执行相同的完整命令，从第二次开始逐次计数 | -0.1 |

`A → B → A` 不惩罚；`A → A → A` 惩罚两次。无动作/非法动作打断连续计数，
不会在同一步再扣重复惩罚。合法命令即使返回 `Nothing happens` 也算一次有效执行；
这不是按动作类别或共享函数名计数。保留原生环境 reward，包括最后一步成功奖励。

SwanLab 新名称：`alfworld_penalty/no_action_count`、`invalid_action_count`，以及
`alfworld/valid_action_count/{min,max,mean}`。旧的 `no_tool_call_count`、
`invalid_tool_call_count`、`valid_tool_call_count/*` 保留为同一计数的兼容别名，
**不是额外扣分**。`repeated_action_count` 仍为 rollout batch 中连续重复次数之和。

### 决策预算与训练回放

当前两个 Qwen3.5 配置均为环境驱动，最多 50 次决策，每次最多生成 256 tokens
（包含 thinking 和文本动作）。无动作、非法动作也消耗一次决策额度，但不推进 TextWorld。
环境结束或耗尽决策额度即停止；未闭合思考不会被当成动作执行。
输出存储容量自动派生为 `max_steps × max_new_tokens_per_turn`，当前为 12800。

完整原始输出 token/log-prob 不因解析而裁剪、重编码或替换为 XML。
`alfworld_turn_contexts` 保存每轮实际 prompt（含动作历史）及输出 token，
actor/reference log-prob 和梯度仍使用已有 FSDP turn-context replay。
历史虽然在后续 prompt 中再次出现，但不是新采样的 response token，不会单独增加 loss mask。
thinking token 仍在生成 response mask 内参与训练，不是“隐藏所以不训练”。

加入历史会增加每轮上下文长度；开启思考后，256-token 预算内若没有输出 `</think>` 和
动作会计入无动作惩罚。此次不调整已有步数、KL、entropy 或 PPO 超参数，也不启动训练。
Qwen2.5 保留其原有工具调用协议与终止路径，图像修复训练不变。

## 训练边界

目前只完成隔离组件和无 GPU smoke/preflight；尚未启动正式 baseline 训练。默认只训练一个 `seed0`；
`seed1/seed2` 和三 seed 串行脚本仅用于后续需要均值/方差时的可选重复实验。

### Trajectory-local consecutive-action penalty

Only consecutive valid executions of the same exact ALFWorld command are penalized.
Each streak contributes `-0.1` per repeat after its first execution, in addition to
native environment rewards: `A → A → A` incurs two penalties, but `A → B → A`
incurs none. Commands are compared including arguments (not just the shared
`alfworld_action` tool name). A different command, invalid call, or missing call
breaks the streak. Streak state resets per trajectory, including when reusing
pooled environments.
Each decision receives at most one category: no call (-0.1), invalid call (-0.1),
or valid consecutive repeated action. Terminal success reward is preserved.
`alfworld_penalty/repeated_action_count` retains its name but now counts only
penalized consecutive repeats, summed over the rollout batch. No state-change bonus or history
context is introduced.
