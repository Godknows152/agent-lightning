# ALFWorld baseline（Qwen3.5 v7 精简XML / Qwen2.5 工具协议）

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
| 无动作 | thinking 未闭合或没有可解析的单行文本动作（含多动作、XML/JSON 等） | 首次出现立即终止轨迹，追加一次 -5 |
| 非法动作 | 文本命令已解析，但不在当前 admissible 列表中，或环境执行报错 | -0.1 |
| 连续重复动作 | 连续有效执行相同的完整命令，从第二次开始逐次计数 | -0.1 |

`A → B → A` 不惩罚；`A → A → A` 惩罚两次。无动作/非法动作打断连续计数，
不会在同一步再扣重复惩罚。合法命令即使返回 `Nothing happens` 也算一次有效执行；
这不是按动作类别或共享函数名计数。保留原生环境 reward，包括最后一步成功奖励。

SwanLab 统一仅记录 action 名称：`alfworld_penalty/no_action_count`、`invalid_action_count`，以及
`alfworld/valid_action_count/{min,max,mean}`。旧的 `no_tool_call_count`、
`invalid_tool_call_count`、`valid_tool_call_count/*` 不再重复上报。
内部轨迹字段保留旧名称以兼容历史数据，此调整不改变计数或奖励。`repeated_action_count` 仍为 rollout batch 中连续重复次数之和。

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
Each decision receives at most one category: no call (one-time -5, terminates trajectory), invalid call (-0.1),
or valid consecutive repeated action. Terminal success reward is preserved.
`alfworld_penalty/repeated_action_count` retains its name but now counts only
penalized consecutive repeats, summed over the rollout batch. No state-change bonus or history
context is introduced.

### Thinking budget 独立采样验证

2026-09-10 已添加 Qwen3.5/SGLang 预算对照实验：
`docs/THINKING_BUDGET_SGLANG_20260910.md`。配置位于
`config/diagnostics/thinking_budget_qwen35_9b.json`，可用
`scripts/test_sglang_thinking_budget.sh` 在空闲 GPU 上复现。
这是独立采样测试，不是正式 PPO 开关：预算强制 token 的 log-prob 与原始模型概率不同，
在接入当前 bypass + turn replay 训练前需要处理确定性边界的 loss mask/概率一致性。

### 无动作立即终止（2026-09-10）

首次无可解析动作后不再采样后续步；失败步 token、log-prob 和 replay 上下文仍参与训练。
追加一次 -5 惩罚，不覆盖此前环境奖励、非法动作或连续重复动作惩罚。
终止原因是 `no_tool_call`，统计为 `alfworld_termination/no_action_count`，不计入环境成功或步数耗尽。
非法动作和连续重复动作仍各扣 -0.1，不因本项规则提前终止。

### v6：简短思考与512-token单步预算

当前 Qwen3.5 提示词版本为 `alfworld_qwen35_v6_action_history_brief_thinking`。
仅判断下一步，要求1–2句简短思考，不复述任务、观察或枚举候选动作；随后闭合thinking并输出一行 `Action:`。
保留动作历史、无schema文本动作和thinking模式；这是提示词软约束，不强制插入结束token。
两份工具配置的 `max_new_tokens_per_turn` 均为768（在原512基础上增加256），包含thinking和最终动作。
运行时仍按实际 `max_steps * 512` 推导response存储容量；50步配置为38400 token。
无动作立即终止并追加−5、非法/连续重复动作各−0.1的规则保持不变。

### v7：精简XML工具schema（当前版本）

`alfworld_qwen35_v7_compact_xml_history_thinking` 替代v6的文本动作输出。
系统模板每个请求仅注入一次 `alfworld_action(action: string)` 工具定义和XML调用格式，
不序列化动态schema的动作enum；当前合法动作只在用户状态提示中列出。
保留1–2句简短thinking、历史动作及状态、512-token总生成预算。
闭合thinking后调用格式为：

```xml
<tool_call>
<function=alfworld_action>
<parameter=action>look</parameter>
</function>
</tool_call>
```

解析只检查thinking之后的调用，执行时仍按当前环境合法动作严格校验。
无可解析/完整调用立即终止并追加−5；非法调用、连续重复有效动作仍各−0.1。
SwanLab保持统一action命名；回放保留原始生成token及log-prob。

### v7 惩罚分类与完整输出校验（2026-09-11）

以下规则取代v5/v6的文本动作判定；惩罚系数不变，每步只归入一个类别。

| 输出情况 | 处理 |
|---|---|
| thinking未闭合、没有XML调用、调用/函数/参数标签不完整 | 无动作，追加−5，立即结束轨迹 |
| 完整调用但工具名错误，参数缺失、为空、重复、多余或为多行 | 非法动作，−0.1，不执行环境动作，继续（除非步数耗尽） |
| 多个调用、一个完整调用后又开始第二个调用 | 非法动作，−0.1；不再截断后执行第一个 |
| 调用前后有额外说明（thinking内说明除外） | 非法动作，−0.1，不执行 |
| 合法XML但命令不在当前环境合法列表中或执行失败 | 非法动作，−0.1 |
| 连续成功执行完全相同的命令 | 第二次起每次−0.1；不同命令/失败打断连续计数 |

校验只读取thinking后的文本；仅在解析视图剥离末尾传输标记，不改训练token。
XML模式不再裁掉首个调用后的生成内容，避免掩盖多调用和额外说明。
`alfworld_last_decision_reason` 在轨迹元数据中区分细分原因，不增加tool_call别名指标。
正式训练、预检和协议诊断共享 `parse_xml_decision`；历史v5测试路径仍保留。


### 思考超长无动作统计（2026-09-11）

单步输出预算从512增加到768 token。若XML模式生成达到上限仍未生成 `</think>`，
该步按无动作规则追加−5并终止轨迹，同时记录 `alfworld_penalty/thinking_truncated_no_action_count`。
该指标按rollout batch统计轨迹数，不是token数；只有“thinking未闭合”计入，
已闭合thinking但XML缺失/不完整的无动作不计入。

### Ray dashboard agent 端口冲突防护（2026-09-12）

ALFWorld 入口 `alfworld_baseline.main_ppo` 自动安装进程内启动防护，适用于共用该入口的
2B/9B 训练，无需更改训练 YAML 或启动命令，不修改安装环境内的 Ray 文件。

Ray 2.54 在节点 IP 上自动分配 agent gRPC 端口后，还会将同一端口绑定到
`127.0.0.1`，可能撞到 Clash Verge 等仅监听回环地址的服务。启动防护在创建 raylet
前以 `0.0.0.0:0` 申请候选端口（不启用端口复用），再将结果传给
`start_raylet(metrics_agent_port=...)`。通配地址检查覆盖所有本地 IPv4 地址，避免只检查
网卡地址而遗漏回环端口。探测 socket 不监听连接；Ray 实际监听地址不变。
已有显式端口保留，但占用时提前报清楚的错误，不擅自换端口、结束服务或重试整个训练。
启动日志会输出 `ALFWorld Ray dashboard agent gRPC port=...`。

边界：探测 socket 必须在 Ray 绑定前释放，因此仍有很短的跨进程端口交接竞态窗口；
此防护避免已存在的端口冲突，不声称对任意并发新服务的抢占提供原子保证。
附加到已有 Ray 集群时不创建 raylet，不改变其端口。升级 Ray 后应重新运行兼容性测试。

检查：
```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest -q \
  examples/alfworld_grpo_baseline/tests/test_ray_startup.py
```

验收：9 项新增 CPU 单测和 129 项原有回归测试通过（共 138 项）；在 `alfworld-verl` 环境、原 Clash Verge 服务保持运行的情况下，
以 `num_gpus=0`、1 CPU、128 MiB object store 成功初始化独立 Ray 实例，运行一个 CPU
remote task 后使用 `ray.shutdown()` 关闭该测试实例。未启动 RL/SFT，未验证模型加载。

### 9B RL 关闭 thinking（2026-09-12）

当前 `qwen35_9b` profile 设置 `data.apply_chat_template_kwargs.enable_thinking: false`，
提示词版本为 `alfworld_qwen35_v7_compact_xml_history_nothinking`。
Agent 尊重配置，不再强制开启 thinking；初始及后续状态提示词、解析器和轨迹版本随同切换。
2B profile 保持 thinking 开启。9B 使用与非思考 SFT 相同的输入侧空
`<think>\n\n</think>\n\n` 前缀；模型应直接生成 XML，无须自己生成 `</think>`。
空 thinking 标签出现在输入中不代表模型生成了思考。
严格 XML、无动作立即终止并罚 −5、非法及重复动作惩罚不变。启动预检同时支持两种模式，
并验证 PROMPT_VERSION 与开关一致。切换模式时需同时更新 profile 的开关和版本名。

### SwanLab 轨迹终止指标（2026-09-12）

`alfworld_termination` 现在只输出互斥的五分类及总数：

- `success_count`：环境返回 `won=True`，任务真正完成；
- `environment_timeout_count`：环境步数达到 50 步且未完成；
- `decision_limit_count`：Agent 决策次数达到上限，且未被其他原因终止；
- `no_tool_call_count`：模型输出无法解析为可执行动作；
- `env_failure_count`：其他环境终止或未分类终止；
- `total_trajectories`：本 rollout batch 的轨迹总数。

五类计数加总应等于 `total_trajectories`。`done_count` 和 `max_steps_count` 不再写入
SwanLab。`invalid_action_count` 和 `repeated_action_count` 仍属于
`alfworld_penalty`，因为它们不是终止原因。环境工具层使用 `won` 区分成功与 TextWorld
因步数上限返回的 `done=True`；后者记录为 `environment_timeout`，避免把超时误记成成功。
