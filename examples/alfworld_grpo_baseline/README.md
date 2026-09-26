# ALFWorld baseline（GiGPO 对齐提示词）

本目录隔离 ALFWorld 文本环境与原生 veRL baseline。Qwen2.5-1.5B、Qwen3.5-2B 与 Qwen3.5-9B 的模型、parser、提示词、parquet、Hydra 入口、启动脚本、日志、checkpoint 和 SwanLab 实验名均按 profile 分离；当前默认 profile 为 `qwen35_2b`。三个 profile 共用 `alfworld_gigpo_qwen3_xml_v1` 提示词协议，公共的 ALFWorld 环境、奖励和 Validator 保持共用，模型专属的运行时/性能覆盖保存在各自的 Hydra 配置中。当前 Qwen3.5-2B 使用独立的8个 AgentLoop worker、环境池、关闭 Actor FSDP offload、关闭梯度检查点和独立 PPO batch 设置。详情见 `MODEL_PROFILES.md`。ALFWorld 使用隔离的 `alfworld_tool_agent`；图像修复继续使用共享的 `tool_agent`，不修改其行为。

## 两套可选奖励与优势后端（2026-09-23）

原入口仍默认使用 `trajectory`，新增入口使用 `gigpo_grpo`。两者共用环境、模型和严格 XML
协议，但奖励分配、优势计算及默认输出目录独立。

| 行为 | 原版 `trajectory` | 新版 `gigpo_grpo` |
| --- | --- | --- |
| 环境成功奖励 | 10，失败为 0 | 10，失败为 0 |
| 无动作 / 非法动作 | 各次 −0.2，累计到轨迹总分 | 对应步骤各 −0.2 |
| 重复动作 | 不扣分（0），保留重复次数统计 | 不扣分（0），保留重复次数统计 |
| 分组归一化 | 同一初始任务下每条轨迹一个总分 | 同一初始任务下所有决策步骤各一个分数 |
| 优势应用 | 同一轨迹所有步骤共享优势 | 每步单独得到优势，该步生成 token 共享它 |

新后端模仿 `/home/LXJ/Python_Projects/GiGPO` 中 **GRPO 基线** 的
`compute_mean_std_cross_steps=True` 路径，不包含 GiGPO 算法的相同状态分组。
令 `S_i` 为轨迹是否成功，`c_it` 为该步的惩罚（0 或 −0.2）：

```text
q_it = 10 × S_i + c_it
G_x = 同一初始任务 x 下所有轨迹的所有非 padding 决策步
A_it = (q_it − mean(q in G_x)) / (sample_std(q in G_x) + 1e-6)
```

例如失败轨迹三个步骤中只有第二步非法，训练分数为 `[0, -0.2, 0]`；若最终成功，
则为 `[10, 9.8, 10]`。惩罚不会累加后再广播给整条轨迹。组内均值和标准差仍受所有步骤影响，
因此局部扣分仍会间接改变组内其他步骤的优势；长轨迹也会贡献更多步骤。这里既不按步序号
对齐，也不按相同观察分组。多样本常数组优势为零；单样本组沿用基线的 mean=0、std=1。

无动作与非法动作各扣 −0.2；重复动作惩罚已取消，首次与重复的有效动作均不扣分。
重复仍指同一轨迹内此前有效执行过的**完整命令**，不要求连续；兼容字段
`alfworld_penalty/repeated_action_count` 继续记录重复次数，其数值不再代表扣分次数。
保留本仓库的完整输出、工具格式和 admissible action 检查，不改用 GiGPO 较宽松的格式判定。
验证阶段采用纯环境得分 0/10，完整轨迹展示继续保留；训练分数包含本步惩罚。
原有熵、KL、PPO loss、模型配置及数据集不变。

```bash
# 新后端：默认 Qwen3.5-2B，仅预检
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_step_grpo.sh --preflight

# 新后端：自动恢复最新 checkpoint；目录为空时新建训练（后台）
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_step_grpo.sh

# 原版后端：原入口与默认行为保留
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh
```

新入口支持 `--smoke`、`--pilot` 及 `ALFWORLD_MODEL_PROFILE`。也可以在共用启动脚本前设置
`ALFWORLD_TRAINING_BACKEND=gigpo_grpo`；直接调用 Hydra 入口时添加
`+training_backend=gigpo_grpo`。应整套切换，不能仅替换奖励或优势计算器。
新配置位于 `config/training_backend/gigpo_grpo.yaml`，Qwen3.5-2B 正式实验名为
`qwen3.5_2B_GiGPO后端`，smoke/pilot 实验名追加对应的 `_smoke_seedN` / `_pilot_seedN` 后缀。
Qwen3.5-2B 正式训练的 checkpoint/rollout/验证数据及本地 SwanLab 目录位于
`log/alfworld/qwen35_2b/gigpo_grpo/`；smoke/pilot 分别使用其 `smoke_seedN/`、`pilot_seedN/` 子目录。
其他模型仍使用 `outputs/alfworld/<profile>/gigpo_grpo/v1/2gpu/seedN/`。
主日志位于 `log/alfworld/<profile>/gigpo_grpo/`；不复用原版实验的自动恢复目录。

当前 `alfworld_step_grpo.sh` 的 2B 正式训练与预检默认叠加
`config/resume_run/qwen35_2b_gigpo.yaml`：使用 `resume_mode: auto`、`resume_from_path: null`，
自动恢复上述输出目录中最新发布的 checkpoint，总目标仍为 150 步。
恢复时复用 `.swanlab_experiment.json` 中的实验 ID，并以 `resume="must"` 续写；
目录为空时从初始模型开始并创建新实验。预检会检查 checkpoint 分片和实验身份。
Ray 固定使用物理网卡地址 `10.246.1.30`，避免 Mihomo TUN 被关闭时失联；地址变化时可通过
`ALFWORLD_RAY_NODE_IP` 更新。smoke/pilot 和其他模型不加载此续训配置。
以后要从头新建实验，可通过 `ALFWORLD_OUTPUT_DIR` 指定新的空目录。

新后端使用独立的 AgentLoop、队列 worker 和优势注册名：适配现有 parquet 的旧 `agent_name`，
避开原生 V1 的“末步奖励覆盖整条轨迹”和“轨迹优势广播”路径。共享 veRL/GiGPO 源码未修改。
后续历史说明未注明后端的奖励规则均指原版；当前原版系数以本节及代码为准。

## SGLang LoRA 同步兼容

`workers.py` 在 ALFWorld 专用 Ray worker 内选择 `sglang_rollout.py` 的适配器。
LoRA 动态加载遵循当前 SGLang 的 `serialized_tensors` 协议，发送一个完整张量字典，
由 SGLang 内部进行 TP 切分；所有训练 rank 都参与 FSDP 张量收集，只有推理 TP leader 发送请求。
基座权重同步仍使用原生路径；共享 veRL 和图像恢复后端文件无需修改。
当前 2B 配置关闭 `use_fused_kernels`，避免现有 Liger 缺少
`LigerFusedLinearScaledCrossEntropyFunction` 导致参考策略和 Actor 前向失败；使用标准输出层计算损失。
ALFWorld 指标汇总直接迭代 TensorDict 返回的 `extra_fields`，兼容 `LinkedList`，
并只统计非 padding 轨迹的最后一个环境步。

## 验证采样中的完整轨迹

SwanLab 的 `val/generations` 每条记录展示一条完整轨迹：`input` 是首步输入，
`output` 按 `Step 1`、`Step 2` 等顺序保留每步的完整输入、模型思考和回答，
包括格式错误或无效动作；`score` 是整条轨迹的最终奖励。
`trainer.log_val_generations: 8` 表示抽取 8 条轨迹。

`validation_logging.py` 通过 ALFWorld 专用 trainer 的验证导出钩子聚合步骤，
保持原生逐步训练样本和奖励统计不变。所有验证步骤同时保存至
`${trainer.default_local_dir}/validation/<global_step>.jsonl`，其中 `uid` 为
`<task_uid>_<rollout_id>_<step_index>`，步序号从 0 开始。
完整展示依赖 `trainer.validation_data_dir`；显式设为 `null` 会恢复原生的末步展示。
原生 `num_turns` 仍表示单步样本的轮数，完整轨迹长度见展示文本的 `decision steps`。
已启动的训练进程需要在重启/续训后才会加载新的展示逻辑，旧验证表不会自动补全。

## 原版后端的惩罚对齐（2026-09-25）

原版后端的无动作、非法动作均为每次 −0.2，重复动作惩罚和递增系数均为 0，
与 GiGPO 风格后端的三类惩罚值一致。原版仍按轨迹累计：
`轨迹分数 = 10 × 是否成功 − 0.2 × (无动作次数 + 非法动作次数)`。
例如一条失败轨迹含两次非法动作和一次无动作，则整条轨迹分数为 −0.6，
再按原版轨迹级 GRPO 计算优势；不会切换成逐步奖励或逐步优势。
以下计数规则仅用于统计，取代下文历史版本中的重复动作扣分规则。
同一轨迹中，完整命令在此前有效执行过后再次有效执行，就计为一次重复，不要求连续。
所有命令共享该轨迹的重复计数；切换命令或非法调用不重置计数，新轨迹从零开始。
`A → A → B → B → A` 共三次重复，但不扣分。
`alfworld_penalty/repeated_action_count` 仍统计次数，不再表示扣分次数。

## 当前 GiGPO 提示词（`alfworld_gigpo_qwen3_xml_v1`）

三个 ALFWorld 模型 profile 都使用 `src/alfworld_baseline/prompts_gigpo.py` 中的
GiGPO 模板。首轮只放当前 observation 和 admissible actions；后续轮次放任务目标、
完整轨迹步数、最近两条 `[Observation i, Action i]` 历史、当前 observation 和动作列表。
每轮模型输出仍必须是 Qwen3 XML 工具调用：

```xml
<think>...</think>
<tool_call>
<function=alfworld_action>
<parameter=action>EXACT_COMMAND</parameter>
</function>
</tool_call>
```

`EXACT_COMMAND` 必须逐字来自当前 admissible actions；环境仍负责最终合法性检查。
`<|im_end|>` 和 `<|endoftext|>` 只作为传输层结束标记剥离。

验证命令（只加载本地模型，不启动 Ray/GRPO）：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src:/home/LXJ/Python_Projects/verl \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python \
  examples/alfworld_grpo_baseline/scripts/verify_tool_protocol.py \
  --samples 32 --max-new-tokens 128 --device cuda:0
```

结果写入 `outputs/diagnostics/tool_protocol.json`。

切换模型后应重新运行诊断；该抽样只作为格式回归证据，不等同于对任意随机采样或正式训练
成功率的保证。

训练文件按统一的 profile 约定组织：共享参数位于 `config/alfworld_common_config_2gpu.yaml`，模型参数位于 `config/model_profiles/`，模型入口位于 `config/alfworld/<profile>/v1/`，启动脚本位于 `scripts/alfworld/`。checkpoint、rollout、主日志和 SwanLab 数据分别写入 `outputs/alfworld/<profile>/v1/2gpu/` 与 `log/alfworld/<profile>/v1/2gpu/`。完整目录规范见 `TRAINING_LAYOUT.md`。

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

# 组件、Qwen tokenizer 与原生 veRL/ALFWorld 统一环境
PYTHONPATH=examples/alfworld_grpo_baseline/src:/home/LXJ/Python_Projects/verl \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest -q examples/alfworld_grpo_baseline/tests
/home/LXJ/anaconda3/envs/alfworld-verl/bin/python examples/alfworld_grpo_baseline/scripts/preflight.py
```

`alfworld-verl` 是 ALFWorld 运行环境；训练代码通过 `ALFWORLD_VERL_ROOT`（默认 `/home/LXJ/Python_Projects/verl`）加载原生 veRL。`preflight.py` 检查正式 veRL runtime 所需的 `alfworld`、`gymnasium`、`stable_baselines3`、`transformers`、`pandas`、`pyarrow` 和 `omegaconf`，要求 Qwen2.5 tokenizer 存在原生 chat template，并要求隔离目录下已存在 `data/train.parquet` 与 `data/test.parquet`；`preflight_alfworld.py` 只检查独立 ALFWorld 数据环境。图像修复继续使用自己的共享后端。

Qwen3.5 的线性注意力训练需要 FLA 和 causal-conv1d 快速内核；预检会在内核不可用时直接报错，
避免无意间使用缓慢的 PyTorch 后备实现。A800 / Python 3.12 / PyTorch 2.9.1+cu128 环境使用
`requirements-qwen35-kernels.txt` 固定版本。causal-conv1d 使用上游发布的
`cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64` wheel；安装依赖时保留现有 PyTorch、Triton 和
`.pydeps` 中的 Transformers 版本。Qwen3.5-2B 已关闭梯度检查点；无检查点时 actor 微批 16
在熵反向计算中超出 80GB 显存，因此 actor 微批设为 8、reference 有效微批保留 16。
梯度累积维持全局 PPO minibatch 为 128，采样规模仍为 16 个任务 × 8 条轨迹。

GPU 内核回归测试覆盖 FLA 与 PyTorch 参考实现的前向和反向数值，以及原生 veRL 中不同长度样本
拼接后的输出、输入梯度和参数梯度与独立执行的一致性。它使用真实模型配置中的线性注意力维度，
独立于禁止 CUDA 初始化的 CPU 测试集运行：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src:/home/LXJ/Python_Projects/verl:examples/image_restoration_multi_agent/old_verl_grpo/.pydeps \
  CUDA_VISIBLE_DEVICES=0 /home/LXJ/anaconda3/envs/alfworld-verl/bin/python \
  examples/alfworld_grpo_baseline/scripts/test_qwen35_fast_kernels.py
```

先生成 veRL 数据（该步骤会为每条任务 reset 一次，以构造首轮 observation 和动态 admissible actions；脚本按 64 个游戏分块加载，避免逐条重复初始化 TextWorld）：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src \
  conda run -n alfworld-verl python examples/alfworld_grpo_baseline/scripts/prepare_verl_dataset.py \
  --split train --limit 64 --output examples/alfworld_grpo_baseline/data/train.parquet
PYTHONPATH=examples/alfworld_grpo_baseline/src \
  conda run -n alfworld-verl python examples/alfworld_grpo_baseline/scripts/prepare_verl_dataset.py \
  --split test --limit 32 --output examples/alfworld_grpo_baseline/data/test.parquet
```

生成的数据包含 `extra_info.tools_kwargs.alfworld_action.create_kwargs.game_file`，供 veRL
工具实例创建时加载对应游戏；训练配置在 pilot 前需将 `variables.DATA_DIR` 覆盖为该目录。

环境接受的最终动作始终是 ALFWorld 原生文本字符串，例如 `go to cabinet 1`；模型通过 Qwen3 XML 的 `action` 参数传递它。

该 baseline 是多步轨迹级 GRPO：一次合法工具调用对应一次环境 `step`，随后把新的 observation 和 admissible actions 回传给模型继续决策。`single prompt` 只描述每轮 prompt 组织方式。

## 历史协议记录

以下 v5/v7 小节保留用于复现实验记录；当前三个训练 profile 已统一使用上面的 GiGPO 协议。

## Qwen3.5 v5：历史动作 + 文本动作 + thinking（历史记录，2026-09-10）

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

After thinking, output exactly one Qwen3 XML tool call.
```

- 这是历史文本动作协议；当前 profile 使用 Qwen3 XML 工具调用。
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
| 无动作 | thinking 未闭合或没有可解析的单行文本动作（含多动作、XML/JSON 等） | 每次扣 -2，消耗一个决策步后继续 |
| 非法动作 | 文本命令已解析，但不在当前 admissible 列表中，或环境执行报错 | -0.1 |
| 连续重复动作 | 连续有效执行相同的完整命令，从第二次开始逐次计数 | -0.1 |

`A → B → A` 不惩罚；`A → A → A` 惩罚两次。无动作/非法动作打断连续计数，
不会在同一步再扣重复惩罚。合法命令即使返回 `Nothing happens` 也算一次有效执行；
这不是按动作类别或共享函数名计数。保留原生环境 reward，包括最后一步成功奖励。

SwanLab 统一仅记录 action 名称：`alfworld_penalty/no_action_count`、`invalid_action_count`，以及
`alfworld/valid_action_count/{min,max,mean}`。旧的 `no_tool_call_count`、
`invalid_tool_call_count`、`valid_tool_call_count/*` 不再重复上报。
内部轨迹字段保留旧名称以兼容历史数据，此调整不改变计数或奖励。`repeated_action_count` 为 rollout batch 中全轨迹重复次数之和。

### 原生 veRL 迁移（实现完成，CPU 验证）

保留“最近两步观察/动作 + 当前状态、每轮重建 prompt”。AgentLoop 为每步返回
真实 prompt/response 的原生 `AgentLoopOutput`，由原生 V1 按完整轨迹归一化
GRPO 优势、广播至步骤，再交给原生 FSDP。已移除定制 turn-context 回放依赖。

已完成后端隔离、入口、单步预算、奖励/指标去重、异常资源释放、checkpoint 完整性检查与
SwanLab 原 run ID 续接。损失使用原生 `token-mean`，长轨迹保留更多训练 token。
Qwen3.5-2B 使用合并后的 SFT 完整模型初始化 Actor，并做全参数更新；KL 参考是
独立冻结的同一 SFT 模型。所有 RL profile 和两种后端均关闭 LoRA。详情、CPU 复现命令和未运行的 GPU 验证范围见
[迁移状态](docs/NATIVE_VERL_MIGRATION.md)。


## 训练边界

本次迁移已完成实现和 CPU 回归/预检；按要求没有启动 GPU 训练或分布式保存/恢复测试。默认只训练一个 `seed0`；
`seed1/seed2` 和三 seed 串行脚本仅用于后续需要均值/方差时的可选重复实验。

### Trajectory-local consecutive-action penalty

Only consecutive valid executions of the same exact ALFWorld command are penalized.
Each streak contributes `-0.1` per repeat after its first execution, in addition to
native environment rewards: `A → A → A` incurs two penalties, but `A → B → A`
incurs none. Commands are compared including arguments (not just the shared
`alfworld_action` tool name). A different command, invalid call, or missing call
breaks the streak. Streak state resets per trajectory, including when reusing
pooled environments.
Each decision receives at most one category: no call (-2 per decision, continues within the step budget), invalid call (-0.1),
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
在接入思考预算控制前需要处理确定性边界的 loss mask/概率一致性。

### 无动作后继续生成（2026-09-21）

每次没有可解析的工具调用时扣 2 分，消耗一个决策步，不调用环境、不释放环境实例。
保留当前任务、观察和合法动作，将失败决策纳入历史后重建下一步输入并继续生成。
失败步的原始 token 和 log-prob 仍参与训练；累计惩罚保留此前奖励，即使随后成功也不会豁免。
轨迹仅在环境结束或达到决策步数上限时正常结束；当前上限为 50 步，全部无动作时累计 -100。
`alfworld_penalty/no_action_count` 及其分类指标按发生次数累计，旧终止原因 `no_tool_call` 仅兼容历史数据。
非法动作与重复动作规则保持不变。

### v6：简短思考与512-token单步预算

历史 Qwen3.5 实验曾使用提示词版本 `alfworld_qwen35_v6_action_history_brief_thinking`。
仅判断下一步，要求1–2句简短思考，不复述任务、观察或枚举候选动作；随后闭合thinking并输出一行 `Action:`。
保留动作历史、无schema文本动作和thinking模式；这是提示词软约束，不强制插入结束token。
原生 veRL 路径不使用 `max_new_tokens_per_turn` 或按环境预算派生 response 容量；
每轮生成和整条多轮响应长度由训练 YAML 控制。
无动作每步扣 2 分并继续，其他动作惩罚规则保持不变。

### v7：精简XML工具schema（历史版本）

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
无可解析/完整调用每步扣 2 分并继续；非法调用与重复动作惩罚规则保持不变。
SwanLab保持统一action命名；回放保留原始生成token及log-prob。

### v7 惩罚分类与完整输出校验（2026-09-11）

以下规则取代v5/v6的文本动作判定；惩罚系数不变，每步只归入一个类别。

| 输出情况 | 处理 |
|---|---|
| thinking未闭合、没有XML调用、调用/函数/参数标签不完整 | 无动作，每步扣 2 分，继续至环境结束或决策步数上限 |
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
该步按无动作规则扣 2 分并继续，累计到 `alfworld_penalty/no_action/overlong_thinking_count`。
该指标按 rollout batch 统计无动作决策次数，不是 token 数；只有“thinking未闭合”计入，
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
历史非思考实验的提示词版本为 `alfworld_qwen35_v7_compact_xml_history_nothinking`。
Agent 尊重配置，不再强制开启 thinking；初始及后续状态提示词、解析器和轨迹版本随同切换。
2B profile 保持 thinking 开启。9B 使用与非思考 SFT 相同的输入侧空
`<think>\n\n</think>\n\n` 前缀；模型应直接生成 XML，无须自己生成 `</think>`。
空 thinking 标签出现在输入中不代表模型生成了思考。
严格 XML；无动作每步扣 2 分并继续，非法及重复动作惩罚不变。启动预检同时支持两种模式，
并验证 PROMPT_VERSION 与开关一致。切换模式时需同时更新 profile 的开关和版本名。

### SwanLab 轨迹终止指标（2026-09-12）

`alfworld_termination` 现在只输出互斥的五分类及总数：

- `success_count`：环境返回 `won=True`，任务真正完成；
- `environment_timeout_count`：环境步数达到 50 步且未完成；
- `decision_limit_count`：Agent 决策次数达到上限，且未被其他原因终止；
- `no_tool_call_count`：兼容历史无动作终止；当前无动作不再终止，此项为 0；
- `env_failure_count`：其他环境终止或未分类终止；
- `total_trajectories`：本 rollout batch 的轨迹总数。

五类计数加总应等于 `total_trajectories`。`done_count` 和 `max_steps_count` 不再写入
SwanLab。`invalid_action_count` 和 `repeated_action_count` 仍属于
`alfworld_penalty`，因为它们不是终止原因。环境工具层使用 `won` 区分成功与 TextWorld
因步数上限返回的 `done=True`；后者记录为 `environment_timeout`，避免把超时误记成成功。


### 全参数 RL（2026-09-26）

所有 ALFWorld RL 配置统一使用 `lora_rank: 0`、`lora_adapter_path: null`，新版
`model.lora` 字段也关闭 adapter。模型路径由 profile 的 `SFT_MODEL` 决定，可用
`ALFWORLD_SFT_MODEL` 指定已有完整模型。运行入口拒绝 adapter 目录或重新启用 LoRA。

Qwen3.5-2B 的基底来自 `qwen35_2b_gigpo_expert_v1_latest/adapter/checkpoint-100`，
其真实目录为 `qwen35_2b_gigpo_expert_v1_20260921_105449_688178`。首次启动前在 CPU
合并 adapter，导出到该目录的 `merged-checkpoint-100`，附带源路径、adapter SHA256 和
参数量清单。完整模型不纳入 Git；保留原 SFT adapter。

```bash
PYTHONPATH=examples/image_restoration_multi_agent/old_verl_grpo/.pydeps \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python \
  examples/alfworld_grpo_baseline/scripts/export_sft_model.py --model-profile qwen35_2b
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh
```

默认后台使用原生 veRL V1 trajectory GRPO，保持 16 个任务 × 8 条轨迹、学习率、KL
和熵系数。2 × A800 80GB 上使用动态 token 分批：Actor 每卡 24576 token，参考模型
每卡 65536 token；全局 PPO mini-batch 仍为 128 条交互步骤。保留模型梯度检查点，关闭
熵重计算及 Actor 参数/优化器卸载，使用 FSDP `reshard_after_forward=false` 和梯度累积
末尾同步。静态 micro-batch 字段仅作为关闭动态分批时的回退值。
调优比较见 [吞吐测试记录](docs/THROUGHPUT_TUNING_20260926.md)。
新输出位于 `outputs/qwen3.5_2B/full_sft_grpo_tuned`，SwanLab 实验为
`qwen3.5_2B_full_sft_grpo_tuned`，避免读取调优前或旧 LoRA checkpoint。GiGPO 后端输出也增加
`full_sft` 子目录，两种后端仍各自独立。

9B 原 SFT checkpoint-300 当前不在磁盘，使用前需恢复并导出，或提供完整 SFT 模型；
预检会报错而不会回退到未合并的基础模型。Qwen2.5 旧 profile 没有本地 ALFWorld SFT
记录，保留上游 Instruct 完整模型，支持通过同一环境变量指定 ALFWorld SFT 导出。
