# ALFWorld structured-tool baseline

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

**Qwen3.5-2B 当前配置：关闭 thinking，每次决策最多生成 256 tokens，没有额外的轨迹累计 token 终止条件。**

- 专用工具配置 `config/alfworld_tool_config_qwen35_2b.yaml` 中设置 `environment_driven: true`、`max_new_tokens_per_turn: 256`、`max_steps: 16`。
- 仅因环境 `done` 或耗尽上述 Max Steps 额度正常结束；`max_assistant_turns`、`max_user_turns`、`max_generated_response_length` 不再控制该模式。
- **一次模型决策占用一步额度，包括无工具调用、截断/损坏调用、未知工具或非法动作。** 每一步先判断是否解析到工具调用：没有解析到（无工具、schema 不完整、malformed XML 等）记为惩罚 1 并加入 `-0.1`；只有解析到工具调用后，才判断工具名、参数名/参数 schema 和动作是否合法，非法时记为惩罚 2 并加入 `-0.1`。单步两类惩罚互斥，但同一轨迹的不同决策步可以累计。失败时不伪造合法动作、不调用 TextWorld `env.step()`，因此原生环境执行步数可以小于决策步数。这避免了连续非法输出导致无限重试。
- 输出训练槽位只保存模型生成 token；每轮真实 observation/schema/prompt 保存在 `alfworld_turn_contexts`，由现有 FSDP 回放路径用于训练。没有丢弃模型实际看到的上下文，也没有把拼接后的各轮输出当成一个长上下文训练。
- VERL 仍需固定形状的训练张量。ALFWorld 入口在创建 worker 前自动将 `data.max_response_length` 和 rollout `response_length` 设为 `max_steps × max_new_tokens_per_turn`，默认 **4096**。它是容纳所有可能输出的容量，不是额外的轨迹终止预算；修改工具配置中的步数/单步预算会自动重算，不能通过缩小张量来静默截断。
- SwanLab 记录 `alfworld/valid_tool_call_count/min`、`max`、`mean`，统计每条轨迹中真正调用环境的有效工具调用次数；无工具调用和非法工具调用不会计入该指标。另记录 `alfworld_penalty/no_tool_call_count` 和 `alfworld_penalty/invalid_tool_call_count`，不写入旧的 `num_turns/*` 或其它 penalty series。另保留 `alfworld_termination/done_count` 和 `alfworld_termination/max_steps_count`，统计当前 batch 的结束原因。`done` 表示环境结束，不应直接等同于 `won`。原有 `response_length/clip_ratio` 只是输出是否填满预分配容量，不能作为生成截断率解释。

Qwen3.5-9B、Qwen2.5 等未启用 `environment_driven` 的配置仍保留旧的 4096 累计生成预算及原终止逻辑。此修改不启动训练。

当前 ALFWorld 的 GRPO 奖励由 `ALFWorldTool.execute()` 的原生环境 reward 加上两类且仅两类轨迹惩罚组成：无法解析工具调用为惩罚 1，解析后调用非法工具为惩罚 2；两者每次均为 `-0.1`，同一轨迹按决策步累计。SwanLab 记录有效环境工具调用次数及两类惩罚次数，不再记录旧的 `num_turns/*` 系列；同时保留两个环境终止原因计数。

当前隔离 `ALFWorldTool` 已返回 `done/truncated` 指标，并由 ALFWorld 专用 AgentLoop 完成 `done → TERMINATED` 桥接；环境完成后不会继续生成。Qwen3.5-2B 当前配置的决策上限为 16 次，不能将该上限直接解释为实际平均交互次数。

该桥接已实现为隔离的 `alfworld_tool_agent`，通过 `config/agent_loops.yaml` 注册；图像修复仍使用共享的 `tool_agent`，不会进入 ALFWorld 分支。生成的 VERL parquet 将 `agent_name` 固定为 `alfworld_tool_agent`；手工构造数据时必须同时设置 `agent_name=alfworld_tool_agent` 和 `data_source=alfworld`，否则会回退到共享 loop。

## 训练边界

目前只完成隔离组件和无 GPU smoke/preflight；尚未启动正式 baseline 训练。默认只训练一个 `seed0`；
`seed1/seed2` 和三 seed 串行脚本仅用于后续需要均值/方差时的可选重复实验。

### Trajectory-local repeated-action penalty

Valid executions of the same exact ALFWorld command are counted across the entire
trajectory, including nonconsecutive repetitions. Occurrence `k` contributes
`-0.1 * (k - 1)` in addition to its native environment reward (first execution: 0;
second: -0.1; third: -0.2). There is no cap. Invalid attempts do not enter the count,
and counters reset per trajectory, including when reusing pooled environments.
Each decision receives at most one category: no call (-0.1), invalid call (-0.1),
or valid repeated action. Terminal success reward is preserved. The third counter
is `alfworld_penalty/repeated_action_count`. No state-change bonus or history
context is introduced.
