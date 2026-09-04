# ALFWorld structured-tool baseline

本目录隔离 ALFWorld 文本环境与 old-VERL baseline。Qwen2.5-1.5B、Qwen3.5-2B 与 Qwen3.5-9B 的模型、parser、提示词、parquet、Hydra 入口、启动脚本、日志、checkpoint 和 SwanLab 实验名均按 profile 分离；当前默认 profile 为 `qwen35_2b`。公共的 ALFWorld 环境、奖励、Validator 与 GRPO 参数保持共用。详情见 `MODEL_PROFILES.md`。ALFWorld 使用隔离的 `alfworld_tool_agent`；图像修复继续使用共享的 `tool_agent`，不修改其行为。

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

该 baseline 是多步轨迹级 GRPO：一次工具调用对应一次环境 `step`，随后把新的 observation 和 admissible actions 回传给模型继续决策；单条轨迹的环境交互理论上限为 50 次，但会在任务 `done`、工具循环终止条件或累计生成 token 上限时提前结束。当前配置的 `max_generated_response_length=4096` 是整条轨迹累计生成预算，实际交互次数通常会低于 50。外层 `data.max_response_length` 同步设为 `4096`，避免 rollout 响应张量容量低于多轮累计生成预算。`single prompt` 只描述每轮 prompt 组织方式。

当前 ALFWorld 使用原生环境奖励加协议惩罚：GRPO 最终奖励为 `ALFWorldTool.execute()` 从 `env.step()` 返回的 reward 与协议错误单步惩罚之和。格式错误、无工具调用、未知工具、非法 JSON 参数、以及不在当前 admissible actions 中的动作，各自按实际发生的 assistant turn 计一次 `-0.05`；合法动作的环境 reward 不变。惩罚原因和次数会以 `alfworld_penalty/*` 指标记录到 SwanLab，包含 `format_error_count`、`malformed_tool_call_count`、`no_tool_call_count`、`unknown_tool_count`、`invalid_action_count`、`total_count` 和 `total_value`。

SwanLab 指标含义：

- `alfworld_penalty/format_error_count`：完整工具调用外夹带文本、重复工具调用等格式错误次数；
- `alfworld_penalty/malformed_tool_call_count`：包含工具调用标记但 XML/结构无法解析的次数；
- `alfworld_penalty/no_tool_call_count`：本轮没有任何工具调用的次数；
- `alfworld_penalty/unknown_tool_count`：调用未注册工具名的次数；
- `alfworld_penalty/invalid_action_count`：`action` 不在当前 admissible action 列表中的次数；
- `alfworld_penalty/total_count`：以上惩罚事件总次数；
- `alfworld_penalty/total_value`：惩罚总值，正常为 `-0.05 × total_count`。

这些指标在每个训练 step 聚合当前 rollout batch 后写入 SwanLab；它们是诊断统计，不改变 ALFWorld 原生 `won`/`score` 的定义。

注意：当前隔离 `ALFWorldTool` 已返回 `done/truncated` 指标，但 old-VERL 通用 `ToolAgentLoop` 的提前终止逻辑原生只识别 `stop` 工具。正式 pilot 前必须增加并验证 ALFWorld 专用的 done→TERMINATED 桥接；在该桥接完成前，不能声称“环境成功后必然立即结束”，也不能把 50 次作为实际平均交互次数。

该桥接已实现为隔离的 `alfworld_tool_agent`，通过 `config/agent_loops.yaml` 注册；图像修复仍使用共享的 `tool_agent`，不会进入 ALFWorld 分支。生成的 VERL parquet 将 `agent_name` 固定为 `alfworld_tool_agent`；手工构造数据时必须同时设置 `agent_name=alfworld_tool_agent` 和 `data_source=alfworld`，否则会回退到共享 loop。

## 训练边界

目前只完成隔离组件和无 GPU smoke/preflight；尚未启动正式 baseline 训练。默认只训练一个 `seed0`；
`seed1/seed2` 和三 seed 串行脚本仅用于后续需要均值/方差时的可选重复实验。
