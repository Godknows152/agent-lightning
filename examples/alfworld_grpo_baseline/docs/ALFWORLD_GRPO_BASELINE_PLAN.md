# ALFWorld 文本版 GRPO Baseline 实验计划

## 1. 后端确认

当前项目 `examples/image_restoration_multi_agent/old_verl_grpo/` 使用的是项目内定制的
VERL 后端，而不是直接依赖 Conda 环境中的通用 pip 版 VERL。

- 后端源码：`examples/image_restoration_multi_agent/verl_backend/`
- 后端版本：`verl/version/version = 0.8.0.dev`
- 当前后端 commit：`f5b5eae0c5789e9c5af3698a4154f56473a0ab67`
- 训练入口：`python -m verl.trainer.main_ppo`
- 原始 Python：`/home/LXJ/anaconda3/envs/verl/bin/python`
- 统一 Python：`/home/LXJ/anaconda3/envs/alfworld-verl/bin/python`
- Ray：`/home/LXJ/anaconda3/envs/alfworld-verl/bin/ray`

ALFWorld baseline 必须复用相同的 `verl_backend`、但使用新建的统一 `alfworld-verl` Conda
环境和 `main_ppo` 入口；原始 `verl` 环境保持不变。
launcher 必须显式设置 `PYTHONPATH`，并在 preflight 中打印 `verl.__file__`，确认解析到
项目内 `examples/image_restoration_multi_agent/verl_backend/verl/`。

## 2. 目录隔离

ALFWorld 的新代码、配置、launcher、日志、checkpoint 和评估结果全部放在：

    /home/LXJ/Python_Projects/Agent_Lightning/examples/alfworld_grpo_baseline/

计划目录：

    examples/alfworld_grpo_baseline/
    ├── ALFWORLD_GRPO_BASELINE_PLAN.md
    ├── README.md
    ├── config/
    │   ├── env.yaml
    │   ├── alfworld_common_config_2gpu.yaml
    │   ├── alfworld/v1/alfworld_config_2gpu.yaml
    │   ├── grpo_qwen2.5_1.5b.yaml  # 兼容别名
    │   └── pilot.yaml
    ├── src/alfworld_baseline/
    │   ├── agent.py
    │   ├── env_adapter.py
    │   ├── tool_registry.py
    │   ├── prompts.py
    │   ├── parser.py
    │   ├── validator.py
    │   ├── rewards.py
    │   └── datasets.py
    ├── templates/
    │   └── qwen25_alfworld_tool.jinja
    ├── scripts/
    │   ├── preflight.sh
    │   ├── smoke_test.py
    │   ├── run_pilot.sh
    │   ├── run_baseline.sh
    │   └── evaluate.py
    ├── data/
    ├── log/alfworld/v1/2gpu/{pilot_seed0,seed0}/
    └── outputs/alfworld/v1/2gpu/{pilot_seed0,seed0}/

不得写入图像恢复的 `old_verl_grpo/outputs/` 或 `old_verl_grpo/log/`。不得修改当前工作区
已有的图像修复代码和配置。新的 launcher 不得执行全局 `pkill`、`ray stop` 或不加范围的
端口清理。

## 3. 数据与环境

复用已下载的 ALFWorld source，不重复复制约 2.5 GB 数据。固定：

    ALFWORLD_DATA=/home/LXJ/Python_Projects/Agent_Lightning/contrib/recipes/envs/agl_envs/alfworld/alfworld_source
    train=/home/LXJ/Python_Projects/Agent_Lightning/contrib/recipes/envs/agl_envs/task_data/alfworld/train.parquet
    test=/home/LXJ/Python_Projects/Agent_Lightning/contrib/recipes/envs/agl_envs/task_data/alfworld/test.parquet

如需新目录中的数据入口，只建立软链接或使用绝对路径。preflight 必须检查
`base_config.yaml`、train/test parquet、`game_file`、`traj_data.json` 和 `game.tw-pddl`。

## 4. Baseline 定义

第一版实现“ALFWorld 文本环境 + 结构化动作工具”的标准 GRPO。环境仍然是 ALFWorld
text-only；模型不直接输出裸文本，而是通过结构化 `alfworld_action` 工具提交 ALFWorld
原生文本动作。这样可以复用当前项目 old-VERL 的多轮工具执行链路，同时不改变环境动作
语义。

```text
Qwen3.5-9B
  → Qwen3.5 tokenizer native chat template
  → alfworld_action tool call
  → Parser
  → Validator
  → ALFWorld admissible action / step
  → reward、done、tool response
  → GRPO update
```

工具调用 baseline 的**模型可见输出契约**固定为一个 Qwen XML 调用、一个参数（Parser
内部才会把它归一化为 `name`/`arguments` 字典）：

```xml
<tool_call>
<function=alfworld_action>
<parameter=action>
go to cabinet 1
</parameter>
</function>
</tool_call>
```

第一版训练配置为：

    Qwen3.5-9B（本地 `/home/LXJ/Python_Projects/Models/Qwen3.5-9B`）
    → ALFWorld text rollout
    → 每任务 8 条 trajectory
    → episode-level reward
    → group-relative advantage
    → clipped GRPO update

### 4.0 轨迹粒度与交互次数

这是**多步（multi-step）轨迹级 GRPO**，不是单步强化学习。这里的 `single prompt` 仅表示
每轮把当前任务状态组织成一条 user prompt，不表示一条轨迹只生成一次动作。每个
`alfworld_action` 工具调用对应一次 ALFWorld 原生 `env.step(action)`，环境返回新的
observation、admissible actions、reward 和 done 后，才进入下一轮模型生成。

单条轨迹的流程为：

```text
reset（不计作 action step）
  → 模型生成一次 alfworld_action
  → Parser/Validator
  → ALFWorld env.step（第 1 次交互）
  → tool response
  → 模型再次生成 alfworld_action
  → ...
  → done 或截断
```

当前计划参数给出的理论上限是 50 次环境交互：

- ALFWorld adapter `max_steps=50`；
- old-VERL `max_user_turns=50`；
- old-VERL `max_assistant_turns=50`。

实际次数取以下条件的最小值：环境提前返回 `done`、模型没有生成 tool call、Validator
失败后按策略终止、达到 50 次 turn，或累计生成 token 达到
`max_generated_response_length=4096`。因此 50 是硬上限，不是每条轨迹保证执行 50 次；
通常会因任务完成、模型停止或 token 预算提前结束。每个 task 的 8 条 rollout 是 8 条
独立的多步轨迹，不是 8 个单步样本。

实现验收的额外要求：`ALFWorldTool.execute()` 返回的 `done/truncated` 必须在 ALFWorld 专用
agent-loop 桥接中转换为 `TERMINATED`，避免环境已经完成后仍继续生成并调用 `env.step()`。
在该桥接通过前，50 只能称为配置理论上限，不能作为实际交互次数承诺。

固定模型：`/home/LXJ/Python_Projects/Models/Qwen3.5-9B`（当前本机未发现 Qwen3.5-7B）。固定环境：text-only、
single prompt、多轮 tool interaction、`max_steps=50`。第一版不加入 action rarity、skill retrieval、AHEAD/OPSD/
SDAR/RLSD、GiGPO、PPO critic、额外 reward model 或图像恢复逻辑。

建议初始参数：2 GPU、tensor model parallel 2、learning rate `1e-6`、150 training steps、
单次 `seed0`。这一次训练即可作为当前 baseline；只有在需要论文均值、方差和随机性分析时，
再额外运行 `seed1/seed2`。实际配置必须遵循该定制后端已有 Hydra/VERL schema。

为保持 reward 可解释，建议新 baseline 使用：

    format_penalty: -0.05
    invalid_action_penalty: -0.05
    unknown_tool_penalty: -0.05
    no_tool_call_penalty: -0.05
    malformed_tool_call_penalty: -0.05
    reward_scale: 1.0
    use_success_rate: false
    algorithm.adv_estimator: grpo
    algorithm.use_kl_in_reward: false
    algorithm.use_final_reward_as_step_reward: true
    algorithm.use_intrinsic_reward: false

若适配层暂时需要旧字段 `reawrd_scale`，必须在 README 和结果中注明，不能称为完全论文复现。

### 4.1 工具调用组件与职责

五个组件必须在新目录中独立实现，并通过单元测试固定接口；不得直接复制图像修复模块中
与视觉输入、`restore_image` 或 Qwen3.5 专用 token 相关的逻辑。

1. **ToolRegistry（`tool_registry.py`）**

   - 定义唯一函数名 `alfworld_action` 和唯一参数 `action`。
   - 每次环境 `reset` 或 `step` 后，根据当前 `admissible_commands` 生成当前状态的 schema。
   - `action` 使用字符串类型；可将当前 admissible actions 放入 `enum`，但 Validator 始终是
     最终权威，不能假设 vLLM 会执行 JSON-schema 级别的约束解码。
   - 保留原始动作字符串，不做大小写、空格或标点归一化，建立 schema 动作与环境动作的一一
     对应关系。
   - 提供 `build_tool_schema()`、`available_actions()`、`validate_action()` 和
     `to_runtime_action()`，并拒绝空动作、未知动作和重复字段。

2. **Prompt（`prompts.py`）**

   - system prompt 明确要求恰好调用一次 `alfworld_action`，不得输出裸文本、Markdown 或第二个
     工具调用。
   - user prompt 包含任务目标、当前 observation、历史 action/tool response 和当前
     admissible actions；不暴露 `game_file`、隐藏标签或数据集路径。
   - 每轮只注入当前状态；工具 schema 在首轮由 chat template 序列化，后续轮次按 old-VERL
     上下文协议更新工具集合，避免重复拼接 system 消息。
   - 动作必须逐字匹配当前 admissible action；首版不额外引入 `stop` 工具，以免改变 ALFWorld
     原生动作空间。达到环境 `done` 或 `max_steps` 时由环境循环终止。

3. **Qwen3.5 原生 chat template**

   - 直接使用 Qwen3.5 tokenizer 自带模板，格式为 `<tool_call>/<function>/<parameter>`。
   - 模板负责渲染 system/user/tool 消息、当前工具 schema、`<tool_response>` 和 generation
     prompt，并保证 rollout tokenizer、actor/reference 重 tokenization 使用完全相同的文本边界。
   - 不通过 `model.custom_chat_template` 注入旧 Qwen2.5/Hermes JSON 模板，避免 rollout 与 actor/reference
     使用不同的消息边界。
   - preflight 必须输出模板渲染文本摘要、token 数和 SHA256；同一条样本在 rollout、actor、
     reference 三处的 prompt hash 必须一致。

4. **Parser（`parser.py`）**

   - 使用 old-VERL 注册的 `qwen3_coder` parser，解析 Qwen3.5 原生 XML；不修改共享 parser 行为。
   - 同时支持 vLLM 已解析的 `message.tool_calls` 和原始文本中的
     `<tool_call><function=...><parameter=...>` XML，两个入口最终转换为同一个内部结构：
     `name`、`arguments`、`raw_text`、`tool_call_id`。
   - 明确区分空响应、JSON 语法错误、函数名错误、多工具调用和参数类型错误，供日志与终止控制
     使用。

5. **Validator（`validator.py`）**

   - 检查函数名必须为 `alfworld_action`，arguments 必须是对象且只能包含 `action`。
   - 检查 action 为非空字符串，并逐字存在于本轮 admissible actions；非法动作不调用环境。
   - 合法时调用环境 `step(action)`；统一返回
     `observation, executed_action, is_valid, step_reward, terminated, truncated, info,
     available_actions_hint`，与 AGL adapter 和 EnvAgent 契约一致。
   - 非法输出记录稳定原因码：`EMPTY_RESPONSE`、`INVALID_JSON`、`UNKNOWN_FUNCTION`、
     `MISSING_FIELD`、`UNKNOWN_ACTION`、`MULTIPLE_TOOL_CALLS`；这些事件现在按 ALFWorld 隔离配置施加一次 `-0.05` 协议惩罚，并按原因计入 SwanLab 的 `alfworld_penalty/*` 指标；原生环境 reward 仍单独保留。

ToolRegistry、Prompt、Template、Parser、Validator 的职责必须保持分离：Prompt/Template 提供
协议和上下文，Parser 负责结构解析，Validator 负责环境动作契约；不能把 Validator 的动作
白名单硬编码到 prompt，也不能把 parser 当作约束解码器。

### 4.1.1 old-VERL 后端隔离边界

ALFWorld 需要的后端扩展仅通过隔离配置和外部 AgentLoop 注册完成：

- `config/agent_loops.yaml` 注册 `alfworld_tool_agent`；
- `config/alfworld_tool_config.yaml` 注册 `ALFWorldTool`；
- `config/grpo_qwen2.5_1.5b.yaml` 仅覆盖 ALFWorld 的 `default_agent_loop`、
  `agent_loop_config_path`、`tool_config_path`、`multi_turn.format` 和本地数据/输出路径；
   - Qwen3.5 tokenizer 原生 `chat_template` 由 old-VERL 模型加载路径直接使用；ALFWorld 不覆盖共享模板；
- `data/*.parquet` 通过 `data_source=alfworld`、`agent_name=alfworld_tool_agent` 和
  `extra_info.tools_kwargs` 把任务传给工具实例。

不得为 ALFWorld 修改共享 `tool_agent_loop.py`、共享 reward manager、共享 tokenizer/parser
注册表或图像修复配置。任何确需修改共享后端的 bugfix，必须先建立等价的 ALFWorld 隔离
覆盖、补充图像修复回归测试，并单独记录影响评估。

当前已实施的隔离扩展：`src/alfworld_baseline/agent_loop.py` 继承共享 `ToolAgentLoop`，
只在 `data_source=alfworld` 时将工具 metrics 中的 `done/truncated` 映射为
`AgentState.TERMINATED`；图像修复仍走原有 `tool_agent`。后续若需修改 VERL 的 dataset、
   reward、parser 或 tokenizer 行为，也必须优先采用同样的外部扩展/独立配置方式，并把共享
后端改动视为需要单独回归的例外。

### 4.2 Baseline 与原生文本动作的关系

本计划的主 baseline 是“结构化工具调用版”，论文对照时必须明确标注为
`ALFWorld-text + alfworld_action tool wrapper`。另保留一个不使用 ToolRegistry/Parser 的
原生裸文本 smoke/evaluation 分支，作为接口回归和消融对照；两者不得合并统计，也不得把
工具格式有效率与环境任务成功率混为一个指标。

## 5. 实施阶段

### A. 隔离代码骨架

在新目录实现独立 ALFWorld adapter、EnvAgent、ToolRegistry、Prompt、Qwen3.5 原生 chat template、
Parser、Validator、single-turn/multi-turn prompt builder、parquet loader、reward adapter、
preflight、smoke test、pilot 和正式 launcher。可以参考 `contrib/recipes/envs`，但不得直接
使用其中会清理全局进程的 `train_env_agent.py`。

### B. Preflight

不启动模型、Ray 或训练，只检查 Python/VERL 后端路径、版本、git commit、vLLM/Ray/PyTorch/CUDA、
GPU、模型文件、ALFWorld 数据、Hydra 组合结果和独立输出路径。额外检查：

- Qwen3.5 tokenizer 可加载，原生模板可渲染；兼容模板（如使用）也必须编译为 Qwen XML；
- ToolRegistry 可由一组 admissible actions 生成 schema；
- Parser 可解析一条合法 Qwen3.5 XML tool call 和至少五类非法输出；
- Validator 对合法动作放行、对未知/空/重复字段拒绝；
- 模板渲染结果与 old-VERL `apply_chat_template` 的 token/hash 契约一致；
- 配置中的 `multi_turn.format`、`tool_config_path`、`chat_template` 与实际 parser/template
  版本一致。

### C. 环境与 AGL smoke test

分别验证
`reset → observation → admissible actions → ToolRegistry schema → Prompt/template → 合法
tool call → Parser → Validator → reward/done → tool response → close`，以及
`make_env_manager → EnvAgent → 同一套 parser/validator → step 八元组 → close`。默认 3～5 步，
并测试 `max_steps=1` 截断、未知动作、空响应、裸文本和多工具调用。

### D. Pilot

使用 32～64 条训练样本、20～32 条验证样本、2～4 rollouts、3～5 training steps，输出到
`runs/pilot/`，禁用 resume。只验证模型加载、vLLM rollout、多轮环境、工具调用解析、动作
校验、reward、advantage、loss、checkpoint 和退出清理。

### E. 正式 baseline

pilot 通过后，以 Qwen3.5-9B、2 GPU、8 rollouts/task、50 max steps、1e-6 learning
rate、150 steps 分别运行 seed 0、1、2。每个 seed 独立输出、日志、checkpoint 和评估目录，
禁止自动读取其他运行的 checkpoint。

## 6. 评估与完成标准

正式 GRPO 前先跑 zero-shot。每个 checkpoint 报告总成功率、Pick/Look/Clean/Heat/Cool/Pick2、
平均交互步数、最大步数截断比例、工具调用解析有效率、动作校验有效率、实际环境 step 比例
和 episode reward；训练过程记录 `critic/rewards/mean`、`critic/advantages/mean`、
`actor/entropy`、prompt/response length 和成功率曲线。

验收报告必须分别给出：

- `tool_call_parse_valid_rate`：输出能否被 Parser 解析；
- `tool_action_valid_rate`：解析后的 action 是否通过 Validator；
- `environment_step_rate`：通过校验并实际执行的比例；
- `task_success_rate`：ALFWorld 最终成功率。

四项指标不能用一个“格式正确率”代替，否则无法区分模型格式问题、动作选择问题和环境本身
的问题。

只有在 3 个 seed 均从同一本地模型独立启动、确认使用项目内 VERL、无隐式 resume、
ToolRegistry/Prompt/template/parser/validator 链路通过、合法工具调用率和动作校验结果可追踪、
reward/advantage/loss 无 NaN/Inf、checkpoint 可重载并完成完整 test 评估，且所有新产物位于
`examples/alfworld_grpo_baseline/` 时，才认为 baseline 完成。

## 7. 论文比较限制

SAPO 使用 Qwen2.5-1.5B/7B-Instruct；AHEAD 使用 Qwen2.5-3B/7B-Instruct 和 Qwen3-1.7B-Instruct。
当前 baseline 与 SAPO 的 Qwen2.5-1.5B GRPO 结果最接近，但不能直接声称完全复现。比较前需
对齐模型、split、prompt、工具包装方式、最大步数、rollout 数、reward、训练步数和 seed。
