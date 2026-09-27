# GiGPO 原生 GRPO / Qwen3.5-2B

此入口运行 GiGPO `20bd331bdbc9026a5668e11362178e10ab7400c8` 自带的
`verl.trainer.main_ppo.TaskRunner`。它独立于已有 trajectory 和 gigpo_grpo 风格后端。

```bash
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_gigpo_native_grpo.sh --preflight
bash examples/alfworld_grpo_baseline/scripts/alfworld/alfworld_gigpo_native_grpo.sh
```

默认在后台运行，启动器打印 PID 和日志位置。完整配置：
`config/alfworld/qwen35_2b/gigpo_native/alfworld_config_2gpu.yaml`。
`ALFWORLD_FOREGROUND=1` 可前台运行，其他参数使用 Hydra override。

## 来源与隔离

`scripts/prepare_gigpo_native.py` 从本机 `/home/LXJ/Python_Projects/GiGPO`
创建固定提交的独立 worktree `.runtime/gigpo-native`，应用
`patches/gigpo_native_qwen35.patch`。不修改原始 GiGPO 工作区。
启动器独立设置 PYTHONPATH，不混入 native VERL V1。
prepare 每次检查奖励管理器、优势计算、PPO trainer、actor loss/update 和环境
执行文件与该提交逐字节一致，记录 SHA256 清单。

## 保持的算法行为

- 成功环境奖励 10，失败 0；每个实际执行的决策步骤获得整条轨迹的环境累计奖励。
- 当前步骤格式错误时，由原版 `apply_invalid_action_penalty` 扣 0.1。
- 同任务组内所有决策行一起计算 GRPO 均值和样本标准差。
- 沿用原版环境随机种子分组、50 步、历史长度 2、采样和 batch 调整流程。
- 不添加重复惩罚、动作可执行性惩罚、全失败组过滤或其他奖励 shaping。
- 原版旧策略 log-prob 重算、冻结 reference、KL/熵损失及 PPO 更新不变。

## 适配范围

- 原有 `prompts_gigpo.build_user_prompt` 和聊天模板原样复用，保留 XML 调用。
- `parse_xml_decision` 判断是否有且仅有一个合格的 alfworld_action 调用。
  生成前缀已经提供 `<think>`，不要求模型再生成一次；必须关闭 `</think>`。
- 上游禁中文条件保留；无动作/错误结构统一映射到一个格式错误标记。
- 和上游一样，不检查 action 是否属于 admissible 列表。解析结果或错误输出
  的末 30 个字符仍交给环境执行；不会直接跳过环境步。
- 当前合并 SFT checkpoint-100 同时初始化 actor/reference，LoRA 关闭。
- Qwen3.5 使用新版 Transformers/SGLang 依赖覆盖层；使用原生 padded HF
  forward，避免旧版 unpadding 补丁破坏 recurrent layer 的序列边界。
- 生成 token 的 mask 按 SGLang 返回的实际采样长度构造，适配 Qwen3.5 多个
  stop token，防止把 padding 当成有效回复；padding 的 log-prob 填 0。
- SGLang 选择 Triton attention 和 PyTorch sampling，复用本机已经用于 Qwen3.5
  的旧名称 `sgl-kernel`，以实际端到端运行验证兼容性，不替换共享安装。
- GPU micro-batch 降为 2，并按原版梯度累积；有效 PPO mini-batch 仍为 256。
- 保存频率设为 10（原版关闭保存），加入本地完整轨迹、指标和 SwanLab 记录。

## 文件布局

- checkpoint、指标和实验身份：`outputs/qwen3.5_2B/gigpo_native_grpo/full_sft/`
- stdout：`log/alfworld/qwen35_2b/gigpo_native_grpo/full_sft/2gpu/seed0/`
- 完整训练轨迹：输出目录的 `rollouts/trajectories/`
- 完整验证轨迹：输出目录的 `validation/trajectories/`
- 原版训练 step 行：输出目录的 `rollouts/steps/`
- SwanLab 本地文件：输出目录的 `swanlab/`
- Hydra 配置快照：输出目录的 `hydra/`
- Ray 运行时：`/home/LXJ/tmp/ray-gigpo/`（短路径，满足 UNIX socket 长度限制）

原版占位数据仅决定 text 模态及 batch 数；任务由其环境 reset 选择。
本入口在 `data/gigpo_native/` 本地生成等价占位 parquet，不下载无关图形题库。
训练 16 个任务 × 8 条轨迹，验证 128 条；验证温度 0.4，训练温度 1。

## 测试

```bash
ROOT="$PWD/examples/alfworld_grpo_baseline"
ALFWORLD_VERL_ROOT="$ROOT/.runtime/gigpo-native" \
PYTHONPATH="$ROOT/src:$ROOT/.runtime/gigpo-native:$PWD/examples/image_restoration_multi_agent/old_verl_grpo/.pydeps" \
/home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest -q "$ROOT/tests/test_gigpo_native.py"
```

覆盖原有提示词一致性、XML 判定、重复及不可执行动作的边界、原版奖励和跨步优势数值。
`last_completed_step.json` 仅在 trainer 报告 actor 梯度指标后更新；不能用进程启动
或单次 rollout 成功代替真正的训练 step 完成。

原版 FLOPs 估计器不认识 Qwen3.5，MFU 会显示 0；这不代表 GPU 没有工作。
原版 `val/text/test_score` 是展开后的步骤行平均分，受轨迹长度影响；任务成功率
应读取 `val/success_rate`，不能把前者除以 10 当作成功率。
