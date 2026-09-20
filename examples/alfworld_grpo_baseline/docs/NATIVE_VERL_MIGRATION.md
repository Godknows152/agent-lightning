# 原生 veRL 迁移：实现完成，验证范围为 CPU

后端为 `/home/LXJ/Python_Projects/verl`，本轮没有修改该仓库，也没有修改图像修复后端。
ALFWorld 入口、AgentLoop、统计、续训身份适配均留在本目录。

## 数据与训练语义

- 每次决策使用任务目标、完整轨迹步数、最近两步观察/动作、当前观察及合法动作列表，重新生成 prompt；不把旧 reasoning 放回上下文。
- AgentLoop 返回原生 `list[AgentLoopOutput]`。每项保留该步真实 prompt IDs、原始生成 IDs、response mask 和 rollout log-prob。不重新编码生成结果、不裁剪多余 XML，也不再提供 `alfworld_turn_contexts`。
- 完整轨迹结束后结算一次奖励，再给步骤样本同一个轨迹分数。成功 +10，无动作 -5 并终止，非法动作 -0.1 并消耗决策步；重复动作按全轨迹累计，成功轨迹豁免重复惩罚。
- 原生 V1 TransferQueue 使用 `uid_session_index` 标识步骤。原生 `compute_advantage_for_multi_trajectories` 只取每个 session 的最后一步进行 GRPO 组内归一化，然后将优势广播至所有步骤。长轨迹不会因步骤多而重复进入归一化。
- Actor/ref log-prob 与优化使用原生 FSDP。采用原生 `token-mean` 损失聚合：较长轨迹有更多参与训练的 token，这与“每条轨迹等权”不同。没有额外的 ALFWorld FSDP 展开/还原或梯度回放。
- ALFWorld 成功率、终止原因、有效动作和惩罚计数只取轨迹最后一步，排除 padding。veRL 通用样本/长度指标仍描述步骤行。

保留最近两步会限制长程记忆；同时，全轨迹重复动作惩罚可能涉及模型当前看不到的旧动作。
这是保留原有任务定义后的部分可观测性，不是训练 prompt 不一致。没有擅自扩大历史窗口或修改惩罚规则。

## 配置边界

当前支持同步 V1、GRPO、环境奖励、FSDP/FSDP2。入口在启动 Ray 前拒绝异步模式、GAE、reward KL、reward model 和尚不支持所有步骤的蒸馏组合。

三个正式 profile 当前均选共享工具配置：最多 50 次决策，每步最多 768 个生成 token。
`alfworld_tool_config_qwen35_2b.yaml` 是可显式选择的 16 步变体；它不是当前默认。
启动入口按工具预算设置 response capacity 为**单步**预算，不再设置成步数乘以 token 上限。
Prompt 超出配置上限直接报错，不静默截断任务/历史。

已删除原生后端不读取的旧开关，包括 `use_separate_lora_reference`、`force_full_sleep_for_lora` 和旧 AgentLoop 路由开关。
**Qwen3.5-2B 仍从配置的 SFT adapter 初始化 Actor；LoRA 的 KL 参考使用原生 veRL 关闭 adapter 后的基础模型，不是独立冻结的 SFT adapter。**
因此迁移后的实验不应视为与旧参考策略完全相同的训练实验。

## 入口与续训

- 测试和启动入口拒绝混入其他目录的 `verl.*` 模块；使用原生 V1 runner/trainer/manager 接口。
- Ray 运行时设置位于原生入口读取的顶层 `ray_kwargs`。
- 续训前检查 data.pt、所有 rank 的 model/optim/extra_state 和 fsdp_config.json；残缺 checkpoint 或缺失发布标记时明确失败。
- SwanLab 首次初始化后立即原子保存 `.swanlab_experiment.json`。续训显式传原 run ID 和 `resume=must`，检查 checkpoint 与输出目录身份一致，禁止悄悄创建替代实验。
- 适配仅在 ALFWorld runner 进程的上下文内包装 SwanLab 初始化，退出时恢复；不改动 veRL Tracking 源码。
- 文件完整性和 SDK 参数已用 CPU 测试验证；旧 checkpoint 在新后端上的真实分布式载入、优化器恢复与数值连续性没有在本轮验证。

## 依赖与复现

V1 所需 TransferQueue 已安装，版本固定为 `fc33c979db6bd661802f6af458e75e850055fb20`。
最终环境保持原有 numpy 1.26.4 和 setuptools 70.2.0。仓库级 uv override 会覆盖显式
版本约束，必须使用 `--no-config`（本轮已恢复被该 override 临时升级的两个包）：

```bash
uv --no-config pip install --python /home/LXJ/anaconda3/envs/alfworld-verl/bin/python \
  -r examples/alfworld_grpo_baseline/requirements-native.txt
```

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=examples/alfworld_grpo_baseline/src:/home/LXJ/Python_Projects/verl:examples/image_restoration_multi_agent/old_verl_grpo/.pydeps \
/home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest \
  examples/alfworld_grpo_baseline/tests -q
```

最终完整回归：**189 passed，0 failed，0 skipped**；三个 profile 配置预检和运行时预检均返回 0。
测试后核对原生 veRL HEAD 为 `3efe38c7`，工作树保持干净。

测试夹具禁止 CUDA 初始化。测试覆盖真实 Qwen tokenizer/processor 与 AgentLoop 构造、TextWorld 环境、原生队列步骤序列化、变长轨迹 GRPO、padding、原生 PPO loss 的 CPU 反向传播/优化器更新、奖励/指标、入口与续训身份。
队列传输用内存替身，未启动 Ray 服务、模型推理服务或 GPU；不将 CPU 优化器测试等同于 FSDP 分布式测试。

三个 profile 的配置/入口预检以及本地依赖、模板、编译器预检也仅在 CPU 执行。
当前环境进程退出时可能出现 multiprocess 的 `_recursion_count` 清理告警；这是现有 Python 3.12 环境依赖问题，与测试断言失败分开记录。

本轮没有进行 GPU smoke、正式训练、真实 FSDP checkpoint 保存/恢复或 SwanLab 云端续接验证。
