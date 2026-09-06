# ALFWorld Qwen3.5-2B KL/Entropy 调参记录（Prompt schema v4 新基线）

实验对象：ALFWorld GRPO v1，Qwen3.5-2B，2 GPU。

## 2026-09-06：Prompt schema v4 基线检查

- 运行：`attempt_006_schema_fix_kl0.00320_e0.02000`
- 完成到：`step=26`
- `actor/entropy`：前 5 步均值 `0.2253`，后 5 步均值 `0.3092`，范围 `0.2149–0.3245`
- `actor/kl_loss`：前 5 步均值 `0.0247`，后 5 步均值 `0.1387`，末步 `0.1546`
- `critic/rewards/mean`：前 5 步均值 `-0.0244`，后 5 步均值 `0.0138`，但波动较大，未确认持续上升
- `actor/tool_choice_entropy`：后 5 步均值约 `3.48`，没有归零
- 结论：没有熵爆炸，也未触发 `<0.1` 的硬熵坍缩阈值；但策略熵持续偏低，KL 漂移升高，奖励趋势不稳定，未满足“奖励持续升高”。按新 Prompt 基线重新开始调参。

## 本轮调整

| 时间 | 原因 | KL coefficient | Entropy coefficient | 说明 |
|---|---|---:|---:|---|
| 2026-09-06 | 新 Prompt 基线奖励不稳定、策略熵偏低 | 0.0032 → 0.0032 | 0.020 → 0.030 | 只提高 Entropy 正则，保留 KL 不变，扩大探索余量；不改变其他参数 |

## 清理与重启

- 已停止精确 systemd 任务 `alfworld-qwen35-2b-schema-v5.service`。
- 已删除旧实验目录及其中的 checkpoint、rollout、日志、本地 SwanLab：
  `outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_006_schema_fix_kl0.00320_e0.02000`
- 已删除旧调参文档后重新创建本文件；保留实验计划和 Prompt schema 文档。
- 已删除属于该实验的 Ray session：`session_2026-09-06_00-02-10_964674_2462654`。
- 已使用已安装 SwanLab SDK 对旧云端 run 执行删除并返回成功；云端 run 的最终删除状态应以随后查询为准。

## 新实验

- 输出目录：`outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_007_prompt_v4_kl0.00320_e0.03000`
- 启动脚本：`scripts/alfworld/qwen35_2b_v1.sh`
- 目标：先观察至少 5 个完整训练 step，再判断熵、KL、工具探索和奖励趋势。


## 2026-09-06：attempt_007 崩溃诊断与第二次调整

### 更正此前判断

attempt_006 的策略熵由 0.2253 上升至 0.3092，不能仅凭绝对值低认定熵坍缩；末 5 步奖励均值也已改善。
此前据此提高熵正则的证据不足。以下判断仅依据新 Prompt/replay 下的指标，不引用旧 Prompt 的调参曲线。

### 直接退出原因（与调参动机分开）

- systemd 在 **2026-09-06 05:13:04 +08:00** 以 exit code 1 退出；最后完成 step=47，下一批环境初始化失败。
- 调用链：ALFWorldTool.create → TextWorld PDDL 初始化 → fast_downward.load_lib → shutil.copyfile。
- 原始异常：`OSError: [Errno 28] No space left on device`，目标 `/tmp/tmp1h6h4woo/libdownward.so`。
- 当时写入 /tmp 失败是确定事实；09:36 检查时根分区已恢复约 87 GiB 可用，不能以当前空闲量否认过去的 ENOSPC。
- Fast Downward 每次加载都会在 Python 临时目录复制共享库。此前仅指定 RAY_TMPDIR，**不能改变 Python tempfile 的目录**。
- 发现 /tmp 下另有 HWJ 拥有的三个约 17 GiB 文件，未删除；无法据此认定是谁耗尽了当时磁盘。
- 本次重启使用 `/home/LXJ/tmp/alfworld-tune008` 作为 TMPDIR/TMP/TEMP；Ray 使用该目录下的 ray 子目录。
  这是启动环境的存储位置调整，不修改训练代码/算法/数据/Prompt/其他超参数。启动前验证真实 ALFWorld 环境可创建及执行动作。

### 指标窗口

| 指标 | step 1–5 均值 | step 38–42 均值 | step 43–47 均值 |
|---|---:|---:|---:|
| actor/entropy | 0.225444 | 0.508357 | 0.576329 |
| actor/kl_loss | 0.048742 | 0.295224 | 0.350668 |
| critic/rewards/mean | -0.027188 | -0.020313 | -0.005938 |

- 全程策略熵范围 0.206356–0.601382，未发现熵坍缩/爆炸或对应指标 NaN/Inf；末 5 步工具选择熵均值 3.382265。
- 验证 reward mean@1 在 step=30/40 均为 -0.00392857；该值不是成功率，不能把带 acc 名称的别名当真实准确率。
- 奖励相比开头有所改善但依赖少数正奖励批次，未持续稳定增长。KL 上升比增加探索所带来的收益更明确。
- KL/Entropy 改动不能修复磁盘 ENOSPC。本轮调参是保守试验，不能保证奖励持续上升，也不能由损失标量大小推断梯度贡献。

### 参数（只有两个值变更）

| 参数 | 原值 | 新值 | 原因 |
|---|---:|---:|---|
| actor_rollout_ref.actor.kl_loss_coef | 0.0032 | 0.004 | 增强参考策略约束，尝试控制持续漂移 |
| actor_rollout_ref.actor.entropy_coeff | 0.03 | 0.02 | 回退上轮缺少充分证据的增幅，保留熵正则而不继续鼓励随机性 |

模型/Prompt/replay/奖励/学习率/种子/批量/采样数/训练步数/GPU 不变，不恢复失败 checkpoint。

### 操作记录

- 旧 run：attempt_007_prompt_v4_kl0.00320_e0.03000；云端 run：bm7ss2uu。
- 清理与首个训练更新结果将在验证后补记。保留本节诊断摘要，不保留失败 checkpoint/rollout/原始日志。
- 新 run：attempt_008_prompt_v4_kl0.00400_e0.02000。
- 单元：alfworld-qwen35-2b-tune-v2.service。
- 后续手动启动脚本前应设置 TMPDIR/TMP/TEMP/RAY_TMPDIR 到可写且容量充足的专用目录；仅设置 RAY_TMPDIR 不够。

### 09:40 左右启动前核验

- 旧 unit 已停止，主 PID 不存在，GPU 查询无计算进程。
- 旧 run 目录和对应 Ray session 均已删除并验证路径不存在。
- SDK 删除 bm7ss2uu 返回 True；随后云端项目列表 total=0，确认旧 run 已删除。
- 实际 ALFWorld 环境在新 TMPDIR 创建成功，并成功执行一个合法动作（初始 55 个可用动作）。
- 新 TMPDIR 所在盘可用约 3160 GiB；19 项测试通过。当前空闲量不保证整个训练永不耗尽，仍需观察增长。

### 本轮启动环境（脚本内容未修改）

```bash
mkdir -p /home/LXJ/tmp/alfworld-manual
TMPDIR=/home/LXJ/tmp/alfworld-manual \
TMP=/home/LXJ/tmp/alfworld-manual \
TEMP=/home/LXJ/tmp/alfworld-manual \
RAY_TMPDIR=/home/LXJ/tmp/alfworld-manual \
LD_LIBRARY_PATH=/home/LXJ/anaconda3/envs/alfworld-verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/alfworld-verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda/lib64 \
bash /home/LXJ/Python_Projects/Agent_Lightning/examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh
```

不要与已运行的训练重复启动；正式新实验应另外指定唯一 ALFWORLD_OUTPUT_DIR/ALFWORLD_LOG_DIR/ALFWORLD_SWANLAB_LOG_DIR。
本轮专用目录为 `/home/LXJ/tmp/alfworld-tune008`，上例是下一次手动启动的示意；不要删除仍在使用的目录。


### 2026-09-06 09:47 +08:00：真实训练更新验证

- 参数提交：`a092cccc`；后台主 PID：3184923；云端 run：vdy2vkeb。
- 日志已出现 **step=1 / 150**，包含 rollout、old logprob、reference logprob、actor 更新和权重同步；不是仅初始化成功。
- 首步 entropy=0.205583，KL loss=0.00071814，KL coeff=0.004，reward mean=-0.01875，工具选择熵=3.923626，grad norm=12.75；数值均有限。
- old logprob 耗时约 21.56 秒，reference 25.66 秒，actor 更新 138.44 秒。
- 已读取主进程、TaskRunner、Actor worker、AgentLoop worker 的环境，确认 TMPDIR/TMP/TEMP/RAY_TMPDIR 均指向新专用目录。
- 当前仍在后台训练；首步成功不代表已证实长期稳定或奖励持续改善。至少在 5–10 个完整更新后比较同一 Prompt 下的窗口，不因单点熵偏低再次自动加大正则。
