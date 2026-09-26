# Qwen3.5-2B 全参数 GRPO 吞吐调优（2026-09-26）

硬件为 2 × A800 80GB（每卡实际 79.25 GiB），使用原生 veRL V1 trajectory GRPO。
SFT 基底仍为 checkpoint-100 合并模型，LoRA 关闭。保持 16 个任务 × 8 条轨迹、
全局 PPO mini-batch 128 条 step 行、学习率 1e-6、KL 0.001、熵系数 0.005。
本次只调整执行参数，不减少训练数据或环境交互步数。

## 同样本比较

从调优前运行的第 2 步取得真实 TensorDict（6235 条有效 step 行），按长度选出
256 条代表样本及最长的 128 条。每次重新加载相同 SFT 模型，使用原生 Actor/ref
worker 在两卡上执行：预热、128 条代表样本、128 条长样本，再检查更新后的参考模型。
Actor 每次执行实际的前向、反向及优化器更新，并检查 loss/grad_norm 有限。
额外保留每卡 3 GiB，模拟训练期间休眠的 SGLang 进程。

下表秒数均为一次全局 128 行的代表样本；显存为所有测试阶段的最大 PyTorch
已分配量（含上述 3 GiB），不是单看 `nvidia-smi` 的缓存占用。

| 配置 | Actor 秒 | 参考模型秒（更新后） | 峰值 GiB | 结果 |
| --- | ---: | ---: | ---: | --- |
| 原配置：Actor 2 行 / ref 4 行，CPU 卸载 | 28.94 | 10.58 | 18.15 | 基线 |
| Actor 8 行 / ref 32 行，保留重计算及卸载 | 12.87 | 2.19 | 31.52 | 通过 |
| Actor 16 行，关闭模型与熵重计算 | — | — | 71.97 + 申请 7.33 | OOM，排除 |
| 动态 Actor 8192 / ref 32768，关闭模型重计算 | 9.42 | 2.20 | 50.10 | 通过 |
| 动态 Actor 16384 / ref 32768，保留模型重计算、GPU 常驻 | 7.45 | 2.18 | 41.81 | 通过 |
| 动态 Actor 24576 / ref 32768，其余同上一行 | 7.16 | 2.19 | 50.92 | 通过 |
| 动态 Actor 32768 / ref 32768，其余同上一行 | 6.98 | 2.20 | 64.73 | 收益小、占用增加 |
| **动态 Actor 24576 / ref 65536，关闭熵重计算** | **7.08** | **1.89** | **50.92** | **选用** |

GPU 常驻各档均开启 `use_no_sync_for_gradient_accumulation`，并设置
`reshard_after_forward=false`。所有成功档位均完成长样本和更新后的参考计算；
最终档位长样本 Actor 为 9.88 秒，峰值 50.92 GiB。
32768 档仅比选用档快约 1.4%，却多用约 14 GiB，因此选择 24576。
不关闭模型梯度检查点；它是更大 token 批量能稳定运行的关键。

完整参数及原始数值汇总见 [JSON 记录](throughput_tuning_20260926.json)。
首次遇到新形状可能包含编译耗时，预热及长样本首次调用时间均保留在 JSON 中。
基线使用默认 CUDA allocator；后续使用与原生训练权重同步后相同的 expandable
segments，因此只比较已分配峰值，不用 allocator 的 reserved 值作跨档比较。

## 正式配置与验证

默认启动命令不变：

```bash
bash examples/alfworld_grpo_baseline/scripts/alfworld/qwen35_2b_v1.sh
```

选用参数已写入 `config/alfworld/qwen35_2b/v1/alfworld_config_2gpu.yaml`。
Actor 动态 token 上限 24576，参考模型 65536，rollout old-log-prob 重算上限与
Actor 一致（正常运行仍使用 bypass）。静态 micro-batch 值仅是关闭动态模式时的回退。
保留梯度检查点，关闭熵重计算和 Actor 参数/优化器卸载。冻结参考模型保持原生
FSDP CPU offload 行为，不能把 Actor 的卸载设置误认为参考模型也已常驻 GPU。

正式输出为 `outputs/qwen3.5_2B/full_sft_grpo_tuned`，从完整 SFT 模型新建训练，
不读取调优前 checkpoint。调优前运行的完整一步耗时 2181.50 秒，其中 rollout
200.79 秒、参考计算 538.05 秒、Actor 更新 1420.13 秒。
正式训练已完成第一步，并恢复下一轮 rollout 的权重及 KV cache。实测比较：

| 指标 | 调优前 | 调优后 |
| --- | ---: | ---: |
| 完整一步 | 2181.50 秒（36.36 分钟） | **674.61 秒（11.24 分钟）** |
| Rollout | 200.79 秒 | 196.77 秒 |
| 参考模型 | 538.05 秒 | **102.12 秒** |
| Actor 更新 | 1420.13 秒 | **353.05 秒** |
| Actor 峰值已分配显存 | 14.44 GiB | 51.83 GiB |
| Actor 峰值 reserved 显存 | 21.96 GiB | 60.97 GiB |
| Actor MFU | 9.10% | 37.99% |

端到端快 **3.23 倍**，Actor 更新快 **4.02 倍**，参考计算快 **5.27 倍**。
NVML 每 2 秒采样测得整卡峰值约 **65.3 GiB**，包含 SGLang、CUDA 和 allocator
缓存，仍有约 14 GiB 余量。GPU 在更新阶段持续接近 100% 计算利用率。
两次均为 128 条轨迹及 6272 条补齐后的 step 行；独立 rollout 的有效行数分别为
6184/6163，总 token 数 5521383/5494522，所以端到端数据不是逐 token 完全相同。
同样本微基准用于参数比较，完整训练用于验证实际收益。

新运行第一步 `actor/loss=0.0657103`、`actor/grad_norm=4.35730`、
`actor/kl_loss=0.00132325`、学习率 1e-6，全部指标有限；权重同步 1.40 秒。
日志确认两卡 Actor 各有 1106620832 个可训练参数，冻结参数为 0。
249 个 ALFWorld 回归测试全部通过。

后台主进程 PID 为 3205597；SwanLab 实验为
[qojf5pzo](https://swanlab.cn/@Godknows/ALFWorldRL/runs/qojf5pzo)。
日志为 `log/alfworld/qwen35_2b/full_sft/v1/2gpu/seed0/alfworld_v1_seed0_20260926_124934.log`。
完整首步验证记录保存在正式输出的 `first_step_verification.json`。

调试使用本地独立 worker，没有创建 SwanLab 云端实验。调试 tensors、日志和
其他临时文件在验证后删除，只保留本报告及指标汇总。调优前的原训练记录、
数据集和 SFT 模型保留。
