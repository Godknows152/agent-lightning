# ALFWorld Qwen3.5-2B thinking 重启记录

日期：2026-09-06（Asia/Shanghai）。用户授权停止当前实验、删除产出，并保持现有参数重启。

## 清理

- 旧 unit：`alfworld-qwen35-2b-tune-v4.service`，MainPID 278007 已退出。
- 停止后 GPU 查询已无该实验的 WorkerDict/SGLang 进程；其他用户 GPU 作业保持运行。
- 已删除并检查不存在：
  - `outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_010_prompt_v4_xml_example_kl0.00400_e0.02000`
  - `/home/LXJ/tmp/alfworld-tune010`
- 前一目录包含 checkpoint、rollout、训练日志和本地 SwanLab。
- 云端 `Godknows/ALFWorldRL/waija96v` 通过 SDK `Project.delete_runs` 删除成功，删除后云端列表为空。
- 未删除源码、模型、数据集、历史说明文档或其他实验。
- `/tmp/alfworld-waija96v-{metrics,series}.json` 为旧查询导出；额外删除命令被执行环境拒绝，未宣称已删除这些缓存。

## 新实验

- unit：`alfworld-qwen35-2b-thinking-v1.service`
- MainPID：778903
- 输出：`outputs/alfworld/qwen35_2b/v1/2gpu/adaptive/attempt_011_thinking_4096_kl0.00400_e0.02000`
- 日志：上述目录下 `log/training.log`
- TMPDIR/TMP/TEMP/RAY_TMPDIR：`/home/LXJ/tmp/alfworld-tune011`
- 启动入口：`scripts/alfworld/qwen35_2b_v1.sh`
- 配置代码提交：`d6294fe9`（thinking 支持）、`34879175`（4096 预算）。
- 保持参数：thinking=true；data.max_response_length=4096；max_generated_response_length=4096；KL=0.004；Entropy=0.02；GPU=0,1；seed=0；150 updates；resume_mode=disable。
- 未更改配置，未从旧 checkpoint 恢复。
- Qwen3.5-2B YAML SHA256：`148dec7c98cfcfdbe1866e1716aa3f2a8e9a457e98ba42c5ef2ca99e899158e5`。

## 运行核验

运行时打印配置已确认 thinking=true、两个生成容量=4096、KL=0.004、Entropy=0.02、resume_mode=disable。
等待真实 rollout、old/reference logprob 和首个 actor update。
