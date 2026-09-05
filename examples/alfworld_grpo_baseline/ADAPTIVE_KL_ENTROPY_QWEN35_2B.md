# ALFWorld Qwen3.5-2B 自适应 KL/Entropy 记录

监控对象：ALFWorld GRPO v1，Qwen3.5-2B，2 GPU。每 30 分钟检查一次；只允许修改 `actor.kl_loss_coef` 与 `actor.entropy_coeff`。

判据：连续窗口中 `actor/entropy < 0.1` 或工具探索归零视为熵坍缩；连续窗口 `actor/entropy > 12` 视为熵爆炸；奖励要求最近窗口中位数高于前一窗口。检测到目标未满足时，删除失败尝试的本地产出和 SwanLab 记录，更新参数，提交 git 后重新启动。

| 时间 | 原因 | 调整前（KL / Entropy） | 调整后（KL / Entropy） | 上一轮效果 |
|---|---|---|---|---|
| 2026-09-05 17:04:49 +0800 | bootstrap | 0.005 / 0.006 | 0.004 / 0.0069 | step=None, reward=None, entropy=None, tool_entropy=None |
