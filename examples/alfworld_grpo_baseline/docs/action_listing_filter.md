# 跳过 thinking 中的候选清单复述

2026-10-20，`新动作熵1019` 将 branch entropy 的位置选择由 `first`
改为 `first_non_listing`，系数仍为 0.0005。其他训练超参数及奖励不变；
思考动作单次/重复一致性奖励均为 0。历史配置未指定该选项时仍使用 `first`。

## 规则与边界

仅检查 thinking 内与最终 XML action 大小写一致、词边界完整的提及，
使用当前环境执行动作之前的合法动作集合。依次检查这些提及：

1. **多行清单**：存在 available/admissible actions、options 等可用性标题；
   连续的项目符号或编号列表中至少两项含合法动作；当前项仅含动作名、标点和连接词。
2. **行内枚举**：当前句子或包围该提及的括号内，至少出现两个不同合法动作；
   前导语说明可用动作，且动作之间/之后只有标点、and/or 等枚举连接词。
3. 规则确认的清单提及被跳过，在第一个保留的提及处构建原有合法动作 token trie。
   若后面仍是清单，继续跳过；全部提及都是清单则返回 `only_listing_mentions`，
   不回退到清单或 XML，也不改变该输出原有 GRPO 奖励、KL 和普通 token 熵损失。

不会仅凭 available 或项目符号判定。带解释、优劣分析、选择理由、单项可用性、
执行计划或不规则缩写的模糊文本默认保留。因此这是一种保守过滤，不是完整语义分类器，
更不声称定位到了模型内部真正做出决定的时刻。尤其 `look`/`inventory` 的普通词义、
之前已经隐含形成动作偏好等情况仍沿用原有精确字符串匹配的局限。

原始 rollout token IDs 不变；分叉 logits 的前移一位、候选集合、熵计算、轨迹权重不变。
仅改变选择哪次提及进行原有对齐。对齐不稳定仍使用既有跳过状态。

## 回放证据

规则开发参考已有 step 11 的 180 条分层语义复核样本，并非独立测试集：
已标注的清单复述共 26 条，规则识别 14 条，漏掉 12 条；154 条非清单中无命中。
这不是总体精确率保证，样本经过分层，不能直接外推未加权的召回率/命中率。
漏掉的典型情况包括带一般功能解释的列表项、紧凑编号和不规则枚举。

对 step 11 全量文本回放（6,169 条输出）结果：

| 项目 | 数量 |
| --- | ---: |
| 最终动作合法且 thinking 有完整提及 | 5,951 |
| 至少跳过一次清单提及的输出 | 658 |
| 改用后续提及 | 648 |
| 仅有清单提及，跳过该项熵正则 | 10 |
| 总计跳过的提及 | 666 |

这是文本位置回放，不是原始 token 对齐成功率（dump 不包含原始 token IDs）。
规则识别量占合格输出 11.06%；与先前约 19.3% 的语义抽样估计不同，二者不可混用。
另抽查规则命中的 15 条原文，未发现明显的任务相关选择理由被当成纯列表项。

可复现命令（在 alfworld-verl 环境中，设置本项目 src 的 PYTHONPATH）：

```bash
python examples/alfworld_grpo_baseline/scripts/audit_action_listing.py \
  examples/alfworld_grpo_baseline/outputs/qwen3.5_2B/新动作熵1019/rollouts/11.jsonl \
  --output /tmp/action_listing_audit
```

续训将覆盖同编号 rollout 文件；原始旧规则 step 11/12 已备份在
`outputs/qwen3.5_2B/新动作熵1019/setup/listing_filter_resume_20261020/previous_rollout_*.jsonl`。
复现旧批次分析时应使用这份备份。逐条复核源文件保存在
`outputs/diagnostics/first_mention_listing_20261020/`。

## 监测与续训

每条分叉 metadata 包含 `mention_count`、`selected_mention_index`、
`skipped_listing_count` 和 `skipped_listing_rules`。SwanLab 新增：

- `alfworld_branch/listing_skipped_row_count`
- `alfworld_branch/listing_skipped_mention_count`
- `alfworld_branch/later_mention_selected_count`
- `alfworld_branch/only_listing_mentions_count`

继续同时观察 `match_fraction` / `eligible_fraction`，不要把文本位置选择成功
等同于实际有合法 token 分叉可正则。

此次恢复自最新完整 `global_step_10`，保留模型、optimizer、scheduler/RNG 和 dataloader
恢复机制；同一输出目录、同一 SwanLab ID `2mbeiini`，`resume="must"`。
停止时旧进程已记录 step 12，但 step 11/12 尚未形成 checkpoint。
SwanLab 0.9 按指标拒绝写入已记录的步数，因此既有指标的旧 step 11/12 保留，
重放后从更大的步数继续追加；新增列表指标没有历史点，可记录重放步数。
不删除云端历史，也不人为移动训练 global step。
