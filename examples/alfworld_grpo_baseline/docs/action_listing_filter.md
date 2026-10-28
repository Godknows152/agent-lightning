# 跳过 thinking 中的候选清单复述

**后续重启（2026-10-20 17:11）**：用户要求将预热起点从 0 改为 `1e-6`，
使用 20% 步数线性升至 `3e-6`。已再次删除本轮输出及 SwanLab `gr33p0s0`，
从同一基底模型后台重新启动；branch entropy 仍为 `0.001`，定位规则保持不变。
学习率调用链与边界测试见 `entropy_experiment_profiles.md`。

**此前重启（2026-10-20 11:26）**：按用户要求清理当前及此前同名训练后，
branch entropy 系数改为 **0.001**，从原基底模型重新后台启动，
包含本文的功能说明过滤与大小写兼容规则；其余训练超参数不变。
旧 checkpoint、采样、诊断产出、日志和 SwanLab `2mbeiini` 已删除。
下文保留历史回放结论，但历史诊断文件链接/复现输入已随清理删除。
基底模型已原样移至
`/home/LXJ/Python_Projects/Models/Qwen3.5-2B-ALFWorld-raw-RL-step150`。

2026-10-20，`新动作熵1019` 将 branch entropy 的位置选择由 `first`
改为 `first_non_listing`，系数仍为 0.0005。其他训练超参数及奖励不变；
思考动作单次/重复一致性奖励均为 0。历史配置未指定该选项时仍使用 `first`。

## 规则与边界

仅检查 thinking 内与最终 XML action 忽略大小写后相同、词边界完整的提及，
使用当前环境执行动作之前的合法动作集合。依次检查这些提及：

1. **多行清单**：存在 available/admissible actions、options 等可用性标题；
   连续的项目符号或编号列表中至少两项含合法动作；当前项仅含动作名、标点和连接词，
   或者仅附带下文定义的通用功能说明。
2. **行内枚举**：当前句子或包围该提及的括号内，至少出现两个不同合法动作；
   前导语说明可用动作，且各项只有标点、and/or 等枚举连接词或通用功能说明。
3. 规则确认的清单提及被跳过，在第一个保留的提及处构建原有合法动作 token trie。
   若后面仍是清单，继续跳过；全部提及都是清单则返回 `only_listing_mentions`，
   不回退到清单或 XML，也不改变该输出原有 GRPO 奖励、KL 和普通 token 熵损失。

不会仅凭 available 或项目符号判定。带任务相关解释、优劣分析、选择理由、单项可用性、
执行计划或不规则缩写的模糊文本默认保留。因此这是一种保守过滤，不是完整语义分类器，
更不声称定位到了模型内部真正做出决定的时刻。尤其 `look`/`inventory` 的普通词义、
之前已经隐含形成动作偏好等情况仍沿用原有精确字符串匹配的局限。

原始 rollout token IDs 不变；分叉 logits 的前移一位、合法动作集合、熵计算、轨迹权重不变。
改变提及位置，并按观测大小写渲染候选字符串后对齐。对齐不稳定仍使用既有跳过状态。

## 2026-10-20：功能说明和大小写兼容补充

通用功能说明采用整段白名单匹配，必须同时满足上述候选清单结构：

- `inventory - could see what I currently have`、`look - could reveal more information`。
- `go to coffeetable 2 - This would take me to the second coffee table`：
  导航说明的目标须与动作一致；允许冠词、复合词分写、编号省略及 first/second/third 序数。
- `open drawer 4 - I can explicitly try to open the drawer`、
  `take keychain 2 from shelf 13 - to take the keychain` 等直接功能复述。
- 对取出、移动、清洁、加热、冷却说明，如同时写出物品和来源/目标/工具，逐个核对参数。
  `examining ... is a valid action` 等合法性复述也可识别。
- 行内括号说明也支持；所在枚举的每项都须为纯动作或通用说明。

不是匹配到 `This would` 就跳过。说明尾部出现任务目标、因果、优先级或额外评价，
将不满足整段匹配。例如 `open the drawer first`、
`take me to the cabinet because the target is there` 和
`access the cabinet and put the soapbar inside it` 都保留。
没有可用动作标题、只有一个候选或执行计划标题时，也不会用该规则删除提及。

`go to cabinet 1`、`Go to cabinet 1`、`GO TO CABINET 1`、`Go To Cabinet 1`
现在属于同一完整动作的提及；`cabinet 10` 不会误匹配 `cabinet 1`。
XML 最终动作合法性检查仍使用环境的原始动作字符串。
此大小写兼容同时用于 `first` 和 `first_non_listing`；后者才执行清单过滤。

构建 trie 时，对所有候选统一应用观测提及的大小写风格，包括小写、句首大写、
全大写和标题式大写；混合大小写按观测词/字符位置映射，额外字符默认小写。
保留原始采样 token，仍验证采样路径及所有候选的前缀稳定性，不删除难以对齐的候选。
新增清单原因值为 `generic_gloss_bullet_list` / `generic_gloss_inline_list`。

### 补充规则的回放与测试

使用此前 step 41–45 的 120 条已复核样本，属于规则开发回放集，不是独立测试集：

| 项目 | 结果 |
| --- | ---: |
| 原先标记为清单的 22 条中，旧定位被新增规则跳过 | 8 |
| 原先标记为重复确认的 13 条中，因大小写兼容前移 | 2 |
| 原先标记为选择/比较的 76 条，定位发生变化 | 0 |
| 全部样本中，定位发生变化 | 10 / 120 |
| 新位置按文本重建 token 后成功对齐 | 119 / 120 |
| 候选前缀不稳定，跳过该项正则 | 1 / 120 |

样本 #9 从后文 `go to stoveburner 4` 前移到计划中的 `Go to stoveburner 4`；
#13 从后文 `go to cabinet 1` 前移到 `I should: 1. Go to cabinet 1 ...`。
#58 跳过功能说明后，新的选择句存在候选 token 前缀不稳定，返回 `unstable_prefix`。
这些是文本重建的对齐结果，不是运行中训练实际保存的逐条 token metadata。

不能把这 10 条全部算作“已找到真正决策点”：例如 #7 跳过功能清单后，
仍定位到 `I need to determine which one I should choose` 的候选比较；
#114 仍可能定位到普通名词用法的 `inventory`。
冠词插入、同义改写、隐含计划和其他重复确认尚不属于大小写修复范围。

另回放更早的 180 条首提及标注样本：26 条清单中识别量由旧版 14 条增至 18 条，
其余 154 条未新增清单误判。两个数据集均参与过规则设计，不能据此外推总体误判率。

验证命令：

```bash
PYTHONPATH=examples/alfworld_grpo_baseline/src:/home/LXJ/Python_Projects/verl \
  /home/LXJ/anaconda3/envs/alfworld-verl/bin/python -m pytest -q \
  examples/alfworld_grpo_baseline/tests/test_action_mentions.py \
  examples/alfworld_grpo_baseline/tests/test_branch_entropy.py
```

结果为 **96 passed**，覆盖任务理由保护、动作参数核对、大小写 token 对齐、
原始 token 不变、native queue 和损失梯度路径。进程退出时仍有环境已有的
ResourceTracker 析构提示，测试退出码为 0。
回放脚本、逐条前后对照 HTML、JSON 和代码哈希位于
`outputs/diagnostics/gloss_case_filter_20261020/`。

规则补充当时只修改源码与测试，未停止或重启已有训练进程；后来在上述全新启动时加载。
下面的续训记录描述的是此前首次启用 `first_non_listing` 的操作。

## 原版规则的回放证据（补充前）

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
