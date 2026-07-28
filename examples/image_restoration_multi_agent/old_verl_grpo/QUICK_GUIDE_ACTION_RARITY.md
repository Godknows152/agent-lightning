# 快速修改 action_rarity_reward_coeff 指南

## 背景

根据 fog_v3 实验分析，当前 `action_rarity_reward_coeff = 0.02` 的影响力太小（仅占单步变化的 ~1%），建议提升到 0.5-1.0 以真正发挥探索作用。

## 修改方法

### 方法 1: 直接修改配置文件（推荐）

为每个专家的 v3 配置文件修改系数：

```bash
# 修改 fog v3 的系数
vim examples/image_restoration_multi_agent/old_verl_grpo/config/fog/v3/fog_config_2gpu.yaml

# 在文件末尾找到 algorithm 部分，修改为：
algorithm:
  action_rarity_reward_coeff: 0.5  # 从 0.02 改为 0.5（保守方案）
  # 或
  action_rarity_reward_coeff: 1.0  # 从 0.02 改为 1.0（推荐方案）
```

对所有专家批量修改：

```bash
cd examples/image_restoration_multi_agent/old_verl_grpo

# 批量修改为 0.5
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.02/action_rarity_reward_coeff: 0.5/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done

# 或批量修改为 1.0
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.02/action_rarity_reward_coeff: 1.0/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done
```

### 方法 2: 通过命令行参数覆盖（临时测试）

不修改配置文件，通过启动参数覆盖：

```bash
# 测试 coeff = 0.5
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=0.5

# 测试 coeff = 1.0
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=1.0

# 测试 coeff = 2.0（激进方案）
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=2.0
```

## 推荐的实验方案

### 阶段 1: 验证方向（coeff = 0.5）

```bash
# 修改所有 v3 配置为 0.5
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.02/action_rarity_reward_coeff: 0.5/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done

# 运行训练（以 fog 为例）
bash scripts/fog/fog_v3.sh
```

**观察指标（5-10 steps）：**
- `actor/tool_choice_entropy`: 期望从 2.45 → 2.50+
- `critic/rewards/mean`: 可接受下降 <15%
- `actor/action_rarity_reward_mean`: 期望从 0.009 → 0.22

**判断标准：**
- ✅ Entropy 提升且总奖励下降 <10%: 升级到阶段 2
- ⚠️ Entropy 提升但总奖励下降 10-20%: 继续观察
- ❌ Entropy 无变化: 直接跳到阶段 2

### 阶段 2: 强化探索（coeff = 1.0）

```bash
# 修改所有 v3 配置为 1.0
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.5/action_rarity_reward_coeff: 1.0/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done

# 运行训练
bash scripts/fog/fog_v3.sh
```

**观察指标（至收敛）：**
- `actor/tool_choice_entropy`: 期望 ≥ 2.50
- `actor/tool_choice_unique_action_count`: 保持 15-16
- 最终性能 vs baseline（v2）

### 阶段 3（可选）: 动态系数

如果固定系数效果好，可以实现课程学习：

```yaml
# 在配置文件中暂时不支持动态系数
# 需要修改代码实现，或使用不同阶段手动切换配置
```

## 验证命令

查看当前配置的系数：

```bash
# 查看单个专家
grep "action_rarity_reward_coeff" config/fog/v3/fog_config_2gpu.yaml

# 查看所有专家
for expert in fog rain snow lowlight; do
  echo "=== ${expert} v3 ==="
  grep "action_rarity_reward_coeff" config/${expert}/v3/${expert}_config_2gpu.yaml
done
```

## 监控关键指标

训练时重点关注：

```python
# 期望看到的模式（coeff = 0.5-1.0 时）
当 action_rarity_reward_coeff 从 0.02 → 0.5-1.0：
  ✓ actor/action_rarity_reward_mean: 0.009 → 0.22-0.45
  ✓ actor/tool_choice_entropy: 2.45 → 2.50-2.60
  ✓ actor/tool_choice_unique_action_count: 保持 15-16
  ? critic/rewards/mean: 短期可能下降 10-20%（探索代价）
  ✓ actor/entropy: 持续上升（保持探索）
```

## 回滚方案

如果发现系数过大导致训练不稳定：

```bash
# 回滚到原始值 0.02
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: [0-9.]\+/action_rarity_reward_coeff: 0.02/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done

# 或回滚到中间值 0.5
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: [0-9.]\+/action_rarity_reward_coeff: 0.5/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done
```

## 理论依据

根据 fog_v3 实验分析：

- **当前问题**: coeff = 0.02 时，稀有度奖励 ≈ 0.009，仅占单步变化（0.5-2.0）的 ~1%
- **目标范围**: 稀有度奖励应达到 0.3-1.5，即占单步变化的 20-100%
- **计算公式**: `稀有度奖励 = rarity_score × coeff × gate_ratio ≈ 0.88 × coeff × 0.51`
- **推荐系数**:
  - 保守: coeff = 0.5 → 稀有度奖励 ≈ 0.22 (占 20%)
  - 推荐: coeff = 1.0 → 稀有度奖励 ≈ 0.45 (占 45%)
  - 激进: coeff = 2.0 → 稀有度奖励 ≈ 0.89 (占 90%)

## 常见问题

**Q: 为什么负相关不是问题？**
A: 稀有度奖励的目标是维持探索，而非优化总奖励。负相关说明模型在探索"好但不常见"的工具，这是预期行为。

**Q: 多大的系数算合理？**
A: 稀有度奖励应占单步奖励变化的 20-100%。根据 fog_v3 数据，0.5-2.0 是合理范围。

**Q: 如果总奖励下降太多怎么办？**
A: 短期下降 10-20% 是正常的探索代价。如果下降 >30%，说明系数过大，应回退到更小的值。

**Q: v1 和 v2 也需要修改吗？**
A: 不需要。v1 和 v2 没有 action_rarity_reward，只有 v3 有这个机制。

## 相关文件

- 配置文件位置: `config/{expert}/v3/{expert}_config_2gpu.yaml`
- 启动脚本: `scripts/{expert}/{expert}_v3.sh`
- 分析报告: `fog_v3_analysis.png`, `fog_v3_report.txt`
- 详细文档: `CONFIG_STRUCTURE.md`
