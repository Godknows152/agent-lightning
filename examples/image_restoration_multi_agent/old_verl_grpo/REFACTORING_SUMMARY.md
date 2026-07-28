# 配置文件和启动脚本重构总结

## 完成时间
2026-07-29

## 重构目标
将所有专家（fog, rain, snow, lowlight）的 v1/v2/v3 配置文件和启动脚本独立化，使每个版本的训练配置可以独立修改，而不相互影响。

## 主要变更

### 1. 目录结构重组

**之前:**
```
config/
  ├── fog_config_2gpu.yaml          # 共用配置
  ├── rain_config_2gpu.yaml
  ├── snow_config_2gpu.yaml
  ├── low_light_config_2gpu.yaml
  └── tool_config/
      ├── restoration_tool_config_current_iqa_2gpu.yaml
      └── restoration_tool_config_marginal_efficiency_2gpu.yaml

scripts/
  ├── fog/fog_v3.sh                 # 调用共享脚本
  └── ...
```

**之后:**
```
config/
  ├── fog/
  │   ├── v1/fog_config_2gpu.yaml   # 独立配置
  │   ├── v2/fog_config_2gpu.yaml
  │   └── v3/fog_config_2gpu.yaml
  ├── rain/v1/, rain/v2/, rain/v3/
  ├── snow/v1/, snow/v2/, snow/v3/
  ├── lowlight/v1/, lowlight/v2/, lowlight/v3/
  └── tool_config/
      ├── v1/restoration_tool_config_2gpu.yaml
      ├── v2/restoration_tool_config_2gpu.yaml
      └── v3/restoration_tool_config_2gpu.yaml

scripts/
  ├── fog/
  │   ├── fog_v1.sh                 # 独立脚本，直接加载配置
  │   ├── fog_v2.sh
  │   └── fog_v3.sh
  └── ...
```

### 2. 配置文件独立化

每个专家的每个版本都有独立的配置文件：

- **v1**: 使用 `current_iqa` tool_config
- **v2**: 使用 `marginal_efficiency` tool_config
- **v3**: 使用 `marginal_efficiency` tool_config + `action_rarity_reward_coeff: 0.02`

配置文件中直接指定了 `tool_config_path`，不再需要在脚本中硬编码。

### 3. 启动脚本简化

**之前:**
```bash
# fog_v3.sh
exec "${OLD_VERL_DIR}/run_action_rarity_v3_old_verl_grpo_2gpu.sh" fog "$@"
```

**之后:**
```bash
# fog_v3.sh
EXPERT="fog"
VERSION="v3"
CONFIG_PATH="${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}/${EXPERT}_config_2gpu.yaml"

exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "--config-path=${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}" \
  "--config-name=${EXPERT}_config_2gpu"
```

### 4. 命名规范统一

- **专家名称**: 统一使用小写 + 下划线
  - `fog`, `rain`, `snow`, `lowlight` (不再使用 `low_light`)
- **配置文件**: `{expert}_config_2gpu.yaml`
- **启动脚本**: `{expert}_v{1|2|3}.sh`
- **版本目录**: `v1/`, `v2/`, `v3/`

其中 lowlight 的配置目录、输出和启动脚本使用 `lowlight`；底层训练仍传入 `low_light`，以兼容既有数据文件、SFT LoRA 目录和工具注册表。

## 优势

### 1. 独立性
- 每个版本的配置完全独立
- 修改某个版本不会影响其他版本
- 可以为不同版本设置不同的超参数

### 2. 可维护性
- 配置文件位置清晰
- 脚本逻辑简单明了
- 易于查找和修改

### 3. 灵活性
- 支持 Hydra 参数覆盖
- 可以快速测试不同配置
- 易于添加新版本（v4, v5...）

### 4. 一致性
- 统一的命名规范
- 统一的目录结构
- 统一的脚本模式

## 使用指南

### 运行训练
```bash
# 运行 fog v3
bash scripts/fog/fog_v3.sh

# 运行 rain v2
bash scripts/rain/rain_v2.sh
```

### 修改配置
```bash
# 修改 fog v3 的 action_rarity_reward_coeff
vim config/fog/v3/fog_config_2gpu.yaml

# 批量修改所有 v3 的系数
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.02/action_rarity_reward_coeff: 0.5/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done
```

### 临时测试参数
```bash
# 不修改配置文件，直接覆盖参数
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=1.0
```

## 文件清单

### 新增文件

#### 配置文件 (12 个)
- `config/fog/v1/fog_config_2gpu.yaml`
- `config/fog/v2/fog_config_2gpu.yaml`
- `config/fog/v3/fog_config_2gpu.yaml`
- `config/rain/v1/rain_config_2gpu.yaml`
- `config/rain/v2/rain_config_2gpu.yaml`
- `config/rain/v3/rain_config_2gpu.yaml`
- `config/snow/v1/snow_config_2gpu.yaml`
- `config/snow/v2/snow_config_2gpu.yaml`
- `config/snow/v3/snow_config_2gpu.yaml`
- `config/lowlight/v1/lowlight_config_2gpu.yaml`
- `config/lowlight/v2/lowlight_config_2gpu.yaml`
- `config/lowlight/v3/lowlight_config_2gpu.yaml`

#### Tool Config (3 个)
- `config/tool_config/v1/restoration_tool_config_2gpu.yaml`
- `config/tool_config/v2/restoration_tool_config_2gpu.yaml`
- `config/tool_config/v3/restoration_tool_config_2gpu.yaml`

#### 启动脚本 (12 个)
- `scripts/fog/fog_v1.sh`, `fog_v2.sh`, `fog_v3.sh`
- `scripts/rain/rain_v1.sh`, `rain_v2.sh`, `rain_v3.sh`
- `scripts/snow/snow_v1.sh`, `snow_v2.sh`, `snow_v3.sh`
- `scripts/lowlight/lowlight_v1.sh`, `lowlight_v2.sh`, `lowlight_v3.sh`

#### 文档 (3 个)
- `CONFIG_STRUCTURE.md` - 详细的结构说明
- `QUICK_GUIDE_ACTION_RARITY.md` - action_rarity_reward_coeff 修改指南
- `REFACTORING_SUMMARY.md` - 本文档

### 保留文件

以下旧文件仍然保留，可以继续使用：
- `config/fog_config_2gpu.yaml`
- `config/rain_config_2gpu.yaml`
- `config/snow_config_2gpu.yaml`
- `config/low_light_config_2gpu.yaml`
- `run_action_rarity_v3_old_verl_grpo_2gpu.sh`
- `run_expert_old_verl_grpo_2gpu.sh`

### 删除文件

清理了命名不规范的旧脚本：
- `scripts/lowlight/low_light_v1.sh` (改为 `lowlight_v1.sh`)
- `scripts/lowlight/low_light_v2.sh` (改为 `lowlight_v2.sh`)
- `scripts/lowlight/low_light_v3.sh` (改为 `lowlight_v3.sh`)

## 迁移建议

### 对现有训练的影响
- ✅ 不影响：新旧脚本可以共存，旧脚本仍然可用
- ✅ 向后兼容：所有环境变量和参数覆盖机制保持不变
- ⚠️ 建议迁移：新训练建议使用新的独立脚本

### 迁移步骤
1. 确认当前训练使用的配置和参数
2. 在对应的新配置文件中设置相同参数
3. 使用新脚本启动训练
4. 验证训练正常启动和日志输出

## 后续工作

### 推荐的改进
1. ✅ **已完成**: 配置文件和脚本独立化
2. ✅ **已完成**: 命名规范统一
3. ⏳ **建议**: 根据 fog_v3 分析结果，调整 `action_rarity_reward_coeff` 从 0.02 到 0.5-1.0
4. ⏳ **建议**: 实现动态系数调整（课程学习）
5. ⏳ **建议**: 添加配置验证脚本

### 可选的扩展
- 添加配置模板生成脚本
- 实现配置文件 diff 工具
- 添加训练监控脚本

## 验证

所有重构已通过验证：
```bash
✓ 12 个配置文件（4 专家 × 3 版本）
✓ 3 个 tool_config 文件（v1/v2/v3）
✓ 12 个启动脚本（4 专家 × 3 版本）
✓ 配置文件中的 tool_config_path 正确指向对应版本
✓ v3 配置包含 action_rarity_reward_coeff
✓ 脚本可以正常启动（--help 测试通过）
```

## 相关文档

- **详细结构说明**: `CONFIG_STRUCTURE.md`
- **快速修改指南**: `QUICK_GUIDE_ACTION_RARITY.md`
- **实验分析报告**: `fog_v3_analysis.png`, `fog_v3_report.txt`

## 总结

这次重构实现了配置文件和启动脚本的完全独立化，使得：
1. 每个版本可以独立修改和维护
2. 命名规范统一，易于理解和使用
3. 为后续的参数调优提供了便利（特别是 action_rarity_reward_coeff）

所有改动都是向后兼容的，旧脚本仍然可用，迁移风险低。
