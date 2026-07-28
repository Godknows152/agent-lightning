# 配置文件和启动脚本结构说明

## 目录结构

重构后的配置文件和启动脚本采用版本化的独立结构：

```
examples/image_restoration_multi_agent/old_verl_grpo/
├── config/                                    # 配置文件目录
│   ├── fog/                                   # fog 专家配置
│   │   ├── v1/
│   │   │   └── fog_config_2gpu.yaml          # v1 配置（使用 current_iqa tool_config）
│   │   ├── v2/
│   │   │   └── fog_config_2gpu.yaml          # v2 配置（使用 marginal_efficiency tool_config）
│   │   └── v3/
│   │       └── fog_config_2gpu.yaml          # v3 配置（marginal_efficiency + action_rarity）
│   ├── rain/                                  # rain 专家配置
│   │   ├── v1/, v2/, v3/
│   ├── snow/                                  # snow 专家配置
│   │   ├── v1/, v2/, v3/
│   ├── lowlight/                              # lowlight 专家配置
│   │   ├── v1/, v2/, v3/
│   └── tool_config/                           # 工具配置目录
│       ├── v1/
│       │   └── restoration_tool_config_2gpu.yaml      # current_iqa
│       ├── v2/
│       │   └── restoration_tool_config_2gpu.yaml      # marginal_efficiency
│       └── v3/
│           └── restoration_tool_config_2gpu.yaml      # marginal_efficiency
│
└── scripts/                                   # 启动脚本目录
    ├── fog/
    │   ├── fog_v1.sh
    │   ├── fog_v2.sh
    │   └── fog_v3.sh
    ├── rain/
    │   ├── rain_v1.sh, rain_v2.sh, rain_v3.sh
    ├── snow/
    │   ├── snow_v1.sh, snow_v2.sh, snow_v3.sh
    └── lowlight/
        ├── lowlight_v1.sh, lowlight_v2.sh, lowlight_v3.sh
```

## 版本差异

### v1
- **Tool Config**: `current_iqa` - 基于当前 IQA 得分的工具奖励
- **特点**: 基础版本，无 action_rarity 奖励

### v2
- **Tool Config**: `marginal_efficiency` - 基于边际效率的工具奖励
- **特点**: 改进的工具选择奖励机制

### v3
- **Tool Config**: `marginal_efficiency` - 基于边际效率的工具奖励
- **Action Rarity Reward**: `0.02` - 稀有动作探索奖励
- **特点**: 在 v2 基础上增加稀有度奖励，鼓励工具探索

## 使用方法

### 1. 运行特定版本的训练

```bash
# 运行 fog v3 版本
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh

# 运行 rain v2 版本
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/rain/rain_v2.sh

# 运行 lowlight v1 版本
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/lowlight/lowlight_v1.sh
```

### 2. 修改配置

每个版本的配置文件完全独立，修改配置只需编辑对应的 YAML 文件：

```bash
# 修改 fog v3 的配置
vim examples/image_restoration_multi_agent/old_verl_grpo/config/fog/v3/fog_config_2gpu.yaml

# 修改 action_rarity_reward_coeff
# 在配置文件末尾的 algorithm 部分修改：
algorithm:
  action_rarity_reward_coeff: 0.5  # 从 0.02 改为 0.5
```

### 3. 修改工具配置

```bash
# 修改 v3 版本的工具配置（影响所有使用 v3 的专家）
vim examples/image_restoration_multi_agent/old_verl_grpo/config/tool_config/v3/restoration_tool_config_2gpu.yaml
```

### 4. 传递 Hydra 参数覆盖

启动脚本支持 Hydra 参数覆盖：

```bash
# 覆盖单个参数
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=0.5

# 覆盖多个参数
bash scripts/fog/fog_v3.sh \
  algorithm.action_rarity_reward_coeff=1.0 \
  trainer.n_gpus_per_node=4
```

### 5. 查看帮助

```bash
bash scripts/fog/fog_v3.sh --help
```

## 配置文件说明

### 专家配置文件结构

每个专家的配置文件包含以下部分：

```yaml
# 1. Hydra 配置
hydra:
  searchpath:
    - file:///path/to/verl/trainer/config

defaults:
  - restoration_common_config_2gpu  # 继承通用配置
  - _self_

# 2. 数据配置
data:
  train_files:
    - path/to/train.parquet
  val_files:
    - path/to/val.parquet

# 3. 模型和推理配置
actor_rollout_ref:
  model:
    path: /path/to/base/model
    lora_adapter_path: /path/to/sft/lora
  rollout:
    multi_turn:
      tool_config_path: config/tool_config/v3/restoration_tool_config_2gpu.yaml

# 4. 训练器配置
trainer:
  experiment_name: "fog_v3"
  default_local_dir: outputs/fog/v3
  ray_kwargs:
    ray_init:
      runtime_env:
        env_vars:
          SWANLAB_LOG_DIR: "/path/to/swanlab"
          VERL_LOG_DIR: "/path/to/log"

# 5. 算法配置（仅 v3）
algorithm:
  action_rarity_reward_coeff: 0.02
```

## 命名规范

### 文件命名
- 配置文件: `{expert}_config_2gpu.yaml`
- 工具配置: `restoration_tool_config_2gpu.yaml`
- 启动脚本: `{expert}_v{1|2|3}.sh`

### 目录命名
- 专家名称: `fog`, `rain`, `snow`, `lowlight`（统一小写，使用下划线）
- 版本号: `v1`, `v2`, `v3`

### 变量命名
- `EXPERT`: 专家名称
- `RUNTIME_EXPERT`: 训练运行时专家标识；lowlight 配置使用现有数据集与 LoRA 所需的 `low_light`
- `VERSION`: 版本号
- `CONFIG_PATH`: 配置文件路径
- `OLD_VERL_*`: 环境变量前缀

## 常见任务

### 任务 1: 调整 action_rarity_reward_coeff

编辑 `config/{expert}/v3/{expert}_config_2gpu.yaml`：

```yaml
algorithm:
  action_rarity_reward_coeff: 0.5  # 从 0.02 调整为 0.5
```

### 任务 2: 修改工具选择策略

编辑 `config/tool_config/v3/restoration_tool_config_2gpu.yaml`。

### 任务 3: 更换基础模型

编辑 `config/{expert}/{version}/{expert}_config_2gpu.yaml`：

```yaml
actor_rollout_ref:
  model:
    path: /path/to/new/model
```

### 任务 4: 调整训练超参数

大多数训练超参数在 `config/restoration_common_config_2gpu.yaml` 中定义。
要为特定版本覆盖，在专家配置文件中添加：

```yaml
algorithm:
  action_rarity_reward_coeff: 0.5
  gamma: 0.99
  lam: 0.95
```

## 迁移说明

### 旧脚本 vs 新脚本

**旧方式:**
```bash
# 所有 v3 脚本都依赖 run_action_rarity_v3_old_verl_grpo_2gpu.sh
bash examples/.../scripts/fog/fog_v3.sh
# 内部调用: run_action_rarity_v3_old_verl_grpo_2gpu.sh fog
```

**新方式:**
```bash
# 每个脚本独立，直接指向自己的配置
bash examples/.../scripts/fog/fog_v3.sh
# 直接加载: config/fog/v3/fog_config_2gpu.yaml
```

### 配置文件位置变化

| 旧位置 | 新位置 |
|--------|--------|
| `config/fog_config_2gpu.yaml` | `config/fog/v1/fog_config_2gpu.yaml` |
| (参数在脚本中硬编码) | `config/fog/v2/fog_config_2gpu.yaml` |
| (参数在脚本中硬编码) | `config/fog/v3/fog_config_2gpu.yaml` |

## 注意事项

1. **配置文件完全独立**: 修改某个版本的配置不会影响其他版本
2. **Tool Config 共享**: 同一版本的所有专家共享同一个 tool_config 文件
3. **旧脚本保留**: 原有的 `run_action_rarity_v3_old_verl_grpo_2gpu.sh` 等脚本仍然保留，可以继续使用
4. **日志和输出分离**: 每个版本的日志和输出都存储在独立的目录中

## 故障排查

### 问题 1: 配置文件找不到

```bash
# 检查配置文件是否存在
ls -l config/fog/v3/fog_config_2gpu.yaml

# 检查脚本中的路径
grep "CONFIG_PATH" scripts/fog/fog_v3.sh
```

### 问题 2: Tool Config 路径错误

```bash
# 检查配置文件中的 tool_config_path
grep "tool_config_path" config/fog/v3/fog_config_2gpu.yaml

# 检查 tool_config 文件是否存在
ls -l config/tool_config/v3/restoration_tool_config_2gpu.yaml
```

### 问题 3: 权限问题

```bash
# 确保脚本有执行权限
chmod +x scripts/fog/fog_v3.sh
```

## 进一步扩展

如果需要添加新版本（如 v4）：

1. 创建新的配置目录和文件：
   ```bash
   mkdir -p config/fog/v4 config/tool_config/v4
   cp config/fog/v3/fog_config_2gpu.yaml config/fog/v4/
   cp config/tool_config/v3/restoration_tool_config_2gpu.yaml config/tool_config/v4/
   ```

2. 修改配置文件中的版本引用：
   ```bash
   sed -i 's/v3/v4/g' config/fog/v4/fog_config_2gpu.yaml
   ```

3. 创建新的启动脚本：
   ```bash
   cp scripts/fog/fog_v3.sh scripts/fog/fog_v4.sh
   sed -i 's/v3/v4/g' scripts/fog/fog_v4.sh
   ```

4. 根据需要修改 v4 的配置和参数
