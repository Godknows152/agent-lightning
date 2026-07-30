# Agent Lightning - Project Documentation

**Last Updated**: 2026-07-29  
**Project Status**: Active development on image restoration multi-agent reinforcement learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Current Work: Image Restoration Multi-Agent System](#current-work-image-restoration-multi-agent-system)
4. [Directory Structure](#directory-structure)
5. [Key Components](#key-components)
6. [Development Guidelines](#development-guidelines)
7. [Training Workflows](#training-workflows)
8. [Important Notes](#important-notes)

---

## Project Overview

**Agent Lightning** is a framework for training AI agents with reinforcement learning, developed by Microsoft Research. It enables training of ANY AI agent with (almost) zero code changes, supporting multiple agent frameworks (LangChain, OpenAI Agent SDK, AutoGen, CrewAI, etc.) and multiple algorithms (RL, APO, SFT).

### Core Value Proposition

- **Framework Agnostic**: Works with any agent framework or even raw Python/OpenAI
- **Zero Code Change**: Minimal instrumentation required (`agl.emit_xxx()` or automatic tracing)
- **Selective Optimization**: Optimize specific agents in multi-agent systems
- **Multiple Algorithms**: RL (GRPO/PPO), Automatic Prompt Optimization, Supervised Fine-tuning

### Key Publications

- **arXiv Paper** (8/5/2025): "Agent Lightning: Train ANY AI Agents with Reinforcement Learning"
- **vLLM Blog** (10/22/2025): "No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL"

---

## Core Architecture

Agent Lightning follows a clean separation of concerns:

```
┌─────────────┐
│   Runner    │  ← Executes agents, emits spans
│   (Agent)   │
└──────┬──────┘
       │ spans (traces)
       ↓
┌─────────────┐
│ Lightning   │  ← Central hub: tasks, resources, traces
│   Store     │
└──────┬──────┘
       │ traces
       ↓
┌─────────────┐
│  Algorithm  │  ← Learns from spans, updates resources
│ (GRPO/APO)  │
└──────┬──────┘
       │ updated resources (prompts, weights)
       ↓
┌─────────────┐
│   Trainer   │  ← Orchestrates: datasets → runners → algorithm
└─────────────┘
```

### Key Concepts

1. **Spans**: Structured events (prompts, tool calls, rewards) emitted during agent execution
2. **LightningStore**: Central synchronization hub for tasks, resources, and traces
3. **Algorithms**: Pluggable learning algorithms (GRPO, PPO, APO, SFT)
4. **Trainer**: Orchestrates the training loop, manages resources and inference engines
5. **Runners**: Execute agent workloads and emit spans to the store

---

## Current Work: Image Restoration Multi-Agent System

### Overview

The current focus is on **training four image restoration experts** using GRPO (Group Relative Policy Optimization) reinforcement learning. Each expert specializes in one degradation type:

- **fog** expert
- **rain** expert  
- **snow** expert
- **lowlight** expert

### System Architecture

```
Input Image
    ↓
[Diagnosis Agent] ← Qwen3.5-9B VLM (port 8000)
    ↓
Route to Expert
    ↓
┌────────────────────────────────────────────────┐
│  Expert Subgraphs (LangGraph)                  │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐  │
│  │ Fog  │  │ Rain │  │ Snow │  │ Lowlight │  │
│  └──┬───┘  └──┬───┘  └──┬───┘  └────┬─────┘  │
└─────┼────────┼─────────┼────────────┼────────┘
      └────────┴─────────┴────────────┘
                    ↓
          [Restoration Worker]
          (16 actions: NAFNet, FocalNet, MB-TaylorFormer, etc.)
                    ↓
          [IQA Evaluator]
          (TOPIQ-NR, MUSIQ, NIQE, MANIQA, CLIP-IQA)
                    ↓
          Multi-turn loop → Best image
```

### Training Configuration

**Location**: `examples/image_restoration_multi_agent/old_verl_grpo/`

**Version Structure** (recently refactored 2026-07-29):
```
config/
  ├── fog/v1/, v2/, v3/     # Independent configs per version
  ├── rain/v1/, v2/, v3/
  ├── snow/v1/, v2/, v3/
  ├── lowlight/v1/, v2/, v3/
  └── tool_config/v1/, v2/, v3/

scripts/
  ├── fog/fog_v1.sh, fog_v2.sh, fog_v3.sh
  ├── rain/rain_v1.sh, rain_v2.sh, rain_v3.sh
  ├── snow/snow_v1.sh, snow_v2.sh, snow_v3.sh
  └── lowlight/lowlight_v1.sh, lowlight_v2.sh, lowlight_v3.sh
```

**Version Differences**:
- **v1**: Uses `current_iqa` tool config
- **v2**: Uses `marginal_efficiency` tool config
- **v3**: Uses `marginal_efficiency` tool config + `action_rarity_reward_coeff` (探索奖励)

### Training Pipeline

1. **Base Model**: Qwen3.5-9B multimodal
2. **Initialization**: From SFT LoRA adapters (trained on 0721 dataset)
3. **Algorithm**: GRPO (trajectory-level aggregation)
4. **Reward**: 4-metric IQA (MANIQA, NIQE-normalized, CLIP-IQA, TOPIQ-NR)
5. **Action Space**: 16 restoration actions (shared across experts)
6. **Data**: 1,000 train + 200 validation images per expert (no ground truth)
7. **Tracking**: SwanLab integration (project: `4expert_grpo`)

### Action Rarity Reward (v3 Feature)

**Purpose**: Encourage exploration of diverse tool usage, prevent premature convergence

**Mechanism**:
- Calculates empirical rarity of actions within each rollout batch
- Only rewards trajectories with `advantages > 0` (quality gate)
- Coefficient: `action_rarity_reward_coeff` (recently experimented with 0.02 → 0.5)

**Recent Findings** (2026-07-29):
- **coeff=0.02**: Too small, only ~1% influence
- **coeff=0.5**: 13.9x stronger, but tool diversity unexpectedly decreased
- **Recommendation**: Test intermediate value (0.2) or redesign mechanism

### Key Scripts

```bash
# Run fog v3 training (with action rarity reward)
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh

# Run all four experts sequentially
bash examples/image_restoration_multi_agent/old_verl_grpo/run_four_experts_serial_old_verl_grpo_2gpu.sh

# Smoke tests
python examples/image_restoration_multi_agent/smoke_test.py
python examples/image_restoration_multi_agent/stage_e_smoke_test.py
python examples/image_restoration_multi_agent/stage_f_smoke_test.py
```

---

## Directory Structure

```
Agent_Lightning/
├── agentlightning/              # Core library
│   ├── adapter/                 # Framework adapters
│   ├── algorithm/               # Learning algorithms (APO, VERL/GRPO)
│   ├── cli/                     # Command-line interface
│   ├── emitter/                 # Span emission
│   ├── execution/               # Execution runtime
│   ├── instrumentation/         # Auto-instrumentation
│   ├── runner/                  # Rollout runners
│   ├── store/                   # LightningStore implementation
│   ├── tracer/                  # Tracing backends (OTEL, AgentOps, Weave)
│   ├── trainer/                 # Training orchestration
│   └── types/                   # Type definitions
│
├── examples/
│   ├── image_restoration_multi_agent/    # **CURRENT WORK**
│   │   ├── old_verl_grpo/               # Training configs & scripts
│   │   │   ├── config/                   # Version-specific configs
│   │   │   ├── scripts/                  # Launch scripts
│   │   │   ├── data/                     # Training datasets
│   │   │   ├── outputs/                  # Training outputs
│   │   │   └── log/                      # Training logs
│   │   ├── verl_backend/                 # VERL integration
│   │   ├── controller.py                 # Multi-agent controller
│   │   ├── diagnosis_agent.py            # Degradation diagnosis
│   │   ├── expert_agent.py               # Expert decision making
│   │   ├── restoration_worker.py         # Image restoration execution
│   │   └── evaluator.py                  # IQA evaluation
│   │
│   ├── minimal/                 # Minimal examples
│   ├── sql_agent/              # SQL agent RL training
│   └── README.md
│
├── docs/                        # Documentation (MkDocs)
├── dashboard/                   # Web UI
├── tests/                       # Test suite
├── External_Tools/              # External restoration tools
├── LlamaFactory/                # Fine-tuning framework integration
├── Information_Search/          # Search tools
└── contrib/                     # Community contributions
```

---

## Key Components

### 1. Core Library (`agentlightning/`)

#### Store (`agentlightning/store/`)
- Central synchronization hub
- Manages tasks, resources (prompts, weights), and traces
- Supports MongoDB or in-memory backends

#### Algorithm (`agentlightning/algorithm/`)
- **VERL/GRPO**: Trajectory-level GRPO for agent RL
- **APO**: Automatic Prompt Optimization
- **SFT**: Supervised Fine-Tuning

#### Tracer (`agentlightning/tracer/`)
- Captures agent execution as structured spans
- Backends: OpenTelemetry, AgentOps, Weave, Dummy
- Automatic instrumentation for popular frameworks

#### Runner (`agentlightning/runner/`)
- Executes agent workloads
- Emits spans to store
- Supports batch processing

#### Trainer (`agentlightning/trainer/`)
- Orchestrates training loop
- Manages datasets → runners → algorithm → inference updates
- Supports Ray for distributed training

### 2. Image Restoration Components

#### Controller (`controller.py`)
- Deterministic workflow orchestration
- Routes to expert subgraphs based on diagnosis
- Manages multi-turn restoration loop
- Tracks best image and trajectory JSON

#### Diagnosis Agent (`diagnosis_agent.py`)
- Qwen3.5-9B VLM for degradation classification
- Categories: fog, rain, snow, lowlight
- Hermes tool-call format
- Strict parsing with fallback handling

#### Expert Agent (`expert_agent.py`)
- Four independent expert identities (fog/rain/snow/lowlight)
- Qwen3.5 native XML tool-call format
- 16 shared restoration actions
- Replay mode (scripted) vs VLM-strict mode

#### Restoration Worker (`restoration_worker.py`)
- Isolated `verl` environment execution
- 16 actions: NAFNet, FocalNet, MB-TaylorFormer, BasicSR tools
- GPU allocation: models on cuda:1, IQA on cuda:0
- Stage H persistent HTTP service

#### Evaluator (`evaluator.py`)
- Multi-metric IQA: TOPIQ-NR, MUSIQ, NIQE, MANIQA, CLIP-IQA
- Weighted aggregation for training reward
- Direction-normalized NIQE
- No ground truth required

### 3. VERL Backend (`verl_backend/`)

Custom fork of VERL (Versatile Reinforcement Learning) framework:
- Qwen3.5 multimodal compatibility patches
- LoRA-only policy updates
- Trajectory-level aggregation
- Multi-turn tool execution support
- SwanLab integration

---

## Development Guidelines

### Code Style

- **Python**: 4-space indentation, 120-char lines, Black formatter
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Type Hints**: Exhaustive (enforced by pyright)
- **Docstrings**: Google style, focus on API contracts not implementation

### Testing

```bash
# Run all tests
uv run --no-sync pytest -v

# Run specific markers
uv run --no-sync pytest -m "not mongo" -v

# Type checking
uv run --no-sync pyright

# Linting
uv run --no-sync pre-commit run --all-files
```

### Documentation

```bash
# Build docs locally
uv run --no-sync mkdocs build --strict

# Serve docs
uv run --no-sync mkdocs serve
```

### Environment Setup

```bash
# Install with dev dependencies
uv sync --group dev

# Install with optional groups (VERL, APO, GPU)
uv sync --group verl --group gpu
```

---

## Training Workflows

### Running Single Expert Training

```bash
# Fog expert, v3 configuration (with action rarity reward)
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v3.sh

# Rain expert, v2 configuration (marginal efficiency reward only)
bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/rain/rain_v2.sh
```

### Modifying Training Configuration

**Edit config file directly**:
```bash
vim examples/image_restoration_multi_agent/old_verl_grpo/config/fog/v3/fog_config_2gpu.yaml
```

**Override via command line**:
```bash
bash scripts/fog/fog_v3.sh algorithm.action_rarity_reward_coeff=0.2
```

**Batch modify all v3 configs**:
```bash
cd examples/image_restoration_multi_agent/old_verl_grpo
for expert in fog rain snow lowlight; do
  sed -i 's/action_rarity_reward_coeff: 0.02/action_rarity_reward_coeff: 0.5/' \
    config/${expert}/v3/${expert}_config_2gpu.yaml
done
```

### Monitoring Training

**SwanLab** (web UI):
```bash
# View experiments
https://swanlab.cn/@Godknows/4expert_grpo

# CLI
swanlab api run list Godknows/4expert_grpo
swanlab api run summary Godknows/4expert_grpo/<run_id>
```

**Logs**:
```bash
# Check training log
tail -f examples/image_restoration_multi_agent/old_verl_grpo/log/fog/v3/fog_v3_*.log
```

### Key Metrics to Monitor

- `critic/rewards/mean`: Total trajectory reward (primary metric)
- `restoration_reward/pure_image_reward_mean`: Image quality improvement
- `actor/tool_choice_entropy`: Tool selection diversity
- `actor/tool_choice_unique_action_count`: Number of unique tools used
- `actor/action_rarity_reward_mean`: Exploration bonus (v3 only)
- `actor/entropy`: Policy entropy (exploration indicator)

---

## Important Notes

### Recent Changes (2026-07-29)

1. **Configuration Refactoring**:
   - Separated configs into expert-specific version directories
   - Each expert has independent v1/v2/v3 configs
   - Tool configs versioned (v1/v2/v3)
   - Launch scripts now directly point to config files
   - See: `CONFIG_STRUCTURE.md`, `REFACTORING_SUMMARY.md`

2. **Action Rarity Reward Analysis**:
   - Analyzed fog_v3 training with coeff=0.02 (19 steps)
   - Tested coeff=0.5 (106 steps) - mixed results
   - Tool diversity unexpectedly decreased with higher coefficient
   - Negative correlation with total reward persists (-0.649)
   - See: `fog_v3_comparison_report.md`, `QUICK_GUIDE_ACTION_RARITY.md`

### Known Issues

1. **Action Rarity Reward Design**:
   - Negative correlation with total reward suggests design flaw
   - "Rare" actions are not necessarily "good" actions
   - Gate mechanism (advantages > 0) may cause tool convergence
   - Consider redesigning as "diversity reward" instead

2. **Training Stability**:
   - Performance can decline after peak (e.g., fog_v3_续 peaked at step 81, declined to 106)
   - May need dynamic coefficient scheduling or curriculum learning

3. **Experiment Comparability**:
   - Different runs may start from different checkpoints
   - Reward scale can vary between experiments
   - Always verify starting conditions before comparing

### Best Practices

1. **Starting New Training**:
   - Verify config points to correct SFT LoRA adapter
   - Check tool_config_path is correct for version
   - Ensure CUDA_VISIBLE_DEVICES set appropriately
   - Clear old checkpoints if starting fresh

2. **Debugging**:
   - Use smoke tests to validate pipeline before full training
   - Check logs for model loading, data loading, reward computation
   - Monitor SwanLab for NaN/Inf values early

3. **Hyperparameter Tuning**:
   - Start with small coefficient changes (0.02 → 0.1 → 0.2)
   - Always run ablation studies (with vs without feature)
   - Compare at same training stage (early vs late)
   - Document all configuration changes in git commits

### Environment Requirements

- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU training)
- **RAM**: 64GB+ recommended for 2-GPU training
- **Storage**: ~50GB for models + datasets

**Conda Environments**:
- `agent-lightning`: Main environment (Agent Lightning, vLLM, SwanLab)
- `verl`: Isolated environment (restoration models, IQA models, BasicSR)

### GPU Allocation

**Typical 2-GPU Setup**:
- GPU 0: vLLM model server (diagnosis + expert)
- GPU 1: Restoration models + IQA models (via isolated verl environment)

**Training**:
- Actor/Critic models loaded on GPU 0, GPU 1
- Ray manages distribution automatically

---

## References

### Documentation

- **Main Docs**: https://microsoft.github.io/agent-lightning/
- **Installation**: https://microsoft.github.io/agent-lightning/stable/tutorials/installation/
- **Examples**: `./examples/`
- **Image Restoration README**: `examples/image_restoration_multi_agent/README.md`

### Recent Analysis Reports

- `fog_v3_analysis.png` / `fog_v3_report.txt`: Initial analysis (coeff=0.02)
- `fog_v3_continued_analysis.png` / `fog_v3_comparison_report.md`: Comparison (0.02 vs 0.5)
- `CONFIG_STRUCTURE.md`: Refactored configuration structure
- `QUICK_GUIDE_ACTION_RARITY.md`: Quick guide for modifying action_rarity_reward_coeff
- `REFACTORING_SUMMARY.md`: Summary of configuration refactoring

### Key Papers

- Agent Lightning: https://arxiv.org/abs/2508.03680
- GRPO: Group Relative Policy Optimization
- Qwen3.5: https://github.com/QwenLM/Qwen

---

## Contact & Support

- **Discord**: https://discord.gg/RYk7CdvDR7
- **GitHub Issues**: https://github.com/microsoft/agent-lightning/issues
- **Documentation**: https://microsoft.github.io/agent-lightning/

---

**Note**: This is a living document. Update it as the project evolves, especially when:
- Major architectural changes occur
- New features are added
- Training configurations change significantly
- Important findings emerge from experiments
