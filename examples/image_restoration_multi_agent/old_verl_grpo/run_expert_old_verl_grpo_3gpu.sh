#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_expert_old_verl_grpo_3gpu.sh <fog|low_light|rain|snow> [--smoke|--preflight] [hydra overrides...]

Physical GPU topology:
  GPU0/1: Actor, frozen reference policy, TP2 SGLang rollout
  GPU2: persistent image-restoration models and IQA scoring (replica 0)
  GPU3: persistent image-restoration models and IQA scoring (replica 1)

The launcher exposes physical devices as CUDA_VISIBLE_DEVICES=0,1,2,3.
Inside the process, physical GPU2/3 are logical cuda:2/3. Ray registers only
two GPU resources, so FSDP and SGLang cannot allocate either dedicated tool GPU.

Use a versioned expert launcher, for example:
  bash examples/image_restoration_multi_agent/old_verl_grpo/scripts/fog/fog_v4_1_2_3gpu.sh

The dedicated-GPU topology fields are fixed and cannot be changed with Hydra
overrides. Other training hyperparameters can still be overridden normally.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXPERT="${1:-fog}"
case "${EXPERT}" in
  fog|low_light|rain|snow) ;;
  *)
    echo "Unsupported expert: ${EXPERT}" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ "${EXPERT}" == "low_light" ]]; then
  CONFIG_EXPERT="lowlight"
else
  CONFIG_EXPERT="${EXPERT}"
fi
CONFIG_VERSION="v4.1.2"
OUTPUT_VERSION="v4.1.2"

VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a VISIBLE_DEVICE_LIST <<<"${VISIBLE_DEVICES}"
if [[ "${#VISIBLE_DEVICE_LIST[@]}" != "4" || "${VISIBLE_DEVICE_LIST[0]}" != "0" || "${VISIBLE_DEVICE_LIST[1]}" != "1" || "${VISIBLE_DEVICE_LIST[2]}" != "2" || "${VISIBLE_DEVICE_LIST[3]}" != "3" ]]; then
  echo "The 3-GPU launcher requires OLD_VERL_CUDA_VISIBLE_DEVICES=0,1,2,3; got '${VISIBLE_DEVICES}'." >&2
  exit 2
fi

export OLD_VERL_CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
# This topology uses physical GPU2/3 on NUMA1 in addition to GPU0/1 on
# NUMA0; do not inherit the strict NUMA0 policy from the 2-GPU delegate.
export OLD_VERL_NUMA_BINDING=off
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-${CONFIG_EXPERT}_config_3gpu}"
export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${SCRIPT_DIR}/log/${CONFIG_EXPERT}/${OUTPUT_VERSION}/3gpu}"
export OLD_VERL_INTERMEDIATE_DIR="${OLD_VERL_INTERMEDIATE_DIR:-/home/LXJ/tmp/agent_lightning_old_verl_restoration_3gpu}"

CONFIG_PATH="${SCRIPT_DIR}/config/${CONFIG_EXPERT}/${CONFIG_VERSION}"
CONFIG_NAME="${OLD_VERL_CONFIG_NAME}"
HAS_CONFIG_PATH=0
HAS_CONFIG_NAME=0
PREVIOUS_ARG=""
HYDRA_OVERRIDES=()

PROTECTED_OVERRIDE_PATHS=(
  "trainer.nnodes"
  "trainer.n_gpus_per_node"
  "trainer.ray_kwargs.ray_init.num_gpus"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
  "actor_rollout_ref.rollout.tensor_model_parallel_size"
  "actor_rollout_ref.rollout.data_parallel_size"
  "actor_rollout_ref.rollout.pipeline_model_parallel_size"
  "actor_rollout_ref.rollout.agent.num_workers"
  "actor_rollout_ref.rollout.agent.num_gpus_per_worker"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path"
)

reject_protected_override() {
  local override="$1"
  local key protected

  [[ "${override}" == *=* ]] || return 0
  key="${override%%=*}"
  key="${key#+}"
  key="${key#+}"
  key="${key#\~}"

  for protected in "${PROTECTED_OVERRIDE_PATHS[@]}"; do
    if [[ \
      "${key}" == "${protected}" || \
      "${key}" == "${protected}."* || \
      "${protected}" == "${key}."* \
    ]]; then
      echo "Hydra override '${override}' would change the fixed 3-GPU topology (${protected})." >&2
      exit 2
    fi
  done
}

for arg in "$@"; do
  case "${arg}" in
    --config-path=*)
      CONFIG_PATH="${arg#*=}"
      HAS_CONFIG_PATH=1
      ;;
    --config-name=*)
      CONFIG_NAME="${arg#*=}"
      HAS_CONFIG_NAME=1
      ;;
    *)
      if [[ "${PREVIOUS_ARG}" == "--config-path" ]]; then
        CONFIG_PATH="${arg}"
        HAS_CONFIG_PATH=1
      elif [[ "${PREVIOUS_ARG}" == "--config-name" ]]; then
        CONFIG_NAME="${arg}"
        HAS_CONFIG_NAME=1
      elif [[ "${arg}" != "${EXPERT}" && "${arg}" != "--smoke" && "${arg}" != "--preflight" ]]; then
        reject_protected_override "${arg}"
        HYDRA_OVERRIDES+=("${arg}")
      fi
      ;;
  esac
  PREVIOUS_ARG="${arg}"
done

PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/verl/bin/python}"
"${PYTHON_BIN}" - "${ROOT}" "${CONFIG_PATH}" "${CONFIG_NAME}" "${VISIBLE_DEVICES}" "${HYDRA_OVERRIDES[@]}" <<'PY'
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

root = Path(sys.argv[1]).resolve()
config_dir = Path(sys.argv[2]).resolve()
config_name = sys.argv[3]
visible_devices = sys.argv[4]
overrides = sys.argv[5:]
with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    config = compose(config_name=config_name, overrides=overrides)

errors = []
if config.trainer.nnodes != 1:
    errors.append(f"trainer.nnodes={config.trainer.nnodes!r}, expected 1")
if config.trainer.n_gpus_per_node != 2:
    errors.append(f"trainer.n_gpus_per_node={config.trainer.n_gpus_per_node!r}, expected 2")
if config.trainer.ray_kwargs.ray_init.get("num_gpus") != 2:
    errors.append(f"ray_init.num_gpus={config.trainer.ray_kwargs.ray_init.get('num_gpus')!r}, expected 2")
if config.actor_rollout_ref.rollout.tensor_model_parallel_size != 2:
    errors.append(
        "actor_rollout_ref.rollout.tensor_model_parallel_size="
        f"{config.actor_rollout_ref.rollout.tensor_model_parallel_size!r}, expected 2"
    )
if config.actor_rollout_ref.rollout.data_parallel_size != 1:
    errors.append("SGLang data_parallel_size must remain 1 for one TP2 replica")
if config.actor_rollout_ref.rollout.pipeline_model_parallel_size != 1:
    errors.append("SGLang pipeline_model_parallel_size must remain 1")
if config.actor_rollout_ref.rollout.agent.num_workers != 1:
    errors.append("Exactly one AgentLoop worker is required to avoid duplicate tool runtimes on GPU2")
if config.actor_rollout_ref.rollout.agent.num_gpus_per_worker != 0.0:
    errors.append("AgentLoop must request zero Ray GPU resources")

runtime_env_vars = config.trainer.ray_kwargs.ray_init.runtime_env.env_vars
configured_visible = runtime_env_vars.get("CUDA_VISIBLE_DEVICES")
if configured_visible != visible_devices:
    errors.append(f"runtime CUDA_VISIBLE_DEVICES={configured_visible!r}, expected {visible_devices!r}")
if str(runtime_env_vars.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")) != "1":
    errors.append("runtime RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES must be '1'")

tool_path = Path(str(config.actor_rollout_ref.rollout.multi_turn.tool_config_path))
if not tool_path.is_absolute():
    tool_path = root / tool_path
tool_data = yaml.safe_load(tool_path.read_text(encoding="utf-8"))
tool = tool_data["tools"][0]["config"]
expected_tool_values = {
    "device": "cuda:2",
    "worker_devices": ["cuda:2", "cuda:3"],
    "model_devices": ["cuda:2", "cuda:3"],
    "iqa_devices": ["cuda:2", "cuda:3"],
    "iqa_device": "cuda:2",
    "preload": True,
    "auto_unload": False,
    "keep_models_loaded_between_sampling_steps": True,
}
for key, expected in expected_tool_values.items():
    actual = tool.get(key)
    if actual != expected:
        errors.append(f"tool {key}={actual!r}, expected {expected!r}")

if errors:
    raise SystemExit("Invalid dedicated-tool-GPU topology:\n- " + "\n- ".join(errors))
print(
    "Validated 3-GPU launcher topology: physical GPU0/1=train+TP2, "
    "physical GPU2/3=logical cuda:2/3 persistent restoration+IQA replicas"
)
PY

DELEGATE_ARGS=("$@")
if [[ "${HAS_CONFIG_PATH}" != "1" ]]; then
  DELEGATE_ARGS+=("--config-path=${CONFIG_PATH}")
fi
if [[ "${HAS_CONFIG_NAME}" != "1" ]]; then
  DELEGATE_ARGS+=("--config-name=${CONFIG_NAME}")
fi

exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" "${DELEGATE_ARGS[@]}"
