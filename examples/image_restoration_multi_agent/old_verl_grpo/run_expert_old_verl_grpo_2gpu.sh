#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_expert_old_verl_grpo_2gpu.sh <fog|low_light|rain|snow> [--smoke] [hydra overrides...]

All non-help invocations start in the background. The command prints the PID
and writes stdout/stderr only to old_verl_grpo/log/<expert>_<timestamp>.log.

Default fog resume run:
  bash examples/image_restoration_multi_agent/old_verl_grpo/run_expert_old_verl_grpo_2gpu.sh fog

Environment overrides:
  OLD_VERL_SMOKE=1
  OLD_VERL_MODEL_PATH=/path/to/base-model
  OLD_VERL_ADAPTER_PATH=/path/to/sft-lora-adapter
  OLD_VERL_TOOL_REGISTRY_PATH=/path/to/tools.yaml
  OLD_VERL_CONFIG_NAME=fog_config_2gpu  # defaults to <expert>_config_2gpu
  OLD_VERL_CUDA_VISIBLE_DEVICES=0,1
  OLD_VERL_CLEAR_INTERMEDIATE_IMAGES=1
  OLD_VERL_INTERMEDIATE_DIR=/home/LXJ/tmp/agent_lightning_old_verl_restoration
  OLD_VERL_SHOW_KNOWN_WARNINGS=1  # show otherwise-suppressed third-party warning spam
  OLD_VERL_OUTPUT_SUFFIX=0702  # optional trainer output/name override; YAML wins when unset
  OLD_VERL_RUN_TAG=fog_0702   # optional trainer output override; YAML wins when unset
  OLD_VERL_OUTPUT_DIR=/path/to/output-dir  # optional trainer output override; YAML wins when unset
  PYTHON_BIN=/home/LXJ/anaconda3/envs/verl/bin/python
  RAY_BIN=/home/LXJ/anaconda3/envs/verl/bin/ray
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXPERT="${1:-fog}"
if [[ $# -gt 0 ]]; then
  shift
fi

SMOKE="${OLD_VERL_SMOKE:-0}"
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

case "${EXPERT}" in
  fog|low_light|rain|snow) ;;
  *)
    echo "Unsupported expert: ${EXPERT}" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_DIR="${ROOT}/examples/image_restoration_multi_agent"
BACKEND_ROOT="${EXAMPLE_DIR}/verl_backend"
OLD_VERL_DIR="${EXAMPLE_DIR}/old_verl_grpo"
LOG_DIR="${OLD_VERL_DIR}/log"
CONFIG_DIR="${OLD_VERL_DIR}/config"
CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-${EXPERT}_config_2gpu}"
CONVERTER="${OLD_VERL_DIR}/scripts/convert_current_jsonl_to_verl_parquet.py"
LOCAL_PYDEPS="${OLD_VERL_LOCAL_PYDEPS:-${OLD_VERL_DIR}/.pydeps}"
TOOL_REGISTRY_PATH="${OLD_VERL_TOOL_REGISTRY_PATH:-${EXAMPLE_DIR}/config/tools.yaml}"
INTERMEDIATE_DIR="${OLD_VERL_INTERMEDIATE_DIR:-/home/LXJ/tmp/agent_lightning_old_verl_restoration}"

PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/verl/bin/python}"
RAY_BIN="${RAY_BIN:-/home/LXJ/anaconda3/envs/verl/bin/ray}"
MODEL_PATH_OVERRIDE="${OLD_VERL_MODEL_PATH:-}"
ADAPTER_PATH_OVERRIDE="${OLD_VERL_ADAPTER_PATH:-}"

mkdir -p "${LOG_DIR}"
if [[ "${OLD_VERL_RUN_IN_FOREGROUND:-0}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${TIMESTAMP}.log"
  CHILD_ARGS=("${EXPERT}")
  if [[ "${SMOKE}" == "1" ]]; then
    CHILD_ARGS+=(--smoke)
  fi
  CHILD_ARGS+=("$@")
  nohup env \
    OLD_VERL_RUN_IN_FOREGROUND=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "${CHILD_ARGS[@]}" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

TRAIN_PARQUET="${OLD_VERL_DIR}/data/${EXPERT}_train.parquet"
VAL_PARQUET="${OLD_VERL_DIR}/data/${EXPERT}_val.parquet"
RUN_KIND="full"
TRAIN_LIMIT_ARGS=()
VAL_LIMIT_ARGS=()
CONFIG_OVERRIDES=()
HYDRA_OVERRIDES=()
PROJECT_NAME_OVERRIDE="${OLD_VERL_PROJECT_NAME:-}"
EXPERIMENT_NAME_OVERRIDE="${OLD_VERL_EXPERIMENT_NAME:-}"
OUTPUT_DIR_OVERRIDE="${OLD_VERL_OUTPUT_DIR:-}"
SWANLAB_LOG_DIR_OVERRIDE="${OLD_VERL_SWANLAB_LOG_DIR:-}"
SWANLAB_MODE_OVERRIDE=""

if [[ -z "${OUTPUT_DIR_OVERRIDE}" ]]; then
  if [[ -n "${OLD_VERL_RUN_TAG:-}" ]]; then
    OUTPUT_DIR_OVERRIDE="${OLD_VERL_DIR}/outputs/${OLD_VERL_RUN_TAG}"
  elif [[ -n "${OLD_VERL_OUTPUT_SUFFIX:-}" ]]; then
    OUTPUT_DIR_OVERRIDE="${OLD_VERL_DIR}/outputs/${EXPERT}_${OLD_VERL_OUTPUT_SUFFIX}"
  fi
fi
if [[ -z "${EXPERIMENT_NAME_OVERRIDE}" && -n "${OLD_VERL_OUTPUT_SUFFIX:-}" ]]; then
  EXPERIMENT_NAME_OVERRIDE="${EXPERT}_grpo_${OLD_VERL_OUTPUT_SUFFIX}"
fi
if [[ -z "${SWANLAB_LOG_DIR_OVERRIDE}" && -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  SWANLAB_LOG_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE}/swanlab"
fi

if [[ "${SMOKE}" == "1" ]]; then
  RUN_KIND="smoke"
  if [[ -n "${EXPERIMENT_NAME_OVERRIDE}" ]]; then
    EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE}-smoke"
  fi
  TRAIN_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_TRAIN_LIMIT:-2}")
  VAL_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_VAL_LIMIT:-2}")
  SWANLAB_MODE_OVERRIDE="offline"
  HYDRA_OVERRIDES+=(
    "data.train_batch_size=1"
    "data.val_max_samples=1"
    "actor_rollout_ref.rollout.n=2"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
    "actor_rollout_ref.rollout.max_num_seqs=2"
    "actor_rollout_ref.rollout.max_model_len=4096"
    "actor_rollout_ref.rollout.enforce_eager=true"
    "actor_rollout_ref.actor.ppo_mini_batch_size=1"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=1"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "trainer.val_before_train=false"
  )
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${RAY_BIN}" ]]; then
  echo "Ray executable not found or not executable: ${RAY_BIN}" >&2
  exit 1
fi
if [[ -n "${MODEL_PATH_OVERRIDE}" && ! -d "${MODEL_PATH_OVERRIDE}" ]]; then
  echo "Model path not found: ${MODEL_PATH_OVERRIDE}" >&2
  exit 1
fi
if [[ -n "${ADAPTER_PATH_OVERRIDE}" && ! -d "${ADAPTER_PATH_OVERRIDE}" ]]; then
  echo "SFT adapter path not found: ${ADAPTER_PATH_OVERRIDE}" >&2
  exit 1
fi

if [[ -n "${MODEL_PATH_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("actor_rollout_ref.model.path=${MODEL_PATH_OVERRIDE}")
fi
if [[ -n "${ADAPTER_PATH_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("actor_rollout_ref.model.lora_adapter_path=${ADAPTER_PATH_OVERRIDE}")
fi
if [[ -n "${PROJECT_NAME_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.project_name=${PROJECT_NAME_OVERRIDE}")
fi
if [[ -n "${EXPERIMENT_NAME_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.experiment_name=${EXPERIMENT_NAME_OVERRIDE}")
fi
if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.default_local_dir=${OUTPUT_DIR_OVERRIDE}")
fi
if [[ -n "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR_OVERRIDE}")
fi
if [[ -n "${OLD_VERL_SWANLAB_MODE:-}" ]]; then
  SWANLAB_MODE_OVERRIDE="${OLD_VERL_SWANLAB_MODE}"
elif [[ -n "${GRPO_SWANLAB_MODE:-}" ]]; then
  SWANLAB_MODE_OVERRIDE="${GRPO_SWANLAB_MODE}"
fi
if [[ -n "${SWANLAB_MODE_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE=${SWANLAB_MODE_OVERRIDE}")
fi

# Expose only physical GPU0/1 to the trainer, rollout, restoration, and IQA
# workers. The zero-GPU AgentLoop does not require an additional visible GPU.
export CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -d "${LOCAL_PYDEPS}" ]]; then
  export PYTHONPATH="${LOCAL_PYDEPS}:${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Hide harmless import-time warnings from optional CUDA/NPU dependencies. The
# high-volume LoRA warning is filtered locally around adapter loading because
# its message starts with a dynamic parameter name. Keep all other warnings visible.
if [[ "${OLD_VERL_SHOW_KNOWN_WARNINGS:-0}" != "1" ]]; then
  KNOWN_WARNING_FILTERS="ignore:The pynvml package is deprecated:FutureWarning,ignore:NPU not support router replay for now:UserWarning"
  export PYTHONWARNINGS="${KNOWN_WARNING_FILTERS}${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"
fi
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
if [[ -n "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR_OVERRIDE}"
fi
if [[ -n "${SWANLAB_MODE_OVERRIDE}" ]]; then
  export SWANLAB_MODE="${SWANLAB_MODE_OVERRIDE}"
fi
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
export VERL_LOG_DIR="${OLD_VERL_DIR}/log"
export LD_LIBRARY_PATH="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"

mkdir -p "${OLD_VERL_DIR}/data" "${OLD_VERL_DIR}/log"
if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  mkdir -p "${OUTPUT_DIR_OVERRIDE}"
fi
if [[ -n "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  mkdir -p "${SWANLAB_LOG_DIR_OVERRIDE}"
fi

if [[ "${OLD_VERL_CLEAR_INTERMEDIATE_IMAGES:-1}" == "1" ]]; then
  if [[ -z "${INTERMEDIATE_DIR}" || "${INTERMEDIATE_DIR}" == "/" ]]; then
    echo "Refusing to clear an unsafe intermediate image directory: '${INTERMEDIATE_DIR}'" >&2
    exit 2
  fi
  rm -rf -- "${INTERMEDIATE_DIR}"
  mkdir -p -- "${INTERMEDIATE_DIR}"
  echo "Cleared sampling intermediate images: ${INTERMEDIATE_DIR}"
fi

"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split train --output "${TRAIN_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${TRAIN_LIMIT_ARGS[@]}"
"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split val --output "${VAL_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${VAL_LIMIT_ARGS[@]}"

if [[ "${OLD_VERL_STOP_RAY:-1}" == "1" ]]; then
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
fi

echo "Running old-verl ${RUN_KIND} GRPO for ${EXPERT} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Model:   ${MODEL_PATH_OVERRIDE:-configured by YAML}"
echo "Adapter: ${ADAPTER_PATH_OVERRIDE:-configured by YAML}"
echo "Data:    ${TRAIN_PARQUET} / ${VAL_PARQUET}"
echo "Output:  ${OUTPUT_DIR_OVERRIDE:-configured by YAML}"
echo "Config:  ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "Resume:  configured by ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
printf 'SwanLab: project=%s experiment=%s mode=%s log_dir=%s\n' \
  "${PROJECT_NAME_OVERRIDE:-configured by YAML}" \
  "${EXPERIMENT_NAME_OVERRIDE:-configured by YAML}" \
  "${SWANLAB_MODE_OVERRIDE:-configured by YAML}" \
  "${SWANLAB_LOG_DIR_OVERRIDE:-configured by YAML}"

cd "${ROOT}"
"${PYTHON_BIN}" -u -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  "${CONFIG_OVERRIDES[@]}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
