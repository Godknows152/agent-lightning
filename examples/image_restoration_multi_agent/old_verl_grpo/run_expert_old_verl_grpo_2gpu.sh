#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_expert_old_verl_grpo_2gpu.sh <fog|low_light|rain|snow> [--smoke] [hydra overrides...]

Environment overrides:
  OLD_VERL_SMOKE=1
  OLD_VERL_MODEL_PATH=/path/to/base-model
  OLD_VERL_ADAPTER_PATH=/path/to/sft-lora-adapter
  OLD_VERL_TOOL_REGISTRY_PATH=/path/to/tools.yaml
  OLD_VERL_CUDA_VISIBLE_DEVICES=0,1
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

case "${EXPERT}" in
  low_light) EXPERT_SLUG="low-light" ;;
  *) EXPERT_SLUG="${EXPERT}" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_DIR="${ROOT}/examples/image_restoration_multi_agent"
BACKEND_ROOT="${EXAMPLE_DIR}/verl_backend"
OLD_VERL_DIR="${EXAMPLE_DIR}/old_verl_grpo"
CONFIG_DIR="${OLD_VERL_DIR}/config"
CONVERTER="${OLD_VERL_DIR}/scripts/convert_current_jsonl_to_verl_parquet.py"
LOCAL_PYDEPS="${OLD_VERL_LOCAL_PYDEPS:-${OLD_VERL_DIR}/.pydeps}"
TOOL_REGISTRY_PATH="${OLD_VERL_TOOL_REGISTRY_PATH:-${EXAMPLE_DIR}/config/tools.yaml}"

PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/verl/bin/python}"
RAY_BIN="${RAY_BIN:-/home/LXJ/anaconda3/envs/verl/bin/ray}"
MODEL_PATH="${OLD_VERL_MODEL_PATH:-/home/LXJ/Python_Projects/Models/Qwen3.5-9B}"
ADAPTER_PATH="${OLD_VERL_ADAPTER_PATH:-${ROOT}/LlamaFactory/image_restoration_experts/outputs/qwen3_5/format_cold_start/${EXPERT}}"

TRAIN_PARQUET="${OLD_VERL_DIR}/data/${EXPERT}_train.parquet"
VAL_PARQUET="${OLD_VERL_DIR}/data/${EXPERT}_val.parquet"
RUN_KIND="full"
TRAIN_LIMIT_ARGS=()
VAL_LIMIT_ARGS=()
HYDRA_OVERRIDES=()
EXPERIMENT_NAME="${EXPERT_SLUG}-verl-2gpu"

if [[ "${SMOKE}" == "1" ]]; then
  RUN_KIND="smoke"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-smoke"
  TRAIN_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_TRAIN_LIMIT:-2}")
  VAL_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_VAL_LIMIT:-2}")
  HYDRA_OVERRIDES+=(
    data.train_batch_size=1
    data.val_max_samples=1
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.max_num_seqs=2
    actor_rollout_ref.rollout.max_model_len=4096
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    trainer.total_epochs=1
    trainer.total_training_steps=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=false
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
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "SFT adapter path not found: ${ADAPTER_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -d "${LOCAL_PYDEPS}" ]]; then
  export PYTHONPATH="${LOCAL_PYDEPS}:${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export SWANLAB_LOG_DIR="${OLD_VERL_SWANLAB_LOG_DIR:-${OLD_VERL_DIR}/outputs/${EXPERT}/${RUN_KIND}/swanlab}"
if [[ -n "${OLD_VERL_SWANLAB_MODE:-}" ]]; then
  export SWANLAB_MODE="${OLD_VERL_SWANLAB_MODE}"
elif [[ -n "${GRPO_SWANLAB_MODE:-}" ]]; then
  export SWANLAB_MODE="${GRPO_SWANLAB_MODE}"
elif [[ "${SMOKE}" == "1" ]]; then
  export SWANLAB_MODE="offline"
else
  export SWANLAB_MODE="cloud"
fi
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
export VERL_LOG_DIR="${OLD_VERL_DIR}/log"
export LD_LIBRARY_PATH="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"

mkdir -p "${OLD_VERL_DIR}/data" "${OLD_VERL_DIR}/log" "${OLD_VERL_DIR}/outputs/${EXPERT}/${RUN_KIND}" "${SWANLAB_LOG_DIR}"

"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split train --output "${TRAIN_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${TRAIN_LIMIT_ARGS[@]}"
"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split val --output "${VAL_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${VAL_LIMIT_ARGS[@]}"

if [[ "${OLD_VERL_STOP_RAY:-1}" == "1" ]]; then
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
fi

echo "Running old-verl ${RUN_KIND} GRPO for ${EXPERT} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Model:   ${MODEL_PATH}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Data:    ${TRAIN_PARQUET} / ${VAL_PARQUET}"
echo "SwanLab: project=image-restoration-multi-agent experiment=${EXPERIMENT_NAME} mode=${SWANLAB_MODE} log_dir=${SWANLAB_LOG_DIR}"

cd "${ROOT}"
"${PYTHON_BIN}" -u -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name restoration_expert_grpo_2gpu \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${ADAPTER_PATH}" \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name="image-restoration-multi-agent" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OLD_VERL_DIR}/outputs/${EXPERT}/${RUN_KIND}" \
  trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR}" \
  trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE="${SWANLAB_MODE}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
