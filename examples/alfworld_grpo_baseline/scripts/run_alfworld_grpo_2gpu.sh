#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}"
SWANLAB_BIN="${SWANLAB_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/swanlab}"
MODEL_PROFILE="${ALFWORLD_MODEL_PROFILE:-qwen35_2b}"
CONFIG_NAME="alfworld_config_2gpu"
SEED="${SEED:-0}"
RUN_KIND="full"
FOREGROUND="${ALFWORLD_FOREGROUND:-0}"
TOTAL_STEPS="${ALFWORLD_TOTAL_STEPS:-150}"

case "${MODEL_PROFILE}" in
  qwen25_1_5b)
    MODEL_PATH="/home/LXJ/Python_Projects/Models/Qwen2.5-1.5B-Instruct"
    DATA_DIR="${ROOT}/data/qwen25_1_5b"
    CONFIG_PATH="${ROOT}/config/alfworld/qwen25_1_5b/v1"
    ;;
  qwen35_9b)
    MODEL_PATH="/home/LXJ/Python_Projects/Models/Qwen3.5-9B"
    DATA_DIR="${ROOT}/data/qwen35_9b"
    CONFIG_PATH="${ROOT}/config/alfworld/qwen35_9b/v1"
    ;;
  qwen35_2b)
    MODEL_PATH="/home/LXJ/Python_Projects/Models/Qwen3.5-2B"
    DATA_DIR="${ROOT}/data/qwen35_2b"
    CONFIG_PATH="${ROOT}/config/alfworld/qwen35_2b/v1"
    ;;
  *)
    echo "Unknown ALFWORLD_MODEL_PROFILE: ${MODEL_PROFILE}" >&2
    echo "Expected qwen25_1_5b, qwen35_2b, or qwen35_9b." >&2
    exit 2
    ;;
esac

usage() {
  cat <<'EOF'
Usage:
  run_alfworld_grpo_2gpu.sh [--preflight|--smoke|--pilot] [Hydra overrides...]

Environment:
  ALFWORLD_MODEL_PROFILE=qwen25_1_5b|qwen35_2b|qwen35_9b  Select an isolated model profile
  SEED=0|1|2                 Output/checkpoint seed directory (default: 0)
  ALFWORLD_FOREGROUND=1      Keep launcher attached; default is background
  ALFWORLD_TOTAL_STEPS=150   Full-run step count
  ALFWORLD_SWANLAB_MODE=cloud|offline|local
  ALFWORLD_SWANLAB_LOG_DIR=/path/to/swanlab
  ALFWORLD_LOG_DIR=/path/to/log
  ALFWORLD_OUTPUT_DIR=/path/to/output
  CUDA_VISIBLE_DEVICES=0,1   Physical GPUs (default: 0,1)
  ALFWORLD_SKIP_SWANLAB_VERIFY=1  Skip cloud credential verification
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--preflight" ]]; then RUN_KIND="preflight"; shift; fi
if [[ "${1:-}" == "--smoke" ]]; then RUN_KIND="smoke"; shift; fi
if [[ "${1:-}" == "--pilot" ]]; then RUN_KIND="pilot"; shift; fi

OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR:-${ROOT}/outputs/alfworld/${MODEL_PROFILE}/v1/2gpu/seed${SEED}}"
LOG_DIR="${ALFWORLD_LOG_DIR:-${ROOT}/log/alfworld/${MODEL_PROFILE}/v1/2gpu/seed${SEED}}"
SWANLAB_LOG_DIR="${ALFWORLD_SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlab}"
SWANLAB_MODE="${ALFWORLD_SWANLAB_MODE:-cloud}"

if [[ "${RUN_KIND}" == "smoke" ]]; then
  TOTAL_STEPS=1
  OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR:-${ROOT}/outputs/alfworld/${MODEL_PROFILE}/v1/2gpu/smoke_seed${SEED}}"
  LOG_DIR="${ALFWORLD_LOG_DIR:-${ROOT}/log/alfworld/${MODEL_PROFILE}/v1/2gpu/smoke_seed${SEED}}"
  SWANLAB_LOG_DIR="${ALFWORLD_SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlab}"
  SWANLAB_MODE="${ALFWORLD_SWANLAB_MODE:-offline}"
elif [[ "${RUN_KIND}" == "pilot" ]]; then
  TOTAL_STEPS="${ALFWORLD_TOTAL_STEPS:-5}"
  OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR:-${ROOT}/outputs/alfworld/${MODEL_PROFILE}/v1/2gpu/pilot_seed${SEED}}"
  LOG_DIR="${ALFWORLD_LOG_DIR:-${ROOT}/log/alfworld/${MODEL_PROFILE}/v1/2gpu/pilot_seed${SEED}}"
  SWANLAB_LOG_DIR="${ALFWORLD_SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlab}"
  SWANLAB_MODE="${ALFWORLD_SWANLAB_MODE:-offline}"
fi

if [[ "${RUN_KIND}" != "preflight" && "${FOREGROUND}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  main_log="${LOG_DIR}/alfworld_v1_seed${SEED}_${timestamp}.log"
  child_args=()
  [[ "${RUN_KIND}" == "smoke" ]] && child_args+=(--smoke)
  [[ "${RUN_KIND}" == "pilot" ]] && child_args+=(--pilot)
  nohup env ALFWORLD_FOREGROUND=1 bash "${BASH_SOURCE[0]}" \
    "${child_args[@]}" "$@" >"${main_log}" 2>&1 </dev/null &
  echo "Started ALFWorld v1 ${RUN_KIND} in background (PID $!)."
  echo "Log: ${main_log}"
  echo "Output: ${OUTPUT_DIR}"
  exit 0
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/contrib/recipes/envs/agl_envs/alfworld/alfworld_source}"
export ALFWORLD_MODEL="${MODEL_PATH}"
export ALFWORLD_DATASET_DIR="${DATA_DIR}"
export ALFWORLD_CC="${ALFWORLD_CC:-/usr/bin/gcc-10}"
export ALFWORLD_CXX="${ALFWORLD_CXX:-/usr/bin/g++-10}"
export CC="${ALFWORLD_CC}"
export CXX="${ALFWORLD_CXX}"
export CUDAHOSTCXX="${ALFWORLD_CXX}"
export NVCC_CCBIN="${ALFWORLD_CXX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${ROOT}/src:${PROJECT_ROOT}/examples/image_restoration_multi_agent/old_verl_grpo/.pydeps:${PROJECT_ROOT}/examples/image_restoration_multi_agent/verl_backend:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export SWANLAB_MODE SWANLAB_LOG_DIR
export VERL_LOG_DIR="${LOG_DIR}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${SWANLAB_LOG_DIR}"

bash "${ROOT}/scripts/preflight.sh"
if [[ "${SWANLAB_MODE}" == "cloud" && "${ALFWORLD_SKIP_SWANLAB_VERIFY:-0}" != "1" ]]; then
  if [[ ! -x "${SWANLAB_BIN}" ]]; then
    echo "SwanLab executable not found: ${SWANLAB_BIN}" >&2
    exit 2
  fi
  "${SWANLAB_BIN}" verify
fi
if [[ "${RUN_KIND}" == "preflight" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/preflight_training.py" \
    --seed "${SEED}" --kind full \
    --output-dir "${OUTPUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --swanlab-log-dir "${SWANLAB_LOG_DIR}" \
    --swanlab-mode "${SWANLAB_MODE}" \
    --config-dir "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    --model-profile "${MODEL_PROFILE}"
  echo "ALFWorld ${MODEL_PROFILE} v1 preflight passed: output=${OUTPUT_DIR} log=${LOG_DIR} swanlab=${SWANLAB_LOG_DIR}"
  exit 0
fi

training_kind="full"
[[ "${RUN_KIND}" == "smoke" ]] && training_kind="smoke"
[[ "${RUN_KIND}" == "pilot" ]] && training_kind="pilot"
"${PYTHON_BIN}" "${ROOT}/scripts/preflight_training.py" \
  --seed "${SEED}" --kind "${training_kind}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --swanlab-log-dir "${SWANLAB_LOG_DIR}" \
  --swanlab-mode "${SWANLAB_MODE}" \
  --config-dir "${CONFIG_PATH}" \
  --config-name "${CONFIG_NAME}" \
  --model-profile "${MODEL_PROFILE}" >/dev/null

if [[ ! -s "${DATA_DIR}/train.parquet" || ! -s "${DATA_DIR}/test.parquet" ]]; then
  echo "Missing ${DATA_DIR}/train.parquet or test.parquet" >&2
  exit 2
fi

EXPERIMENT_NAME="alfworld_${MODEL_PROFILE}_v1_seed${SEED}"
if [[ "${RUN_KIND}" == "smoke" ]]; then
  EXPERIMENT_NAME="alfworld_${MODEL_PROFILE}_v1_smoke_seed${SEED}"
elif [[ "${RUN_KIND}" == "pilot" ]]; then
  EXPERIMENT_NAME="alfworld_${MODEL_PROFILE}_v1_pilot_seed${SEED}"
fi

for override in "$@"; do
  case "${override}" in
    trainer.experiment_name=*|trainer.default_local_dir=*|trainer.project_name=*|trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR=*|trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR=*)
      echo "Use ALFWORLD_OUTPUT_DIR/LOG_DIR/SWANLAB_LOG_DIR or the versioned YAML for naming paths; rejected: ${override}" >&2
      exit 2
      ;;
  esac
done

overrides=(
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE=${SWANLAB_MODE}"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR}"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR=${LOG_DIR}"
  "variables.SEED=${SEED}"
)
if [[ "${RUN_KIND}" == "smoke" || "${RUN_KIND}" == "pilot" ]]; then
  overrides+=(
    "variables.NUM_ROLLOUTS=2"
    "data.train_batch_size=8"
    "data.val_batch_size=8"
    "actor_rollout_ref.actor.ppo_mini_batch_size=8"
    # Keep pilot memory settings identical across model profiles.
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
  )
fi

echo "===== ALFWorld ${MODEL_PROFILE} v1 ${RUN_KIND} start $(date) ====="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_DIR}"
echo "SwanLab: project=ALFWorldRL experiment=${EXPERIMENT_NAME} mode=${SWANLAB_MODE} log_dir=${SWANLAB_LOG_DIR}"
exec "${PYTHON_BIN}" -u -m alfworld_baseline.main_ppo \
  --config-path "${CONFIG_PATH}" \
  --config-name "${CONFIG_NAME}" \
  "${overrides[@]}" "$@"
