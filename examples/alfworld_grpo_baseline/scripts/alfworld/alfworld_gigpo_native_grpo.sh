#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}"
export ALFWORLD_GIGPO_ROOT="${ALFWORLD_GIGPO_ROOT:-${ROOT}/.runtime/gigpo-native}"
export ALFWORLD_VERL_ROOT="${ALFWORLD_GIGPO_ROOT}"
export ALFWORLD_GIGPO_DATA="${ROOT}/data/gigpo_native"
export ALFWORLD_OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR:-${ROOT}/outputs/qwen3.5_2B/gigpo_native_grpo/full_sft}"
LOG_DIR="${ALFWORLD_LOG_DIR:-${ROOT}/log/alfworld/qwen35_2b/gigpo_native_grpo/full_sft/2gpu/seed0}"
export ALFWORLD_RAY_TEMP_DIR="${ALFWORLD_RAY_TEMP_DIR:-/home/LXJ/tmp/ray-gigpo}"
export SWANLAB_LOG_DIR="${ALFWORLD_OUTPUT_DIR}/swanlab"
export SWANLAB_MODE="${ALFWORLD_SWANLAB_MODE:-cloud}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/contrib/recipes/envs/agl_envs/alfworld/alfworld_source}"
export PYTHONPATH="${ROOT}/src:${ALFWORLD_GIGPO_ROOT}:${PROJECT_ROOT}/examples/image_restoration_multi_agent/old_verl_grpo/.pydeps"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10 NVCC_CCBIN=/usr/bin/g++-10
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
# This machine uses the legacy sgl-kernel 0.3.21 package also used by the
# existing Qwen3.5 profile. Validate actual model execution, not the renamed
# sglang-kernel package's version gate. No installed package is modified.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: alfworld_gigpo_native_grpo.sh [--preflight] [Hydra overrides...]'
  echo 'Default: background, 2 GPUs, full SFT model, upstream GiGPO GRPO, new isolated outputs.'
  echo 'ALFWORLD_FOREGROUND=1 attaches the process; ALFWORLD_OUTPUT_DIR/ALFWORLD_LOG_DIR override storage.'
  exit 0
fi
"${PYTHON_BIN}" "${ROOT}/scripts/prepare_gigpo_native.py"
if [[ "${1:-}" == "--preflight" ]]; then
  shift
  exec "${PYTHON_BIN}" -m alfworld_baseline.gigpo_native_main \
    --config-path "${ROOT}/config/alfworld/qwen35_2b/gigpo_native" --cfg job --resolve "$@"
fi
mkdir -p "${LOG_DIR}" "${ALFWORLD_OUTPUT_DIR}" "${SWANLAB_LOG_DIR}" "${ALFWORLD_RAY_TEMP_DIR}"
if [[ "${ALFWORLD_FOREGROUND:-0}" != 1 ]]; then
  main_log="${LOG_DIR}/alfworld_gigpo_native_$(date +%Y%m%d_%H%M%S).log"
  nohup setsid env ALFWORLD_FOREGROUND=1 bash "${BASH_SOURCE[0]}" "$@" >"${main_log}" 2>&1 </dev/null &
  pid=$!
  echo "${pid}" >"${ALFWORLD_OUTPUT_DIR}/launcher.pid"
  echo "${main_log}" >"${ALFWORLD_OUTPUT_DIR}/launcher.logpath"
  echo "Started GiGPO native GRPO: PID=${pid}, log=${main_log}, output=${ALFWORLD_OUTPUT_DIR}"
  exit 0
fi
cd "${ALFWORLD_GIGPO_ROOT}"
exec "${PYTHON_BIN}" -m alfworld_baseline.gigpo_native_main \
  --config-path "${ROOT}/config/alfworld/qwen35_2b/gigpo_native" "$@"
