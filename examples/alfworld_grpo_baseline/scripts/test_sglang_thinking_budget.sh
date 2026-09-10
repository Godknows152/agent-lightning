#!/usr/bin/env bash
# Independent two-GPU diagnostic; never starts Ray/PPO or uploads to SwanLab.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PROJECT_ROOT}/examples/image_restoration_multi_agent/old_verl_grpo/.pydeps:${PROJECT_ROOT}/examples/image_restoration_multi_agent/verl_backend:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/contrib/recipes/envs/agl_envs/alfworld/alfworld_source}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CC="${CC:-/usr/bin/gcc-10}" CXX="${CXX:-/usr/bin/g++-10}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}" NVCC_CCBIN="${NVCC_CCBIN:-${CXX}}"
OUTPUT="${ALFWORLD_BUDGET_TEST_OUTPUT:-${ROOT}/outputs/diagnostics/qwen35_9b/thinking_budget_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT}"
echo "Diagnostic output: ${OUTPUT}; requires two free GPUs (${CUDA_VISIBLE_DEVICES})."
"${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}" -u \
  "${ROOT}/scripts/test_sglang_thinking_budget.py" --output "${OUTPUT}" "$@" 2>&1 | tee "${OUTPUT}/run.log"
