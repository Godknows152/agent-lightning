#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep smoke/pilot log and output directories distinct from the original profile.
KIND=full
[[ "${1:-}" == "--smoke" ]] && KIND=smoke
[[ "${1:-}" == "--pilot" ]] && KIND=pilot
export ALFWORLD_MODEL_PROFILE=qwen35_2b
export ALFWORLD_TRAINING_BACKEND=trajectory
export ALFWORLD_CONFIG_PATH="${ROOT}/config/alfworld/qwen35_2b/branch_entropy_1013"
export ALFWORLD_LOG_DIR="${ALFWORLD_LOG_DIR:-${ROOT}/log/alfworld/qwen35_2b/branch_entropy_1013/${KIND}}"
export ALFWORLD_OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR:-${ROOT}/outputs/qwen3.5_2B/branch_entropy_1013}"
if [[ "${KIND}" != full ]]; then
  export ALFWORLD_OUTPUT_DIR="${ALFWORLD_OUTPUT_DIR}/${KIND}_seed${SEED:-0}"
fi
exec bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" "$@"
