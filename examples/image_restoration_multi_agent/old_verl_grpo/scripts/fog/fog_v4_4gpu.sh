#!/usr/bin/env bash
# Old-verl GRPO launcher for fog expert (v4)
# v4: Thinking decision-point entropy regularization
set -euo pipefail

EXPERT="fog"
VERSION="v4"
export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VERSION}_4gpu}"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OLD_VERL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}/${EXPERT}_config_4gpu.yaml"
export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/${EXPERT}/${VERSION}/4gpu}"
LOG_DIR="${OLD_VERL_LOG_DIR}"

mkdir -p "${LOG_DIR}"

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${1:-}" != "--preflight" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${VERSION}_4gpu_${TIMESTAMP}.log"
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "$@" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} ${VERSION} GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  echo "Config: ${CONFIG_PATH}"
  exit 0
fi

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" == "1" ]]; then
  MAIN_LOG="${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required for the background child}"
  exec >>"${MAIN_LOG}" 2>&1
fi

echo "===== ${EXPERT} ${VERSION} run started at $(date) ====="
echo "PID: $$"
echo "Config: ${CONFIG_PATH}"

export OLD_VERL_RUN_IN_FOREGROUND=1
exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_4gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "--config-path=${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}" \
  "--config-name=${EXPERT}_config_4gpu"
