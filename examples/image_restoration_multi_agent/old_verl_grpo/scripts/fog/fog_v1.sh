#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_VERL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

EXPERT="fog"
VERSION="v1"
CONFIG_PATH="${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}/${EXPERT}_config_2gpu.yaml"

export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/${EXPERT}/${VERSION}/2gpu}"
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-${EXPERT}_config_2gpu}"
export OLD_VERL_OUTPUT_DIR="${OLD_VERL_OUTPUT_DIR:-${OLD_VERL_DIR}/outputs/${EXPERT}/${VERSION}/2gpu}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_2gpu.sh" --help
fi

LOG_DIR="${OLD_VERL_LOG_DIR}"
mkdir -p "${LOG_DIR}"

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${1:-}" != "--preflight" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${VERSION}_${TIMESTAMP}.log"
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
exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "--config-path=${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}" \
  "--config-name=${EXPERT}_config_2gpu"
