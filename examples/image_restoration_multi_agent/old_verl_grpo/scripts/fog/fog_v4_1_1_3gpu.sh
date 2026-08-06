#!/usr/bin/env bash
# Fog v4.1.1 3GPU launcher: GPU0/1 train and sample; physical GPU2/3 each
# provide one persistent restoration-model set plus IQA.
set -euo pipefail

EXPERT="fog"
VERSION="v4.1.1"
export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VERSION}}"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OLD_VERL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}/${EXPERT}_config_3gpu.yaml"
export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/${EXPERT}/${VERSION}/3gpu}"
LOG_DIR="${OLD_VERL_LOG_DIR}"

mkdir -p "${LOG_DIR}"

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${1:-}" != "--preflight" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${VERSION}_3gpu_${TIMESTAMP}.log"
  nohup env OLD_VERL_BACKGROUND_CHILD=1 OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "$@" >"${MAIN_LOG}" 2>&1 </dev/null &
  echo "Started ${EXPERT} ${VERSION} 3-GPU GRPO in background (PID $!)."
  echo "Log file: ${MAIN_LOG}"
  echo "Config: ${CONFIG_PATH}"
  exit 0
fi

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" == "1" ]]; then
  exec >>"${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required}" 2>&1
fi

export OLD_VERL_RUN_IN_FOREGROUND=1
exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_3gpu.sh" \
  "${EXPERT}" "$@" \
  "--config-path=${OLD_VERL_DIR}/config/${EXPERT}/${VERSION}" \
  "--config-name=${EXPERT}_config_3gpu"
