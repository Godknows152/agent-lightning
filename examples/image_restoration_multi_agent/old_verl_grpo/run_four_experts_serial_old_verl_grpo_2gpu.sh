#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="${SCRIPT_DIR}/log"
EXPERTS=(fog low_light rain snow)

export OLD_VERL_CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"

mkdir -p "${LOG_DIR}"
if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/four_experts_serial_${TIMESTAMP}.log"
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "$@" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started four-expert serial GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

MAIN_LOG="${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required for the background child}"
exec >>"${MAIN_LOG}" 2>&1

echo "===== serial run started at $(date) ====="
echo "PID: $$"
echo "Log file: ${MAIN_LOG}"

TOOL_INFO_LOG="${LOG_DIR}/restoration_tool_info.log"
TOOL_DEBUG_LOG="${LOG_DIR}/restoration_tools.log"
: > "${TOOL_INFO_LOG}"
: > "${TOOL_DEBUG_LOG}"
echo "Cleared tool logs:"
echo "  ${TOOL_INFO_LOG}"
echo "  ${TOOL_DEBUG_LOG}"

clear_intermediate_images=1
for expert in "${EXPERTS[@]}"; do
  echo "===== old-verl GRPO start: ${expert} on GPU ${OLD_VERL_CUDA_VISIBLE_DEVICES} ====="
  OLD_VERL_RUN_IN_FOREGROUND=1 \
    OLD_VERL_CLEAR_INTERMEDIATE_IMAGES="${clear_intermediate_images}" \
    OLD_VERL_CONFIG_NAME="${expert}_config_2gpu" \
    "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" "${expert}" "$@"
  echo "===== old-verl GRPO done: ${expert} ====="
  clear_intermediate_images=0
done

echo "===== serial run finished at $(date) ====="
