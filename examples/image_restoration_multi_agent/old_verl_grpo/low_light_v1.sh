#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="${SCRIPT_DIR}/log"
EXPERT="low_light"
REWARD_VARIANT="v1"
TOOL_CONFIG_PATH="${SCRIPT_DIR}/config/tool_config/restoration_tool_config_current_iqa_2gpu.yaml"
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-low_light_config_2gpu}"
export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${REWARD_VARIANT}}"
export OLD_VERL_OUTPUT_DIR="${OLD_VERL_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${EXPERT}_${REWARD_VARIANT}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" --help
fi

mkdir -p "${LOG_DIR}"
if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${REWARD_VARIANT}_${TIMESTAMP}.log"
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "$@" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} ${REWARD_VARIANT} GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

MAIN_LOG="${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required for the background child}"
exec >>"${MAIN_LOG}" 2>&1

echo "===== ${EXPERT} ${REWARD_VARIANT} run started at $(date) ====="
echo "PID: $$"
echo "Log file: ${MAIN_LOG}"
echo "Tool config: ${TOOL_CONFIG_PATH}"

TOOL_INFO_LOG="${LOG_DIR}/restoration_tool_info.log"
TOOL_DEBUG_LOG="${LOG_DIR}/restoration_tools.log"
: > "${TOOL_INFO_LOG}"
: > "${TOOL_DEBUG_LOG}"
echo "Cleared tool logs:"
echo "  ${TOOL_INFO_LOG}"
echo "  ${TOOL_DEBUG_LOG}"

export OLD_VERL_RUN_IN_FOREGROUND=1
exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}"
