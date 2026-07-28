#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EXPERT="${1:-fog}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${EXPERT}" in
  fog|low_light|rain|snow) ;;
  *)
    echo "Unsupported expert: ${EXPERT}" >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" --help
fi

VARIANT="v3"
export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${SCRIPT_DIR}/log/${EXPERT}/${VARIANT}}"
LOG_DIR="${OLD_VERL_LOG_DIR}"
TOOL_CONFIG_PATH="${SCRIPT_DIR}/config/tool_config/restoration_tool_config_marginal_efficiency_2gpu.yaml"
ACTION_RARITY_REWARD_COEFF="${OLD_VERL_ACTION_RARITY_REWARD_COEFF:-0.02}"
export OLD_VERL_EXPECT_ACTION_RARITY_REWARD_COEFF="${ACTION_RARITY_REWARD_COEFF}"
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-${EXPERT}_config_2gpu}"
export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VARIANT}}"
export OLD_VERL_OUTPUT_DIR="${OLD_VERL_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${EXPERT}/${VARIANT}}"

mkdir -p "${LOG_DIR}"
if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${1:-}" != "--preflight" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${VARIANT}_${TIMESTAMP}.log"
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "${EXPERT}" "$@" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} ${VARIANT} action-rarity GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" == "1" ]]; then
  MAIN_LOG="${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required for the background child}"
  exec >>"${MAIN_LOG}" 2>&1
fi

echo "===== ${EXPERT} ${VARIANT} action-rarity run started at $(date) ====="
echo "PID: $$"
echo "Tool config: ${TOOL_CONFIG_PATH}"
echo "Action rarity reward coefficient: ${ACTION_RARITY_REWARD_COEFF}"

export OLD_VERL_RUN_IN_FOREGROUND=1
exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH}" \
  "algorithm.action_rarity_reward_coeff=${ACTION_RARITY_REWARD_COEFF}"
