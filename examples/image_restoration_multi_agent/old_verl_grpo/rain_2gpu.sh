#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-rain_config_2gpu}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" --help
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/rain_${TIMESTAMP}.log"

# 将本脚本及所有子进程的输出同时写入终端和日志文件
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "===== rain run started at $(date) ====="
echo "Log file: ${MAIN_LOG}"

TOOL_INFO_LOG="${LOG_DIR}/restoration_tool_info.log"
TOOL_DEBUG_LOG="${LOG_DIR}/restoration_tools.log"
: > "${TOOL_INFO_LOG}"
: > "${TOOL_DEBUG_LOG}"
echo "Cleared tool logs:"
echo "  ${TOOL_INFO_LOG}"
echo "  ${TOOL_DEBUG_LOG}"

exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" rain "$@"
