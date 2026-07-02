#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OLD_VERL_CONFIG_NAME="${OLD_VERL_CONFIG_NAME:-rain_config_2gpu}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" --help
fi

exec "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" rain "$@"
