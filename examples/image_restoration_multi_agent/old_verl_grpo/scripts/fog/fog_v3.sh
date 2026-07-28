#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_VERL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec "${OLD_VERL_DIR}/run_action_rarity_v3_old_verl_grpo_2gpu.sh" fog "$@"
