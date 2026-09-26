#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# The default 2B full run loads automatic resume and physical Ray node settings.
# An explicitly empty value opts out of this overlay.
if [[ "${ALFWORLD_MODEL_PROFILE:-qwen35_2b}" == "qwen35_2b" ]]; then
  export ALFWORLD_RESUME_CONFIG="${ALFWORLD_RESUME_CONFIG-qwen35_2b_gigpo}"
fi
exec env ALFWORLD_TRAINING_BACKEND=gigpo_grpo \
  bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" "$@"
