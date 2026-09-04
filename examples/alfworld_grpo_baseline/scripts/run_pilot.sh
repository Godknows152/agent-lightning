#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${RUN_TRAINING:-0}" != "1" ]]; then
  exec bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" --preflight "$@"
fi
exec bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" --pilot "$@"
