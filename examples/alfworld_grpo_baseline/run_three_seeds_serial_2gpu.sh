#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(0 1 2)

if [[ "${ALFWORLD_SERIAL_CHILD:-0}" != "1" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  serial_log="${ROOT}/log/alfworld/v1/2gpu/serial_${timestamp}.log"
  mkdir -p "$(dirname "${serial_log}")"
  nohup env ALFWORLD_SERIAL_CHILD=1 ALFWORLD_FOREGROUND=1 \
    bash "${BASH_SOURCE[0]}" "$@" >"${serial_log}" 2>&1 </dev/null &
  echo "Started ALFWorld v1 seeds 0,1,2 serially (PID $!)."
  echo "Serial log: ${serial_log}"
  exit 0
fi

for seed in "${SEEDS[@]}"; do
  echo "===== starting ALFWorld v1 seed ${seed} at $(date) ====="
  SEED="${seed}" bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" "$@"
  echo "===== completed ALFWorld v1 seed ${seed} at $(date) ====="
done
