#!/usr/bin/env bash
# Old-verl GRPO launcher for fog expert (v4.1.4)
# v4.1.4: Non-positive-advantage-gated legal-action first-token entropy with cosine decay (0.008 -> 0.0008).
set -euo pipefail

EXPERT="fog"
VERSION="v4.1.4"
CONFIG_VERSION="v4.1.4"
export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VERSION}}"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OLD_VERL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${OLD_VERL_DIR}/config/${EXPERT}/${CONFIG_VERSION}/${EXPERT}_config_2gpu.yaml"
export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/${EXPERT}/${VERSION}/2gpu}"
LOG_DIR="${OLD_VERL_LOG_DIR}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-pinned_use_cuda_host_register:True,pinned_num_register_threads:8,pinned_use_background_threads:True}"

# GPU0/1 are local to NUMA0. Bind the complete launcher so Ray and all workers
# inherit the CPU/memory policy; preferred memory may fall back if node0 is full.
PREFLIGHT_ONLY="${OLD_VERL_PREFLIGHT_ONLY:-0}"
for ARG in "$@"; do
  if [[ "${ARG}" == "--preflight" ]]; then
    PREFLIGHT_ONLY=1
    break
  fi
done
if [[ "${PREFLIGHT_ONLY}" != "1" && "${FOG_V414_NUMA_REEXEC:-0}" != "1" ]]; then
  if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl is required to bind fog v4.1.4 to NUMA0." >&2
    exit 1
  fi
  export FOG_V414_NUMA_REEXEC=1
  exec numactl --cpunodebind=0 --preferred=0 bash "${SCRIPT_PATH}" "$@"
fi

mkdir -p "${LOG_DIR}"

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${1:-}" != "--preflight" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_${VERSION}_${TIMESTAMP}.log"
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "$@" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} ${VERSION} GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  echo "Config: ${CONFIG_PATH}"
  exit 0
fi

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" == "1" ]]; then
  MAIN_LOG="${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required for the background child}"
  exec >>"${MAIN_LOG}" 2>&1
fi

echo "===== ${EXPERT} ${VERSION} run started at $(date) ====="
echo "PID: $$"
echo "Config: ${CONFIG_PATH}"

export OLD_VERL_RUN_IN_FOREGROUND=1
exec "${OLD_VERL_DIR}/run_expert_old_verl_grpo_2gpu.sh" \
  "${EXPERT}" \
  "$@" \
  "--config-path=${OLD_VERL_DIR}/config/${EXPERT}/${CONFIG_VERSION}" \
  "--config-name=${EXPERT}_config_2gpu"
