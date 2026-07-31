#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LOG_ROOT="${SCRIPT_DIR}/log"
EXPERTS=(fog low_light rain snow)

export OLD_VERL_CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1}"
export OLD_VERL_CLEAR_PENALIZED_SAMPLES="${OLD_VERL_CLEAR_PENALIZED_SAMPLES:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"

if [[ -n "${OLD_VERL_ADAPTER_PATH:-}" ]]; then
  echo "OLD_VERL_ADAPTER_PATH cannot be used for a four-expert run because each expert requires its own LoRA." >&2
  echo "Use OLD_VERL_SFT_ADAPTER_ROOT to override the root containing fog/snow/rain/low_light." >&2
  exit 2
fi

for expert in "${EXPERTS[@]}"; do
  mkdir -p "${LOG_ROOT}/${expert}/2gpu"
done
if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  nohup env \
    OLD_VERL_BACKGROUND_CHILD=1 \
    OLD_VERL_LOG_TIMESTAMP="${TIMESTAMP}" \
    bash "${SCRIPT_PATH}" "$@" \
    >/dev/null 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started four-expert serial GRPO in background (PID ${BACKGROUND_PID})."
  for expert in "${EXPERTS[@]}"; do
    echo "${expert} log file: ${LOG_ROOT}/${expert}/2gpu/${expert}_${TIMESTAMP}.log"
  done
  exit 0
fi

TIMESTAMP="${OLD_VERL_LOG_TIMESTAMP:?OLD_VERL_LOG_TIMESTAMP is required for the background child}"

clear_intermediate_images=1
for expert in "${EXPERTS[@]}"; do
  EXPERT_LOG="${LOG_ROOT}/${expert}/2gpu/${expert}_${TIMESTAMP}.log"
  {
    echo "===== old-verl GRPO start: ${expert} on GPU ${OLD_VERL_CUDA_VISIBLE_DEVICES} ====="
    echo "PID: $$"
    echo "Log file: ${EXPERT_LOG}"
    OLD_VERL_RUN_IN_FOREGROUND=1 \
      OLD_VERL_LOG_DIR="${LOG_ROOT}/${expert}/2gpu" \
      OLD_VERL_CLEAR_INTERMEDIATE_IMAGES="${clear_intermediate_images}" \
      OLD_VERL_CLEAR_PENALIZED_SAMPLES="${OLD_VERL_CLEAR_PENALIZED_SAMPLES}" \
      OLD_VERL_CONFIG_NAME="${expert}_config_2gpu" \
      "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" "${expert}" "$@"
    echo "===== old-verl GRPO done: ${expert} ====="
  } >>"${EXPERT_LOG}" 2>&1
  clear_intermediate_images=0
done
