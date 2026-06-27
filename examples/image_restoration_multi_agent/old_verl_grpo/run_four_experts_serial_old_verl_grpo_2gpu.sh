#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
EXPERTS=(fog low_light rain snow)

export OLD_VERL_CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/four_experts_serial_${TIMESTAMP}.log"

# 将本脚本及所有子进程的输出同时写入终端和日志文件
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "===== serial run started at $(date) ====="
echo "Log file: ${MAIN_LOG}"

clear_intermediate_images=1
for expert in "${EXPERTS[@]}"; do
  echo "===== old-verl GRPO start: ${expert} on GPU ${OLD_VERL_CUDA_VISIBLE_DEVICES} ====="
  OLD_VERL_CLEAR_INTERMEDIATE_IMAGES="${clear_intermediate_images}" \
    "${SCRIPT_DIR}/run_expert_old_verl_grpo_2gpu.sh" "${expert}" "$@"
  echo "===== old-verl GRPO done: ${expert} ====="
  clear_intermediate_images=0
done

echo "===== serial run finished at $(date) ====="
