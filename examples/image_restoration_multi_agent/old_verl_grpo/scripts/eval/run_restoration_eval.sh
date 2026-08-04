#!/usr/bin/env bash
# Background launcher for run_restoration_eval.py
# (unified LoRA/SGLang or JarvisIR restoration inference).
#
# Usage:
#   bash scripts/eval/run_restoration_eval.sh [hydra overrides...]          # background
#   bash scripts/eval/run_restoration_eval.sh --preflight [overrides...]    # foreground preflight
#   bash scripts/eval/run_restoration_eval.sh --foreground [overrides...]   # foreground run
#
# Example:
#   bash scripts/eval/run_restoration_eval.sh backend=lora_sglang \
#     run.name=v2 \
#     backend.adapter_path=/home/LXJ/Python_Projects/Agent_Lightning/examples/image_restoration_multi_agent/old_verl_grpo/outputs/fog/LoRA/v2
#
# Logs: log/eval/restoration_eval_<timestamp>.log
# Override log root with OLD_VERL_LOG_DIR, python with PYTHON_BIN.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_VERL_DIR="$(cd "${EVAL_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${OLD_VERL_DIR}/../../.." && pwd)"
LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/eval}"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/verl/bin/python}"
ENTRYPOINT="${EVAL_DIR}/run_restoration_eval.py"

mkdir -p "${LOG_DIR}"

FOREGROUND=0
ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --foreground)
      FOREGROUND=1
      ;;
    --preflight)
      FOREGROUND=1
      ARGS+=(command=preflight)
      ;;
    *)
      ARGS+=("${arg}")
      ;;
  esac
done

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" != "1" && "${FOREGROUND}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/restoration_eval_${TIMESTAMP}.log"
  nohup env OLD_VERL_BACKGROUND_CHILD=1 OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "${ARGS[@]}" >"${MAIN_LOG}" 2>&1 </dev/null &
  echo "Started restoration eval in background (PID $!)."
  echo "Log file: ${MAIN_LOG}"
  echo "Entrypoint: ${ENTRYPOINT}"
  exit 0
fi

if [[ "${OLD_VERL_BACKGROUND_CHILD:-0}" == "1" ]]; then
  exec >>"${OLD_VERL_MAIN_LOG:?OLD_VERL_MAIN_LOG is required}" 2>&1
fi

echo "===== restoration eval started at $(date) ====="
echo "PID: $$"
echo "Entrypoint: ${ENTRYPOINT}"
echo "Overrides: ${ARGS[*]:-}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${ENTRYPOINT}" "${ARGS[@]}"
