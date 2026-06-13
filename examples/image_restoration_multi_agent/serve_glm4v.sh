#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${GLM4V_MODEL_PATH:-/home/LXJ/Python_Projects/Models/GLM-4.1V-9B-Thinking}"
SERVED_MODEL_NAME="${GLM4V_SERVED_MODEL_NAME:-glm-4.1v-9b-thinking}"
HOST="${GLM4V_HOST:-127.0.0.1}"
PORT="${GLM4V_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GLM4V_GPU_MEMORY_UTILIZATION:-0.45}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"shortest_edge": 12544, "longest_edge": 47040000}, "fps": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
