#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${QWEN35_MODEL_PATH:-/home/LXJ/Python_Projects/Models/Qwen3.5-9B}"
SERVED_MODEL_NAME="${QWEN35_SERVED_MODEL_NAME:-qwen3.5-9b}"
HOST="${QWEN35_HOST:-127.0.0.1}"
PORT="${QWEN35_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${QWEN35_GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${QWEN35_MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${QWEN35_MAX_NUM_BATCHED_TOKENS:-1024}"
CHAT_TEMPLATE="${QWEN35_CHAT_TEMPLATE:-${SCRIPT_DIR}/grpo/templates/qwen35_hermes_nothink.jinja}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --chat-template "${CHAT_TEMPLATE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
