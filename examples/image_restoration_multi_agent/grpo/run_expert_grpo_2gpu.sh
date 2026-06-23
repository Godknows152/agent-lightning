#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/LXJ/Python_Projects/Agent_Lightning"
GRPO_CONDA_ENV="${GRPO_CONDA_ENV:-qwen35-agentlightning-rl}"
GRPO_ENV_DIR="${GRPO_ENV_DIR:-/home/LXJ/anaconda3/envs/${GRPO_CONDA_ENV}}"

if [[ ! -d "$GRPO_ENV_DIR" ]]; then
  echo "GRPO conda environment directory does not exist: $GRPO_ENV_DIR" >&2
  exit 1
fi

export GRPO_CONDA_ENV
export GRPO_ENV_DIR
export GRPO_TOPOLOGY="2gpu"
export GRPO_CONFIG_DIR="examples/image_restoration_multi_agent/grpo/configs_2gpu"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export IMAGE_RESTORATION_DEVICES="cuda:0,cuda:1"
export IMAGE_RESTORATION_IQA_DEVICES="cuda:0,cuda:1"
export IMAGE_RESTORATION_WORKERS_PER_DEVICE="1"
export IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE="2"
export GRPO_OFFLINE="${GRPO_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/LXJ/tmp/ray_qwen35_grpo_2gpu}"
export AGENTLIGHTNING_FORCE_LOCAL_ROLLOUT_SERVER="${AGENTLIGHTNING_FORCE_LOCAL_ROLLOUT_SERVER:-$GRPO_OFFLINE}"
export AGENTLIGHTNING_LOCAL_ROLLOUT_HOST="${AGENTLIGHTNING_LOCAL_ROLLOUT_HOST:-127.0.0.1}"
export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
export AGENTLIGHTNING_RAY_NODE_IP_ADDRESS="${AGENTLIGHTNING_RAY_NODE_IP_ADDRESS:-$RAY_NODE_IP_ADDRESS}"
LOCAL_NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="$LOCAL_NO_PROXY,$NO_PROXY"
else
  export NO_PROXY="$LOCAL_NO_PROXY"
fi
export no_proxy="${no_proxy:-$NO_PROXY}"

CUDA12_LIBS="$(find "$GRPO_ENV_DIR/lib/python3.11/site-packages/nvidia" \
  -path '*/lib' -type d ! -path '*/cu13/*' ! -path '*/cu13' 2>/dev/null | paste -sd:)"
if [[ -n "$CUDA12_LIBS" ]]; then
  export LD_LIBRARY_PATH="$CUDA12_LIBS:$GRPO_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$GRPO_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$RAY_TMPDIR"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "This script serially trains: fog, snow, rain, low_light." >&2
  exit 2
fi

if [[ "${GRPO_SKIP_RAY_STOP:-0}" != "1" ]]; then
  conda run --no-capture-output -n "$GRPO_CONDA_ENV" ray stop --force >/dev/null 2>&1 || true
fi

for expert in fog snow rain low_light; do
  echo "Starting ${expert} expert GRPO with the two-GPU topology."
  if ! "$ROOT/examples/image_restoration_multi_agent/grpo/run_expert_grpo.sh" "$expert"; then
    echo "${expert} expert GRPO failed; remaining experts will not be started." >&2
    exit 1
  fi
done

echo "All four expert GRPO training runs completed successfully."
