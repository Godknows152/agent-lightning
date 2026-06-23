#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/LXJ/Python_Projects/Agent_Lightning"
GRPO_CONDA_ENV="${GRPO_CONDA_ENV:-qwen35-agentlightning-rl}"
GRPO_ENV_DIR="${GRPO_ENV_DIR:-/home/LXJ/anaconda3/envs/${GRPO_CONDA_ENV}}"
GRPO_PYTHON="${GRPO_PYTHON:-${GRPO_ENV_DIR}/bin/python}"

if [[ ! -d "$GRPO_ENV_DIR" ]]; then
  echo "GRPO conda environment directory does not exist: $GRPO_ENV_DIR" >&2
  exit 1
fi

if [[ ! -x "$GRPO_PYTHON" ]]; then
  echo "GRPO Python is not executable: $GRPO_PYTHON" >&2
  exit 1
fi

export GRPO_CONDA_ENV
export GRPO_ENV_DIR
export GRPO_PYTHON
export GRPO_TOPOLOGY="4gpu"
export GRPO_CONFIG_DIR="examples/image_restoration_multi_agent/grpo/configs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export IMAGE_RESTORATION_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
export IMAGE_RESTORATION_IQA_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
export IMAGE_RESTORATION_WORKERS_PER_DEVICE="1"
export IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE="2"
export GRPO_OFFLINE="${GRPO_OFFLINE:-1}"
export AGENTLIGHTNING_FORCE_LOCAL_ROLLOUT_SERVER="${AGENTLIGHTNING_FORCE_LOCAL_ROLLOUT_SERVER:-$GRPO_OFFLINE}"
export AGENTLIGHTNING_LOCAL_ROLLOUT_HOST="${AGENTLIGHTNING_LOCAL_ROLLOUT_HOST:-127.0.0.1}"
export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
export AGENTLIGHTNING_RAY_NODE_IP_ADDRESS="${AGENTLIGHTNING_RAY_NODE_IP_ADDRESS:-$RAY_NODE_IP_ADDRESS}"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "This script serially trains: fog, snow, rain, low_light." >&2
  exit 2
fi

if [[ "${GRPO_SKIP_RAY_STOP:-0}" != "1" ]]; then
  conda run --no-capture-output -n "$GRPO_CONDA_ENV" ray stop --force >/dev/null 2>&1 || true
fi

for expert in fog snow rain low_light; do
  echo "Starting ${expert} expert GRPO with the four-GPU topology."
  if ! "$ROOT/examples/image_restoration_multi_agent/grpo/run_expert_grpo.sh" "$expert"; then
    echo "${expert} expert GRPO failed; remaining experts will not be started." >&2
    exit 1
  fi
done

echo "All four expert GRPO training runs completed successfully."
