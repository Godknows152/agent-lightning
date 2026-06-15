#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/LXJ/Python_Projects/Agent_Lightning"

export GRPO_TOPOLOGY="4gpu"
export GRPO_CONFIG_DIR="examples/image_restoration_multi_agent/grpo/configs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export IMAGE_RESTORATION_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
export IMAGE_RESTORATION_IQA_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
export IMAGE_RESTORATION_WORKERS_PER_DEVICE="1"
export IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE="2"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  echo "This script serially trains: fog, snow, rain, low_light." >&2
  exit 2
fi

for expert in fog snow rain low_light; do
  echo "Starting ${expert} expert GRPO with the four-GPU topology."
  if ! "$ROOT/examples/image_restoration_multi_agent/grpo/run_expert_grpo.sh" "$expert"; then
    echo "${expert} expert GRPO failed; remaining experts will not be started." >&2
    exit 1
  fi
done

echo "All four expert GRPO training runs completed successfully."
