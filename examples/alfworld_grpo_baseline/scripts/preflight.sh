#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}"
export PYTHONPATH="${ROOT}/src:${ROOT}/../image_restoration_multi_agent/old_verl_grpo/.pydeps:/home/LXJ/Python_Projects/Agent_Lightning/examples/image_restoration_multi_agent/verl_backend:${PYTHONPATH:-}"
"${PYTHON_BIN}" "${ROOT}/scripts/preflight.py"
