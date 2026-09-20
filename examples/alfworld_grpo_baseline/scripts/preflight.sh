#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}"
ALFWORLD_VERL_ROOT="${ALFWORLD_VERL_ROOT:-/home/LXJ/Python_Projects/verl}"
export PYTHONPATH="${ROOT}/src:${ALFWORLD_VERL_ROOT}:${ROOT}/../image_restoration_multi_agent/old_verl_grpo/.pydeps:${PYTHONPATH:-}"
"${PYTHON_BIN}" "${ROOT}/scripts/preflight.py"
