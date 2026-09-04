#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/alfworld-verl/bin/python}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
"${PYTHON_BIN}" "${ROOT}/scripts/preflight_alfworld.py"
