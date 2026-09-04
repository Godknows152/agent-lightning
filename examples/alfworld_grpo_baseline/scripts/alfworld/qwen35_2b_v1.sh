#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec env ALFWORLD_MODEL_PROFILE=qwen35_2b bash "${ROOT}/scripts/run_alfworld_grpo_2gpu.sh" "$@"
