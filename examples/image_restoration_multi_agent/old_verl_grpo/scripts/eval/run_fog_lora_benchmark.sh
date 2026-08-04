#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="/home/LXJ/anaconda3/envs/verl/bin/python"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_fog_loras.py"
LOG_DIR="$PROJECT_ROOT/examples/image_restoration_multi_agent/old_verl_grpo/log/eval"
PID_FILE="$LOG_DIR/fog_lora_benchmark.pid"
UNIT_FILE="$LOG_DIR/fog_lora_benchmark.unit"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Benchmark is already running: PID $existing_pid" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi
if [[ -f "$UNIT_FILE" ]]; then
    existing_unit="$(<"$UNIT_FILE")"
    if systemctl --user is-active --quiet "$existing_unit" 2>/dev/null; then
        echo "Benchmark is already running: unit $existing_unit" >&2
        exit 1
    fi
    rm -f "$UNIT_FILE"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/fog_lora_benchmark_${timestamp}.log"

cd "$PROJECT_ROOT"
unit_name="fog-lora-benchmark-${timestamp}.service"
systemd-run --user \
    --unit="$unit_name" \
    --collect \
    --working-directory="$PROJECT_ROOT" \
    --property="StandardInput=null" \
    --property="StandardOutput=append:$log_file" \
    --property="StandardError=append:$log_file" \
    --property="KillMode=control-group" \
    --setenv="https_proxy=http://127.0.0.1:7897" \
    --setenv="http_proxy=http://127.0.0.1:7897" \
    --setenv="all_proxy=socks5://127.0.0.1:7897" \
    --setenv="PYTHONUNBUFFERED=1" \
    "$PYTHON_BIN" -u "$BENCHMARK_SCRIPT" command=run resume=true "$@" >/dev/null

sleep 0.5
benchmark_pid="$(systemctl --user show "$unit_name" --property=MainPID --value)"
if [[ ! "$benchmark_pid" =~ ^[1-9][0-9]*$ ]]; then
    if grep -q '^Benchmark already complete:' "$log_file" 2>/dev/null; then
        echo "Fog LoRA benchmark outputs are already complete."
        echo "Log: $log_file"
        exit 0
    fi
    echo "Benchmark service failed to start: $unit_name; inspect $log_file" >&2
    exit 1
fi
printf '%s\n' "$benchmark_pid" > "$PID_FILE"
printf '%s\n' "$unit_name" > "$UNIT_FILE"

echo "Started Fog LoRA benchmark in background."
echo "PID: $benchmark_pid"
echo "Unit: $unit_name"
echo "Log: $log_file"
