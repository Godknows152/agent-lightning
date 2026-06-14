#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {fog|snow|rain|low_light} [--smoke]" >&2
  exit 2
fi

EXPERT="$1"
shift
case "$EXPERT" in
  fog|snow|rain|low_light) ;;
  *) echo "Unknown expert: $EXPERT" >&2; exit 2 ;;
esac

ROOT="/home/LXJ/Python_Projects/Agent_Lightning"
LOG_DIR="$ROOT/examples/image_restoration_multi_agent/grpo/log"
mkdir -p "$LOG_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${EXPERT}_grpo_${RUN_STAMP}.log"
TOOL_LOG_FILE="$LOG_DIR/${EXPERT}_tool_runtime_${RUN_STAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/LXJ/tmp/ray_agent_lightning_grpo}"
mkdir -p "$RAY_TMPDIR"

cd "$ROOT"
SMOKE=false
for argument in "$@"; do
  if [[ "$argument" == "--smoke" ]]; then
    SMOKE=true
  fi
done

TOOL_SERVER_PID=""
cleanup_tool_server() {
  if [[ -n "$TOOL_SERVER_PID" ]] && kill -0 "$TOOL_SERVER_PID" 2>/dev/null; then
    echo "Stopping persistent restoration/IQA service (PID $TOOL_SERVER_PID)."
    kill "$TOOL_SERVER_PID" 2>/dev/null || true
    wait "$TOOL_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_tool_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$SMOKE" == false ]]; then
  TOOL_PYTHON="${IMAGE_RESTORATION_PYTHON:-/home/LXJ/anaconda3/envs/verl/bin/python}"
  TOOL_PORT="${IMAGE_RESTORATION_TOOL_PORT:-8767}"
  RESTORATION_DEVICES="${IMAGE_RESTORATION_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
  IQA_DEVICES="${IMAGE_RESTORATION_IQA_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
  # One complete 16-model restoration set and two IQA replicas per GPU.
  RESTORATION_WORKERS_PER_DEVICE="${IMAGE_RESTORATION_WORKERS_PER_DEVICE:-1}"
  IQA_WORKERS_PER_DEVICE="${IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE:-2}"
  RESTORATION_DEVICE_COUNT=$(awk -F, '{print NF}' <<<"$RESTORATION_DEVICES")
  IQA_DEVICE_COUNT=$(awk -F, '{print NF}' <<<"$IQA_DEVICES")
  RESTORATION_WORKERS=$((RESTORATION_WORKERS_PER_DEVICE * RESTORATION_DEVICE_COUNT))
  IQA_WORKERS=$((IQA_WORKERS_PER_DEVICE * IQA_DEVICE_COUNT))
  TOOL_URL="http://127.0.0.1:${TOOL_PORT}"
  export IMAGE_RESTORATION_SERVICE_URL="$TOOL_URL"
  if [[ ! -x "$TOOL_PYTHON" ]]; then
    echo "Persistent tool Python is not executable: $TOOL_PYTHON" >&2
    exit 1
  fi

  # Clean up stale processes from previous runs
  STALE_PID=$(lsof -ti :$TOOL_PORT 2>/dev/null || true)
  if [[ -n "$STALE_PID" ]]; then
    echo "Killing stale process $STALE_PID on port $TOOL_PORT" >&2
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 1
  fi

  echo "Loading ${IQA_WORKERS_PER_DEVICE} IQA worker(s) per GPU across ${IQA_DEVICES} (${IQA_WORKERS} total)."
  echo "Loading one complete restoration tool set per GPU across ${RESTORATION_DEVICES} (${RESTORATION_WORKERS_PER_DEVICE} worker(s) per model per GPU)."
  echo "Persistent tool log: $TOOL_LOG_FILE"
  "$TOOL_PYTHON" \
    examples/image_restoration_multi_agent/tool_runtime/persistent_tool_server.py \
    --host 127.0.0.1 \
    --port "$TOOL_PORT" \
    --tools-config examples/image_restoration_multi_agent/config/tools.yaml \
    --external-tools-root External_Tools \
    --iqa-repo External_Tools/iqa_repos/IQA-PyTorch \
    --metrics maniqa,niqe,clipiqa,topiq_nr \
    --iqa-devices "$IQA_DEVICES" \
    --restoration-devices "$RESTORATION_DEVICES" \
    --iqa-workers "$IQA_WORKERS" \
    --restoration-workers "$RESTORATION_WORKERS" \
    >"$TOOL_LOG_FILE" 2>&1 &
  TOOL_SERVER_PID=$!

  TOOL_STARTUP_TIMEOUT="${IMAGE_RESTORATION_TOOL_STARTUP_TIMEOUT:-1800}"
  TOOL_WAITED=0
  TOOL_HEALTH_CHECK='import json, sys, urllib.request; '
  TOOL_HEALTH_CHECK+='data=json.load(urllib.request.urlopen(sys.argv[1], timeout=2)); '
  TOOL_HEALTH_CHECK+='assert data["status"] == "ready"; '
  TOOL_HEALTH_CHECK+='assert data["iqa_workers"] == int(sys.argv[2]); '
  TOOL_HEALTH_CHECK+='assert data["restoration_workers"] == int(sys.argv[3]); '
  TOOL_HEALTH_CHECK+='assert data["iqa_devices"] == sys.argv[4].split(","); '
  TOOL_HEALTH_CHECK+='assert data["restoration_devices"] == sys.argv[5].split(","); '
  TOOL_HEALTH_CHECK+='assert all(data["iqa_worker_devices"].count(device) == int(sys.argv[6]) for device in data["iqa_devices"]); '
  TOOL_HEALTH_CHECK+='assert all(all(worker_devices.count(device) == int(sys.argv[7]) for device in data["restoration_devices"]) for worker_devices in data["restoration_worker_devices"].values())'
  until "$TOOL_PYTHON" -c "$TOOL_HEALTH_CHECK" \
    "$TOOL_URL/health" "$IQA_WORKERS" "$RESTORATION_WORKERS" \
    "$IQA_DEVICES" "$RESTORATION_DEVICES" "$IQA_WORKERS_PER_DEVICE" \
    "$RESTORATION_WORKERS_PER_DEVICE" >/dev/null 2>&1; do
    if ! kill -0 "$TOOL_SERVER_PID" 2>/dev/null; then
      echo "Persistent tool service exited during startup. See $TOOL_LOG_FILE" >&2
      exit 1
    fi
    if (( TOOL_WAITED >= TOOL_STARTUP_TIMEOUT )); then
      echo "Persistent tool service did not become ready within ${TOOL_STARTUP_TIMEOUT}s." >&2
      exit 1
    fi
    if (( TOOL_WAITED % 30 == 0 )); then
      echo "Waiting for persistent models to load (${TOOL_WAITED}s elapsed)..."
    fi
    sleep 2
    TOOL_WAITED=$((TOOL_WAITED + 2))
  done
  echo "Persistent restoration/IQA service is ready at $TOOL_URL."
fi

conda run --no-capture-output -n agent-lightning \
  bash -c '
    set -euo pipefail
    for proxy_var in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
      proxy_value="${!proxy_var-}"
      if [[ "$proxy_value" == socks://* ]]; then
        printf -v "$proxy_var" "%s" "socks5://${proxy_value#socks://}"
        export "$proxy_var"
      fi
    done
    exec python "$@"
  ' bash \
  examples/image_restoration_multi_agent/grpo/train_grpo.py \
  --config "examples/image_restoration_multi_agent/grpo/configs/${EXPERT}.yaml" \
  "$@" 2>&1 | /home/LXJ/anaconda3/envs/agent-lightning/bin/python \
    examples/image_restoration_multi_agent/grpo/render_training_log.py \
    --log-file "$LOG_FILE"
