#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_expert_old_verl_grpo_4gpu.sh <fog|low_light|rain|snow> [--smoke|--preflight] [hydra overrides...]

Training and smoke runs start in the background. Preflight runs in the current
shell without creating datasets, starting Ray, or allocating a model on GPU.

Default fog run from its completed SFT LoRA:
  bash examples/image_restoration_multi_agent/old_verl_grpo/run_expert_old_verl_grpo_4gpu.sh fog

Environment overrides:
  OLD_VERL_SMOKE=1
  OLD_VERL_MODEL_PATH=/path/to/base-model
  OLD_VERL_ADAPTER_PATH=/path/to/sft-lora-adapter
  OLD_VERL_SFT_ADAPTER_ROOT=/path/to/four-expert-adapter-root
  OLD_VERL_PREFLIGHT_ONLY=1
  OLD_VERL_TOOL_REGISTRY_PATH=/path/to/tools.yaml
  OLD_VERL_CONFIG_NAME=fog_config_4gpu  # defaults to <expert>_config_4gpu
  OLD_VERL_CUDA_VISIBLE_DEVICES=0,1,2,3
  OLD_VERL_CLEAR_INTERMEDIATE_IMAGES=1
  OLD_VERL_INTERMEDIATE_DIR=/home/LXJ/tmp/agent_lightning_old_verl_restoration_4gpu
  OLD_VERL_CLEAR_PENALIZED_SAMPLES=1  # clear the resolved output's penalized_samples before training
  OLD_VERL_CLEAR_TOOL_LOGS=1  # clear this expert's tool logs before training
  OLD_VERL_SHOW_KNOWN_WARNINGS=1  # show otherwise-suppressed third-party warning spam
  OLD_VERL_RESUME_MODE=auto|disable|resume_path
  OLD_VERL_RESUME_FROM_PATH=/path/to/global_step_N  # adds the "_续" suffix
  OLD_VERL_EXPERIMENT_NAME=custom-name  # optional; standard name is <expert>_4gpu_MMDD[_续]
  OLD_VERL_LOG_DIR=/path/to/log-dir  # optional main/tool log directory override
  OLD_VERL_OUTPUT_DIR=/path/to/output-dir  # optional checkpoint output override; YAML wins when unset
  PYTHON_BIN=/home/LXJ/anaconda3/envs/verl/bin/python
  RAY_BIN=/home/LXJ/anaconda3/envs/verl/bin/ray
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXPERT="${1:-fog}"
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG_PATH_OVERRIDE=""
CONFIG_NAME_OVERRIDE=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-path=*)
      CONFIG_PATH_OVERRIDE="${1#*=}"
      ;;
    --config-path)
      CONFIG_PATH_OVERRIDE="${2:?--config-path requires a value}"
      shift
      ;;
    --config-name=*)
      CONFIG_NAME_OVERRIDE="${1#*=}"
      ;;
    --config-name)
      CONFIG_NAME_OVERRIDE="${2:?--config-name requires a value}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      ;;
  esac
  shift
done
set -- "${FORWARD_ARGS[@]}"

SMOKE="${OLD_VERL_SMOKE:-0}"
PREFLIGHT_ONLY="${OLD_VERL_PREFLIGHT_ONLY:-0}"
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
elif [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
  shift
fi

case "${EXPERT}" in
  fog|low_light|rain|snow) ;;
  *)
    echo "Unsupported expert: ${EXPERT}" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLE_DIR="${ROOT}/examples/image_restoration_multi_agent"
BACKEND_ROOT="${EXAMPLE_DIR}/verl_backend"
OLD_VERL_DIR="${EXAMPLE_DIR}/old_verl_grpo"
LOG_ROOT="${OLD_VERL_DIR}/log"
LOG_DIR="${OLD_VERL_LOG_DIR:-${LOG_ROOT}/${EXPERT}/4gpu}"
CONFIG_DIR="${CONFIG_PATH_OVERRIDE:-${OLD_VERL_DIR}/config}"
CONFIG_NAME="${CONFIG_NAME_OVERRIDE:-${OLD_VERL_CONFIG_NAME:-${EXPERT}_config_4gpu}}"
CONVERTER="${OLD_VERL_DIR}/scripts/convert_current_jsonl_to_verl_parquet.py"
RUN_NAMING_RESOLVER="${OLD_VERL_DIR}/scripts/resolve_training_run_name.py"
LOCAL_PYDEPS="${OLD_VERL_LOCAL_PYDEPS:-${OLD_VERL_DIR}/.pydeps}"
TOOL_REGISTRY_PATH="${OLD_VERL_TOOL_REGISTRY_PATH:-${EXAMPLE_DIR}/config/tools.yaml}"
INTERMEDIATE_DIR="${OLD_VERL_INTERMEDIATE_DIR:-/home/LXJ/tmp/agent_lightning_old_verl_restoration_4gpu}"
DEFAULT_MODEL_PATH="/home/LXJ/Python_Projects/Models/Qwen3.5-9B"
DEFAULT_SFT_ADAPTER_ROOT="${ROOT}/LlamaFactory/image_restoration_experts/outputs/qwen3_5_0721/format_cold_start"

PYTHON_BIN="${PYTHON_BIN:-/home/LXJ/anaconda3/envs/verl/bin/python}"
RAY_BIN="${RAY_BIN:-/home/LXJ/anaconda3/envs/verl/bin/ray}"
MODEL_PATH_OVERRIDE="${OLD_VERL_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
SFT_ADAPTER_ROOT="${OLD_VERL_SFT_ADAPTER_ROOT:-${DEFAULT_SFT_ADAPTER_ROOT}}"
ADAPTER_PATH_OVERRIDE="${OLD_VERL_ADAPTER_PATH:-${SFT_ADAPTER_ROOT}/${EXPERT}}"

mkdir -p "${LOG_DIR}"
if [[ "${OLD_VERL_RUN_IN_FOREGROUND:-0}" != "1" && "${PREFLIGHT_ONLY}" != "1" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MAIN_LOG="${LOG_DIR}/${EXPERT}_4gpu_${TIMESTAMP}.log"
  CHILD_ARGS=("${EXPERT}")
  if [[ "${SMOKE}" == "1" ]]; then
    CHILD_ARGS+=(--smoke)
  fi
  CHILD_ARGS+=("$@")
  nohup env \
    OLD_VERL_RUN_IN_FOREGROUND=1 \
    OLD_VERL_MAIN_LOG="${MAIN_LOG}" \
    bash "${SCRIPT_PATH}" "${CHILD_ARGS[@]}" \
    >"${MAIN_LOG}" 2>&1 </dev/null &
  BACKGROUND_PID=$!
  echo "Started ${EXPERT} GRPO in background (PID ${BACKGROUND_PID})."
  echo "Log file: ${MAIN_LOG}"
  exit 0
fi

TRAIN_PARQUET="${OLD_VERL_DIR}/data/4gpu/${EXPERT}_train.parquet"
VAL_PARQUET="${OLD_VERL_DIR}/data/4gpu/${EXPERT}_val.parquet"
RUN_KIND="full"
TRAIN_LIMIT_ARGS=()
VAL_LIMIT_ARGS=()
CONFIG_OVERRIDES=()
HYDRA_OVERRIDES=()
PROJECT_NAME_OVERRIDE="${OLD_VERL_PROJECT_NAME:-}"
EXPERIMENT_NAME_OVERRIDE="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_4gpu_$(date +%m%d)}"
OUTPUT_DIR_OVERRIDE="${OLD_VERL_OUTPUT_DIR:-}"
SWANLAB_LOG_DIR_OVERRIDE="${OLD_VERL_SWANLAB_LOG_DIR:-}"
SWANLAB_MODE_OVERRIDE=""
RESUME_MODE_OVERRIDE="${OLD_VERL_RESUME_MODE:-}"
RESUME_FROM_PATH_OVERRIDE="${OLD_VERL_RESUME_FROM_PATH:-}"
OUTPUT_DIR_WAS_EXPLICIT=0
RESUME_MODE_WAS_EXPLICIT=0
RESUME_PATH_WAS_EXPLICIT=0

if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  OUTPUT_DIR_WAS_EXPLICIT=1
fi
if [[ -n "${RESUME_MODE_OVERRIDE}" ]]; then
  RESUME_MODE_WAS_EXPLICIT=1
fi
if [[ -n "${RESUME_FROM_PATH_OVERRIDE}" ]]; then
  RESUME_PATH_WAS_EXPLICIT=1
fi

# Treat equivalent Hydra CLI settings as naming inputs so the derived output
# directory and SwanLab log directory cannot drift from the effective config.
for override in "$@"; do
  case "${override}" in
    trainer.experiment_name=*)
      EXPERIMENT_NAME_OVERRIDE="${override#trainer.experiment_name=}"
      ;;
    trainer.default_local_dir=*)
      OUTPUT_DIR_OVERRIDE="${override#trainer.default_local_dir=}"
      OUTPUT_DIR_WAS_EXPLICIT=1
      ;;
    trainer.resume_mode=*)
      RESUME_MODE_OVERRIDE="${override#trainer.resume_mode=}"
      RESUME_MODE_WAS_EXPLICIT=1
      ;;
    trainer.resume_from_path=*)
      RESUME_FROM_PATH_OVERRIDE="${override#trainer.resume_from_path=}"
      RESUME_PATH_WAS_EXPLICIT=1
      if [[ "${RESUME_FROM_PATH_OVERRIDE}" == "null" ]]; then
        RESUME_FROM_PATH_OVERRIDE=""
      fi
      ;;
  esac
done

if [[ "${SMOKE}" == "1" ]]; then
  RUN_KIND="smoke"
  RESUME_MODE_OVERRIDE="disable"
  RESUME_FROM_PATH_OVERRIDE=""
  RESUME_MODE_WAS_EXPLICIT=1
  RESUME_PATH_WAS_EXPLICIT=1
  TRAIN_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_TRAIN_LIMIT:-2}")
  VAL_LIMIT_ARGS=(--limit "${OLD_VERL_SMOKE_VAL_LIMIT:-2}")
  SWANLAB_MODE_OVERRIDE="offline"
  HYDRA_OVERRIDES+=(
    "data.train_batch_size=2"
    "data.val_max_samples=1"
    "actor_rollout_ref.rollout.n=2"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
    "actor_rollout_ref.rollout.max_num_seqs=2"
    "actor_rollout_ref.rollout.max_model_len=4096"
    "actor_rollout_ref.rollout.enforce_eager=true"
    "actor_rollout_ref.actor.ppo_mini_batch_size=2"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=1"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "trainer.val_before_train=false"
  )
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${RAY_BIN}" ]]; then
  echo "Ray executable not found or not executable: ${RAY_BIN}" >&2
  exit 1
fi

IFS=$'\t' read -r \
  CONFIG_RESUME_MODE \
  CONFIG_RESUME_FROM_PATH \
  CONFIG_OUTPUT_DIR \
  CONFIG_SWANLAB_LOG_DIR \
  < <(
    "${PYTHON_BIN}" - "${CONFIG_DIR}" "${CONFIG_NAME}" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

with initialize_config_dir(version_base=None, config_dir=str(Path(sys.argv[1]).resolve())):
    config = compose(config_name=sys.argv[2])

resume_from_path = config.trainer.resume_from_path
fields = (
    str(config.trainer.resume_mode),
    str(resume_from_path) if resume_from_path is not None else "-",
    str(config.trainer.default_local_dir),
    str(config.trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR),
)
print("\t".join(fields))
PY
  )
if [[ "${RESUME_MODE_WAS_EXPLICIT}" != "1" ]]; then
  RESUME_MODE_OVERRIDE="${CONFIG_RESUME_MODE}"
fi
if [[ "${RESUME_PATH_WAS_EXPLICIT}" != "1" && "${CONFIG_RESUME_FROM_PATH}" != "-" ]]; then
  RESUME_FROM_PATH_OVERRIDE="${CONFIG_RESUME_FROM_PATH}"
fi
if [[ -z "${OUTPUT_DIR_OVERRIDE}" ]]; then
  OUTPUT_DIR_OVERRIDE="${CONFIG_OUTPUT_DIR}"
fi
if [[ -z "${SWANLAB_LOG_DIR_OVERRIDE}" && "${OUTPUT_DIR_WAS_EXPLICIT}" != "1" ]]; then
  SWANLAB_LOG_DIR_OVERRIDE="${CONFIG_SWANLAB_LOG_DIR}"
fi

RUN_NAMING_ARGS=(
  --expert "${EXPERT}"
  --output-root "${OLD_VERL_DIR}/outputs/4gpu"
  --resume-mode "${RESUME_MODE_OVERRIDE}"
  --output-dir "${OUTPUT_DIR_OVERRIDE}"
)
if [[ -n "${RESUME_FROM_PATH_OVERRIDE}" ]]; then
  RUN_NAMING_ARGS+=(--resume-from-path "${RESUME_FROM_PATH_OVERRIDE}")
fi
if [[ -n "${EXPERIMENT_NAME_OVERRIDE}" ]]; then
  RUN_NAMING_ARGS+=(--experiment-name "${EXPERIMENT_NAME_OVERRIDE}")
fi
IFS=$'\t' read -r \
  EXPERIMENT_NAME_OVERRIDE \
  OUTPUT_DIR_OVERRIDE \
  RESOLVED_SWANLAB_LOG_DIR \
  RESUME_FROM_PATH_OVERRIDE \
  < <(cd "${ROOT}" && "${PYTHON_BIN}" "${RUN_NAMING_RESOLVER}" "${RUN_NAMING_ARGS[@]}")

if [[ "${RESUME_FROM_PATH_OVERRIDE}" == "-" ]]; then
  RESUME_FROM_PATH_OVERRIDE=""
fi
if [[ "${SMOKE}" == "1" ]]; then
  EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE}_smoke"
fi
if [[ -z "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  SWANLAB_LOG_DIR_OVERRIDE="${RESOLVED_SWANLAB_LOG_DIR}"
fi

if [[ ! -d "${MODEL_PATH_OVERRIDE}" ]]; then
  echo "Model path not found: ${MODEL_PATH_OVERRIDE}" >&2
  exit 1
fi
if [[ ! -d "${ADAPTER_PATH_OVERRIDE}" ]]; then
  echo "SFT adapter path not found: ${ADAPTER_PATH_OVERRIDE}" >&2
  exit 1
fi
if [[ ! -s "${ADAPTER_PATH_OVERRIDE}/adapter_config.json" ]]; then
  echo "SFT adapter config is missing or empty: ${ADAPTER_PATH_OVERRIDE}/adapter_config.json" >&2
  exit 1
fi
if [[ ! -s "${ADAPTER_PATH_OVERRIDE}/adapter_model.safetensors" ]]; then
  echo "SFT adapter weights are missing or empty: ${ADAPTER_PATH_OVERRIDE}/adapter_model.safetensors" >&2
  exit 1
fi

CONFIG_OVERRIDES+=(
  "actor_rollout_ref.model.path='${MODEL_PATH_OVERRIDE}'"
  "actor_rollout_ref.model.lora_adapter_path='${ADAPTER_PATH_OVERRIDE}'"
  "trainer.experiment_name='${EXPERIMENT_NAME_OVERRIDE}'"
  "trainer.default_local_dir='${OUTPUT_DIR_OVERRIDE}'"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR='${SWANLAB_LOG_DIR_OVERRIDE}'"
  "trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR='${LOG_DIR}'"
)
if [[ -n "${PROJECT_NAME_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.project_name='${PROJECT_NAME_OVERRIDE}'")
fi
if [[ -n "${RESUME_FROM_PATH_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=(
    "trainer.resume_mode=resume_path"
    "trainer.resume_from_path='${RESUME_FROM_PATH_OVERRIDE}'"
  )
else
  CONFIG_OVERRIDES+=(
    "trainer.resume_mode=${RESUME_MODE_OVERRIDE}"
    "trainer.resume_from_path=null"
  )
fi
if [[ -n "${OLD_VERL_SWANLAB_MODE:-}" ]]; then
  SWANLAB_MODE_OVERRIDE="${OLD_VERL_SWANLAB_MODE}"
elif [[ -n "${GRPO_SWANLAB_MODE:-}" ]]; then
  SWANLAB_MODE_OVERRIDE="${GRPO_SWANLAB_MODE}"
fi
if [[ -n "${SWANLAB_MODE_OVERRIDE}" ]]; then
  CONFIG_OVERRIDES+=("trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE=${SWANLAB_MODE_OVERRIDE}")
fi

# Expose four physical GPUs to Actor/FSDP, two TP2 rollout replicas, restoration, and IQA.
# The tool runtime creates one restoration/IQA worker on each logical cuda:0-3.
export CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -d "${LOCAL_PYDEPS}" ]]; then
  export PYTHONPATH="${LOCAL_PYDEPS}:${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${BACKEND_ROOT}:${EXAMPLE_DIR}:${PYTHONPATH:-}"
fi
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Hide harmless import-time warnings from optional CUDA/NPU dependencies. The
# high-volume LoRA warning is filtered locally around adapter loading because
# its message starts with a dynamic parameter name. Keep all other warnings visible.
if [[ "${OLD_VERL_SHOW_KNOWN_WARNINGS:-0}" != "1" ]]; then
  KNOWN_WARNING_FILTERS="ignore:The pynvml package is deprecated:FutureWarning,ignore:NPU not support router replay for now:UserWarning"
  export PYTHONWARNINGS="${KNOWN_WARNING_FILTERS}${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"
fi
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
if [[ -n "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR_OVERRIDE}"
fi
if [[ -n "${SWANLAB_MODE_OVERRIDE}" ]]; then
  export SWANLAB_MODE="${SWANLAB_MODE_OVERRIDE}"
fi
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
export VERL_LOG_DIR="${LOG_DIR}"
export LD_LIBRARY_PATH="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"

"${PYTHON_BIN}" - "${MODEL_PATH_OVERRIDE}" "${ADAPTER_PATH_OVERRIDE}" "${EXPERT}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1]).expanduser().resolve()
adapter_path = Path(sys.argv[2]).expanduser().resolve()
expert = sys.argv[3]
config_path = adapter_path / "adapter_config.json"

with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

expected_targets = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
}
actual_targets = set(config.get("target_modules", []))
adapter_base = Path(config.get("base_model_name_or_path", "")).expanduser().resolve()
errors = []
if config.get("peft_type") != "LORA":
    errors.append(f"peft_type={config.get('peft_type')!r}, expected 'LORA'")
if config.get("task_type") != "CAUSAL_LM":
    errors.append(f"task_type={config.get('task_type')!r}, expected 'CAUSAL_LM'")
if config.get("r") != 16:
    errors.append(f"r={config.get('r')!r}, expected 16")
if config.get("lora_alpha") != 32:
    errors.append(f"lora_alpha={config.get('lora_alpha')!r}, expected 32")
if actual_targets != expected_targets:
    errors.append(
        "target_modules differ: "
        f"missing={sorted(expected_targets - actual_targets)}, "
        f"unexpected={sorted(actual_targets - expected_targets)}"
    )
if adapter_base != model_path:
    errors.append(f"base model mismatch: adapter={adapter_base}, configured={model_path}")
if adapter_path.name != expert:
    errors.append(f"adapter directory {adapter_path.name!r} does not match expert {expert!r}")

if errors:
    raise SystemExit("Invalid SFT adapter:\n- " + "\n- ".join(errors))

print(
    f"Validated {expert} SFT LoRA: {adapter_path} "
    f"(r={config['r']}, alpha={config['lora_alpha']}, targets={len(actual_targets)})"
)
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  "${PYTHON_BIN}" - \
    "${CONFIG_DIR}" \
    "${CONFIG_NAME}" \
    "${MODEL_PATH_OVERRIDE}" \
    "${ADAPTER_PATH_OVERRIDE}" \
    "${EXPERIMENT_NAME_OVERRIDE}" \
    "${OUTPUT_DIR_OVERRIDE}" \
    "${SWANLAB_LOG_DIR_OVERRIDE}" \
    "${LOG_DIR}" \
    "${RESUME_FROM_PATH_OVERRIDE}" \
    "${ROOT}" \
    "${CONFIG_OVERRIDES[@]}" \
    "${HYDRA_OVERRIDES[@]}" \
    "$@" <<'PY'
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

config_dir = str(Path(sys.argv[1]).resolve())
config_name = sys.argv[2]
expected_model = str(Path(sys.argv[3]).resolve())
expected_adapter = str(Path(sys.argv[4]).resolve())
expected_experiment_name = sys.argv[5]
expected_output_dir = str(Path(sys.argv[6]).resolve())
expected_swanlab_log_dir = str(Path(sys.argv[7]).resolve())
expected_verl_log_dir = str(Path(sys.argv[8]).resolve())
expected_resume_path = str(Path(sys.argv[9]).resolve()) if sys.argv[9] else None
project_root = Path(sys.argv[10]).resolve()
overrides = sys.argv[11:]

with initialize_config_dir(version_base=None, config_dir=config_dir):
    config = compose(config_name=config_name, overrides=overrides)

actual_model = str(Path(config.actor_rollout_ref.model.path).resolve())
actual_adapter = str(Path(config.actor_rollout_ref.model.lora_adapter_path).resolve())
errors = []
visible_devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
if len(visible_devices) != 4 or len(set(visible_devices)) != 4:
    errors.append(f"CUDA_VISIBLE_DEVICES must contain four unique GPUs, got {visible_devices!r}")
if config.trainer.n_gpus_per_node != 4:
    errors.append(f"trainer.n_gpus_per_node={config.trainer.n_gpus_per_node!r}, expected 4")
if config.actor_rollout_ref.rollout.tensor_model_parallel_size != 2:
    errors.append(
        "actor_rollout_ref.rollout.tensor_model_parallel_size="
        f"{config.actor_rollout_ref.rollout.tensor_model_parallel_size!r}, expected 2 for two TP2 replicas"
    )
tool_config_path = Path(config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
if not tool_config_path.is_absolute():
    tool_config_path = project_root / tool_config_path
if not tool_config_path.name.endswith("_4gpu.yaml"):
    errors.append(f"four-GPU tool config is not selected: {tool_config_path}")
elif not tool_config_path.is_file():
    errors.append(f"four-GPU tool config does not exist: {tool_config_path}")
else:
    import yaml

    with tool_config_path.open(encoding="utf-8") as file:
        tool_config = yaml.safe_load(file)
    runtime_config = tool_config["tools"][0]["config"]
    expected_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    for field in ("worker_devices", "model_devices", "iqa_devices"):
        if runtime_config.get(field) != expected_devices:
            errors.append(f"tool {field}={runtime_config.get(field)!r}, expected {expected_devices!r}")
    expected_keep_models_loaded = tool_config_path.parent.name != "v2"
    if runtime_config.get("keep_models_loaded_between_sampling_steps") is not expected_keep_models_loaded:
        errors.append(
            "tool keep_models_loaded_between_sampling_steps="
            f"{runtime_config.get('keep_models_loaded_between_sampling_steps')!r}, "
            f"expected {expected_keep_models_loaded!r} for {tool_config_path.parent.name}"
        )
    if runtime_config.get("device") != "cuda:0":
        errors.append(f"tool device={runtime_config.get('device')!r}, expected 'cuda:0'")
    if runtime_config.get("iqa_device") != "cuda:0":
        errors.append(f"tool iqa_device={runtime_config.get('iqa_device')!r}, expected 'cuda:0'")
    expected_tool_output = Path("/home/LXJ/tmp/agent_lightning_old_verl_restoration_4gpu")
    if Path(runtime_config.get("output_dir", "")) != expected_tool_output:
        errors.append(f"tool output_dir={runtime_config.get('output_dir')!r}, expected {str(expected_tool_output)!r}")
if actual_model != expected_model:
    errors.append(f"composed base model mismatch: {actual_model} != {expected_model}")
if actual_adapter != expected_adapter:
    errors.append(f"composed adapter mismatch: {actual_adapter} != {expected_adapter}")
if config.trainer.experiment_name != expected_experiment_name:
    errors.append(
        f"composed experiment_name={config.trainer.experiment_name!r}, "
        f"expected {expected_experiment_name!r}"
    )
actual_output_dir = str(Path(config.trainer.default_local_dir).resolve())
if actual_output_dir != expected_output_dir:
    errors.append(f"composed output directory mismatch: {actual_output_dir} != {expected_output_dir}")
actual_swanlab_log_dir = str(
    Path(config.trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR).resolve()
)
if actual_swanlab_log_dir != expected_swanlab_log_dir:
    errors.append(
        f"composed SwanLab log directory mismatch: {actual_swanlab_log_dir} != {expected_swanlab_log_dir}"
    )
actual_verl_log_dir = str(
    Path(config.trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR).resolve()
)
if actual_verl_log_dir != expected_verl_log_dir:
    errors.append(f"composed VERL log directory mismatch: {actual_verl_log_dir} != {expected_verl_log_dir}")
actual_resume_path = config.trainer.resume_from_path
if actual_resume_path is not None:
    actual_resume_path = str(Path(actual_resume_path).resolve())
if actual_resume_path != expected_resume_path:
    errors.append(f"composed resume path mismatch: {actual_resume_path!r} != {expected_resume_path!r}")
if config.actor_rollout_ref.model.lora_rank != 16:
    errors.append(f"composed lora_rank={config.actor_rollout_ref.model.lora_rank!r}, expected 16")
if config.actor_rollout_ref.model.lora_alpha != 32:
    errors.append(f"composed lora_alpha={config.actor_rollout_ref.model.lora_alpha!r}, expected 32")
if config.actor_rollout_ref.ref.use_separate_lora_reference is not True:
    errors.append("frozen KL reference is not configured to retain the initial SFT LoRA")
if config.actor_rollout_ref.ref.fsdp_config.model_dtype != "bfloat16":
    errors.append(
        f"frozen reference model_dtype={config.actor_rollout_ref.ref.fsdp_config.model_dtype!r}, "
        "expected 'bfloat16'"
    )
if config.actor_rollout_ref.ref.fsdp_config.param_offload is not True:
    errors.append("frozen KL reference parameter offload is not enabled")
expected_action_rarity_coeff = os.environ.get("OLD_VERL_EXPECT_ACTION_RARITY_REWARD_COEFF")
if expected_action_rarity_coeff is not None:
    expected_action_rarity_coeff = float(expected_action_rarity_coeff)
    actual_action_rarity_coeff = float(config.algorithm.action_rarity_reward_coeff)
    if abs(actual_action_rarity_coeff - expected_action_rarity_coeff) > 1e-12:
        errors.append(
            f"action_rarity_reward_coeff={actual_action_rarity_coeff!r}, "
            f"expected {expected_action_rarity_coeff!r}"
        )
micro_batch_sizes = {
    "actor.ppo_micro_batch_size_per_gpu": (
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
        1,
    ),
    "rollout.log_prob_micro_batch_size_per_gpu": (
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
        2,
    ),
    "ref.log_prob_micro_batch_size_per_gpu": (
        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
        2,
    ),
}
for name, (value, expected) in micro_batch_sizes.items():
    if value != expected:
        errors.append(f"{name}={value!r}, expected {expected}")

if errors:
    raise SystemExit("Invalid composed RL config:\n- " + "\n- ".join(errors))
PY
  printf 'Preflight passed for %s: experiment=%s output=%s logs=%s swanlab=%s\n' \
    "${EXPERT}" \
    "${EXPERIMENT_NAME_OVERRIDE}" \
    "${OUTPUT_DIR_OVERRIDE}" \
    "${LOG_DIR}" \
    "${SWANLAB_LOG_DIR_OVERRIDE}"
  exit 0
fi

mkdir -p "${OLD_VERL_DIR}/data/4gpu" "${LOG_DIR}"
if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  mkdir -p "${OUTPUT_DIR_OVERRIDE}"
fi
if [[ -n "${SWANLAB_LOG_DIR_OVERRIDE}" ]]; then
  mkdir -p "${SWANLAB_LOG_DIR_OVERRIDE}"
fi

TOOL_INFO_LOG="${LOG_DIR}/restoration_tool_info.log"
TOOL_DEBUG_LOG="${LOG_DIR}/restoration_tools.log"
if [[ "${OLD_VERL_CLEAR_TOOL_LOGS:-1}" == "1" ]]; then
  : > "${TOOL_INFO_LOG}"
  : > "${TOOL_DEBUG_LOG}"
  echo "Cleared ${EXPERT} tool logs:"
else
  touch "${TOOL_INFO_LOG}" "${TOOL_DEBUG_LOG}"
  echo "Appending to ${EXPERT} tool logs:"
fi
echo "  ${TOOL_INFO_LOG}"
echo "  ${TOOL_DEBUG_LOG}"

if [[ "${OLD_VERL_CLEAR_PENALIZED_SAMPLES:-1}" == "1" ]]; then
  OUTPUT_DIR_ABS="$(realpath -m -- "${OUTPUT_DIR_OVERRIDE}")"
  PENALIZED_SAMPLES_DIR="$(realpath -m -- "${OUTPUT_DIR_ABS}/penalized_samples")"
  if [[ -z "${OUTPUT_DIR_ABS}" || "${OUTPUT_DIR_ABS}" == "/" || \
        "$(dirname -- "${PENALIZED_SAMPLES_DIR}")" != "${OUTPUT_DIR_ABS}" || \
        "$(basename -- "${PENALIZED_SAMPLES_DIR}")" != "penalized_samples" ]]; then
    echo "Refusing to clear an unsafe penalized-samples directory: '${PENALIZED_SAMPLES_DIR}'" >&2
    exit 2
  fi
  rm -rf -- "${PENALIZED_SAMPLES_DIR}"
  mkdir -p -- "${PENALIZED_SAMPLES_DIR}"
  echo "Cleared penalized samples: ${PENALIZED_SAMPLES_DIR}"
fi

if [[ "${OLD_VERL_CLEAR_INTERMEDIATE_IMAGES:-1}" == "1" ]]; then
  if [[ -z "${INTERMEDIATE_DIR}" || "${INTERMEDIATE_DIR}" == "/" ]]; then
    echo "Refusing to clear an unsafe intermediate image directory: '${INTERMEDIATE_DIR}'" >&2
    exit 2
  fi
  rm -rf -- "${INTERMEDIATE_DIR}"
  mkdir -p -- "${INTERMEDIATE_DIR}"
  echo "Cleared sampling intermediate images: ${INTERMEDIATE_DIR}"
fi

"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split train --output "${TRAIN_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${TRAIN_LIMIT_ARGS[@]}"
"${PYTHON_BIN}" "${CONVERTER}" --expert "${EXPERT}" --split val --output "${VAL_PARQUET}" --tool-registry "${TOOL_REGISTRY_PATH}" "${VAL_LIMIT_ARGS[@]}"

if [[ "${OLD_VERL_STOP_RAY:-1}" == "1" ]]; then
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
fi

echo "Running old-verl ${RUN_KIND} GRPO for ${EXPERT} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Model:   ${MODEL_PATH_OVERRIDE}"
echo "Adapter: ${ADAPTER_PATH_OVERRIDE}"
echo "Data:    ${TRAIN_PARQUET} / ${VAL_PARQUET}"
echo "Output:  ${OUTPUT_DIR_OVERRIDE}"
echo "Logs:    ${LOG_DIR}"
echo "Config:  ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "Resume:  ${RESUME_FROM_PATH_OVERRIDE:-fresh run}"
printf 'SwanLab: project=%s experiment=%s mode=%s log_dir=%s\n' \
  "${PROJECT_NAME_OVERRIDE:-configured by YAML}" \
  "${EXPERIMENT_NAME_OVERRIDE}" \
  "${SWANLAB_MODE_OVERRIDE:-configured by YAML}" \
  "${SWANLAB_LOG_DIR_OVERRIDE}"

cd "${ROOT}"
"${PYTHON_BIN}" -u -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  "${CONFIG_OVERRIDES[@]}" \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
