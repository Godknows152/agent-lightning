#!/usr/bin/env python3
"""Unified LoRA/SGLang and JarvisIR inference entry.

Two backends behind a single Hydra config:
  backend=lora_sglang -> VERL validation-only path (training-identical agent loop)
  backend=jarvisir    -> JarvisIR native vLLM inference (native tool chain)

Output: <output_root>/<run_name>/
  images/<sample_id>.png
  tool_calls.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

SCRIPT_PATH = Path(__file__).resolve()
OLD_VERL_DIR = SCRIPT_PATH.parents[2]
EXAMPLE_DIR = SCRIPT_PATH.parents[3]
PROJECT_ROOT = SCRIPT_PATH.parents[5]
BACKEND_ROOT = EXAMPLE_DIR / "verl_backend"
LOCAL_PYDEPS = OLD_VERL_DIR / ".pydeps"

EXPECTED_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
}


@dataclass(frozen=True)
class Topology:
    visible_devices: str
    trainer_gpu_count: int
    tensor_parallel_size: int
    tool_device: str


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    n: int


@dataclass(frozen=True)
class DatasetSelectionConfig:
    shuffle: bool
    validation_shuffle: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def validate_run_name(name: str) -> str:
    """Validate run.name is alphanumeric (dots, dashes, underscores allowed)."""
    if not re.match(r"^[A-Za-z0-9._-]+$", name):
        raise ValueError(
            f"run.name must be alphanumeric, got {name!r}"
        )
    return name


def normalized_path(value: str | Path) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def require_file(path: Path, desc: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{desc} is missing or empty: {path}")


def hydra_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def ray_temp_dir(attempt_dir: Path) -> Path:
    return Path("/tmp") / f"rb_{attempt_dir.name[-8:]}"


# ---------------------------------------------------------------------------
# LoRA adapter validation
# ---------------------------------------------------------------------------


def validate_adapter(adapter_path: Path, base_model: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    require_file(config_path, "LoRA adapter config")
    require_file(weights_path, "LoRA adapter weights")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    if cfg.get("peft_type") != "LORA":
        errors.append(f"peft_type={cfg.get('peft_type')!r}")
    if cfg.get("task_type") != "CAUSAL_LM":
        errors.append(f"task_type={cfg.get('task_type')!r}")
    if int(cfg.get("r", -1)) != 16:
        errors.append(f"r={cfg.get('r')!r}, expected 16")
    if int(cfg.get("lora_alpha", -1)) != 32:
        errors.append(f"lora_alpha={cfg.get('lora_alpha')!r}, expected 32")
    actual_targets = set(cfg.get("target_modules", []))
    if actual_targets != EXPECTED_TARGET_MODULES:
        errors.append(
            f"target_modules missing={sorted(EXPECTED_TARGET_MODULES - actual_targets)}, "
            f"unexpected={sorted(actual_targets - EXPECTED_TARGET_MODULES)}"
        )
    configured_base = Path(cfg.get("base_model_name_or_path", "")).expanduser().resolve()
    if configured_base != base_model.resolve():
        errors.append(f"base_model={configured_base}, expected {base_model.resolve()}")
    if errors:
        raise ValueError(f"Invalid LoRA adapter {adapter_path}:\n- " + "\n- ".join(errors))
    return {
        "path": str(adapter_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "rank": 16,
        "alpha": 32,
        "target_modules": sorted(actual_targets),
    }


# ---------------------------------------------------------------------------
# Parquet sample manifest
# ---------------------------------------------------------------------------


def load_validation_manifest(
    dataset_path: Path, expected_sample_count: int
) -> list[dict[str, Any]]:
    import pandas as pd

    require_file(dataset_path, "Validation parquet")
    frame = pd.read_parquet(dataset_path)
    required = {"prompt", "images", "extra_info", "reward_model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation parquet missing columns: {sorted(missing)}")
    if len(frame) != expected_sample_count:
        raise ValueError(
            f"Expected {expected_sample_count} samples, found {len(frame)}"
        )

    rows = []
    for row_index, row in frame.iterrows():
        extra = row["extra_info"]
        sample_id = str(extra.get("sample_id", f"fog-{row_index:06d}"))
        image_path = Path(str(extra["image_path"])).expanduser().resolve()
        require_file(image_path, f"Validation image for {sample_id}")
        create_kwargs = (
            extra.get("tools_kwargs", {})
            .get("restore_image", {})
            .get("create_kwargs", {})
        )
        configured_image = normalized_path(create_kwargs.get("image_path", image_path))
        if configured_image != normalized_path(image_path):
            raise ValueError(
                f"Tool image path mismatch for {sample_id}: "
                f"{configured_image} != {image_path}"
            )
        prompt_messages = (
            row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
        )
        prompt_text = json.dumps(
            prompt_messages, ensure_ascii=False, sort_keys=True, default=str
        )
        rows.append(
            {
                "row_index": int(row_index),
                "sample_id": sample_id,
                "original_image": str(image_path),
                "degradation_type": str(extra.get("degradation_type", "fog")),
                "prompt_version": str(extra.get("prompt_version", "")),
                "prompt_sha256": hashlib.sha256(
                    prompt_text.encode("utf-8")
                ).hexdigest(),
            }
        )
    sample_ids = [r["sample_id"] for r in rows]
    originals = [normalized_path(r["original_image"]) for r in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs are not unique.")
    if len(set(originals)) != len(originals):
        raise ValueError("Original image paths are not unique.")
    return rows


# ---------------------------------------------------------------------------
# Tool config isolation
# ---------------------------------------------------------------------------


def create_tool_config(
    template_path: Path, destination: Path, attempt_dir: Path, tool_device: str
) -> Path:
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    tc = payload["tools"][0]["config"]
    tc.update(
        {
            "device": tool_device,
            "worker_devices": [tool_device],
            "model_devices": [tool_device],
            "iqa_devices": [tool_device],
            "iqa_device": tool_device,
            "output_dir": str((attempt_dir / "tool_outputs").resolve()),
            "tool_result_cache_dir": str((attempt_dir / "tool_cache").resolve()),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    return destination


# ---------------------------------------------------------------------------
# LoRA backend: VERL hydra override building
# ---------------------------------------------------------------------------


def build_lora_overrides(
    *,
    adapter_path: Path,
    dataset_path: Path,
    attempt_dir: Path,
    tool_config_path: Path,
    topology: Topology,
    max_samples: int,
    sampling: SamplingConfig,
    ds: DatasetSelectionConfig,
) -> list[str]:
    logs_dir = attempt_dir / "logs"
    ray_dir = ray_temp_dir(attempt_dir)
    output_dir = attempt_dir / "trainer_output"
    validation_dir = attempt_dir / "validation_generations"
    experiment_name = f"lora_eval_{adapter_path.name}_{attempt_dir.name}"
    return [
        f"actor_rollout_ref.model.lora_adapter_path={hydra_string(adapter_path.resolve())}",
        f"actor_rollout_ref.rollout.temperature={sampling.temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={sampling.temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={sampling.top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={sampling.top_k}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(sampling.do_sample).lower()}",
        f"actor_rollout_ref.rollout.val_kwargs.n={sampling.n}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={hydra_string(tool_config_path.resolve())}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={topology.tensor_parallel_size}",
        f"data.val_files=[{hydra_string(dataset_path.resolve())}]",
        f"data.val_max_samples={max_samples}",
        f"data.shuffle={str(ds.shuffle).lower()}",
        f"data.validation_shuffle={str(ds.validation_shuffle).lower()}",
        f"trainer.n_gpus_per_node={topology.trainer_gpu_count}",
        f"trainer.ray_kwargs.ray_init.num_gpus={topology.trainer_gpu_count}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES={hydra_string(topology.visible_devices)}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR={hydra_string(attempt_dir / 'swanlab')}",
        f"trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR={hydra_string(logs_dir)}",
        f"+trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_SYNC_CPU_OFFLOAD={hydra_string('1')}",
        f"+trainer.ray_kwargs.ray_init._temp_dir={hydra_string(ray_dir)}",
        f"+ray_kwargs.ray_init.num_gpus={topology.trainer_gpu_count}",
        f"+ray_kwargs.ray_init._temp_dir={hydra_string(ray_dir)}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES={hydra_string(topology.visible_devices)}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES={hydra_string('1')}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO={hydra_string('0')}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR={hydra_string(logs_dir)}",
        f"+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_SYNC_CPU_OFFLOAD={hydra_string('1')}",
        "trainer.project_name=lora_eval",
        f"trainer.experiment_name={hydra_string(experiment_name)}",
        f"trainer.default_local_dir={hydra_string(output_dir)}",
        f"trainer.validation_data_dir={hydra_string(validation_dir)}",
        f"trainer.penalized_samples_dir={hydra_string(attempt_dir / 'penalized_samples')}",
        "trainer.logger=[console]",
        "trainer.log_val_generations=0",
        "trainer.val_before_train=true",
        "trainer.val_only=true",
        "trainer.resume_mode=disable",
        "trainer.resume_from_path=null",
        "trainer.keep_best_validation_ckpt=false",
    ]


def build_lora_env(attempt_dir: Path, topology: Topology) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(LOCAL_PYDEPS), str(BACKEND_ROOT), str(EXAMPLE_DIR)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": topology.visible_devices,
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "RAY_TMPDIR": str(ray_temp_dir(attempt_dir)),
            "VERL_LOG_DIR": str((attempt_dir / "logs").resolve()),
            "VERL_FSDP_SYNC_CPU_OFFLOAD": "1",
            "VERL_LOGGING_LEVEL": env.get("VERL_LOGGING_LEVEL", "WARN"),
            "SWANLAB_MODE": "disabled",
        }
    )
    lib_paths = [
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib",
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ]
    if env.get("LD_LIBRARY_PATH"):
        lib_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths)
    cudart = "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    env["LD_PRELOAD"] = (
        cudart + (f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else "")
    )
    env.pop("RAY_ADDRESS", None)
    return env


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def terminate_process_group(pid: int, timeout: float = 20.0) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def count_completed_trajectories(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if (
                '"event":"restoration_trajectory"' in line
                or '"event": "restoration_trajectory"' in line
            ):
                count += 1
    return count


# ---------------------------------------------------------------------------
# LoRA backend: run VERL validation process
# ---------------------------------------------------------------------------


def run_lora_validation(
    python: Path,
    config_dir: Path,
    config_name: str,
    overrides: list[str],
    attempt_dir: Path,
    topology: Topology,
    progress_label: str,
    expected_samples: int,
    progress_interval: int,
) -> None:
    command = [
        str(python),
        "-u",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(config_dir.resolve()),
        "--config-name",
        config_name,
        *overrides,
    ]
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_json(attempt_dir / "command.json", {"command": command, "created_at": utc_now()})
    log_path = attempt_dir / "benchmark.log"
    trajectory_log = attempt_dir / "logs/restoration_tool_info.log"
    env = build_lora_env(attempt_dir, topology)
    print(f"  Launching VERL validation-only; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            started_at = time.monotonic()
            last_completed = -1
            last_printed = 0.0
            while True:
                now = time.monotonic()
                completed = count_completed_trajectories(trajectory_log)
                if completed != last_completed or now - last_printed >= float(
                    progress_interval
                ):
                    elapsed = now - started_at
                    rate = completed / elapsed * 60.0 if elapsed > 0 else 0.0
                    print(
                        f"  [{progress_label}] {completed}/{expected_samples} "
                        f"elapsed={elapsed:.0f}s rate={rate:.2f} img/min",
                        flush=True,
                    )
                    last_completed = completed
                    last_printed = now
                rc = proc.poll()
                if rc is not None:
                    break
                time.sleep(5.0)
        except BaseException:
            terminate_process_group(proc.pid)
            raise
        finally:
            if proc.poll() is None:
                terminate_process_group(proc.pid)
    if proc.returncode != 0:
        raise RuntimeError(
            f"VERL validation failed (exit {proc.returncode}); inspect {log_path}"
        )


def collect_lora_outputs(
    manifest: list[dict[str, Any]],
    trajectory_log: Path,
    model_name: str,
) -> list[dict[str, Any]]:
    rows = [
        r
        for r in read_jsonl(trajectory_log)
        if r.get("event") == "restoration_trajectory"
    ]
    by_orig: dict[str, dict[str, Any]] = {}
    dupes = []
    missing_orig = 0
    for r in rows:
        orig = r.get("original_image")
        if not orig:
            missing_orig += 1
            continue
        key = normalized_path(orig)
        if key in by_orig:
            dupes.append(key)
        by_orig[key] = r
    if missing_orig:
        raise ValueError(f"{missing_orig} trajectories without original_image")
    if dupes:
        raise ValueError(f"{len(set(dupes))} duplicate trajectories")
    expected = {normalized_path(s["original_image"]) for s in manifest}
    unexpected = sorted(set(by_orig) - expected)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} trajectories outside configured parquet slice; "
            f"first: {unexpected[0]}"
        )

    outputs = []
    for sample in manifest:
        orig_path = Path(sample["original_image"]).resolve()
        traj = by_orig.get(normalized_path(orig_path))
        if traj is None:
            raise ValueError(
                f"Missing trajectory for {sample['sample_id']} ({orig_path})"
            )
        final_path = (
            Path(str(traj.get("final_image", orig_path))).expanduser().resolve()
        )
        if not final_path.is_file():
            raise FileNotFoundError(
                f"Final image missing for {sample['sample_id']}: {final_path}"
            )
        action_path = [str(a) for a in traj.get("action_path", [])]
        tool_calls = traj.get("tool_calls", [])
        duration = traj.get("duration_seconds")
        outputs.append(
            {
                "sample_id": sample["sample_id"],
                "model": model_name,
                "original_image": str(orig_path),
                "final_image": str(final_path),
                "action_path": action_path,
                "tool_calls": tool_calls,
                "duration_seconds": float(duration) if duration is not None else None,
            }
        )
    return outputs


# ---------------------------------------------------------------------------
# LoRA backend: full run
# ---------------------------------------------------------------------------


def run_lora_inference(
    run_name: str,
    adapter_path: Path,
    manifest: list[dict[str, Any]],
    dataset_path: Path,
    work_dir: Path,
    config_dir: Path,
    config_name: str,
    tool_template: Path,
    python: Path,
    topology: Topology,
    max_samples: int,
    sampling: SamplingConfig,
    ds: DatasetSelectionConfig,
    progress_interval: int,
    resume: bool,
) -> list[dict[str, Any]]:
    model_dir = work_dir / run_name
    attempt_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
    )
    attempt_dir = model_dir / "attempts" / attempt_name
    tool_config_path = attempt_dir / "tool_config.yaml"
    create_tool_config(
        tool_template, tool_config_path, attempt_dir, topology.tool_device
    )
    overrides = build_lora_overrides(
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        attempt_dir=attempt_dir,
        tool_config_path=tool_config_path,
        topology=topology,
        max_samples=max_samples,
        sampling=sampling,
        ds=ds,
    )
    selected = manifest[:max_samples]
    run_lora_validation(
        python,
        config_dir,
        config_name,
        overrides,
        attempt_dir,
        topology,
        run_name,
        len(selected),
        progress_interval,
    )
    trajectory_log = attempt_dir / "logs/restoration_tool_info.log"
    require_file(trajectory_log, "Trajectory log")
    outputs = collect_lora_outputs(selected, trajectory_log, run_name)
    success_marker = {
        "run_name": run_name,
        "completed_at": utc_now(),
        "attempt_dir": str(attempt_dir.resolve()),
        "sample_count": len(outputs),
    }
    write_json(attempt_dir / "SUCCESS.json", success_marker)
    print(f"  Completed {run_name}: {len(outputs)} samples.", flush=True)
    return outputs


# ---------------------------------------------------------------------------
# JarvisIR backend
# ---------------------------------------------------------------------------


def run_jarvisir_inference(
    python: Path,
    entrypoint: Path,
    model_path: Path,
    repo_root: Path,
    state_dir: Path,
    manifest: list[dict[str, Any]],
    progress_interval: int,
) -> list[dict[str, Any]]:
    selected = manifest
    input_dir = state_dir / "jarvisir_input" / "selected"
    input_dir.mkdir(parents=True, exist_ok=True)
    for sample in selected:
        src = Path(sample["original_image"]).resolve()
        dst = input_dir / f"{sample['sample_id']}.png"
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    raw_output = state_dir / "jarvisir_raw"
    raw_output.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        str(entrypoint),
        f"paths.input_root={hydra_string(input_dir.parent.resolve())}",
        f"paths.output_root={hydra_string(raw_output.resolve())}",
        f"paths.model_path={hydra_string(model_path.resolve())}",
        "data.subdirs=[selected]",
        f"data.max_images={len(selected)}",
        "runtime.execution_mode=split",
        "runtime.strategy=vlm",
        "runtime.vlm_backend=vllm",
        "runtime.write_logs=true",
        "runtime.copy_on_no_tool=false",
        "runtime.resume=true",
        "model.policy_batch_size=256",
        "model.policy_image_size=512",
        "model.max_new_tokens=400",
        "model.torch_dtype=float16",
        "model.vllm.tensor_parallel_size=2",
        "model.vllm.pipeline_parallel_size=1",
        "model.vllm.gpu_memory_utilization=0.55",
        "model.vllm.max_num_seqs=256",
        "tool.gpus=[2]",
        "tool.enable_iqa=true",
        "tool.preload_models=true",
    ]
    log_path = state_dir / "jarvisir_run.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    print(f"  Launching JarvisIR inference; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            started_at = time.monotonic()
            last_printed = 0.0
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                now = time.monotonic()
                if now - last_printed >= float(progress_interval):
                    completed = 0
                    manifest_dir = raw_output / "_logs"
                    if manifest_dir.is_dir():
                        for mf in sorted(manifest_dir.glob("manifest*.jsonl")):
                            completed += sum(1 for _ in mf.open(encoding="utf-8"))
                    elapsed = now - started_at
                    rate = completed / elapsed * 60.0 if elapsed > 0 else 0.0
                    print(
                        f"  [JarvisIR] {completed}/{len(selected)} "
                        f"elapsed={elapsed:.0f}s rate={rate:.2f} img/min",
                        flush=True,
                    )
                    last_printed = now
                time.sleep(5.0)
        except BaseException:
            terminate_process_group(proc.pid)
            raise
        finally:
            if proc.poll() is None:
                terminate_process_group(proc.pid)
    if proc.returncode != 0:
        raise RuntimeError(
            f"JarvisIR failed (exit {proc.returncode}); inspect {log_path}"
        )

    # Read manifest
    manifest_dir = raw_output / "_logs"
    manifest_files = sorted(manifest_dir.glob("manifest*.jsonl")) if manifest_dir.is_dir() else []
    if not manifest_files:
        raise RuntimeError("No JarvisIR manifest files found")
    tool_manifests = [mf for mf in manifest_files if "tool_gpu" in mf.name]
    primary = tool_manifests if tool_manifests else manifest_files
    records = []
    for mf in primary:
        records.extend(read_jsonl(mf))
    final = [r for r in records if r.get("status") == "ok"]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for r in final:
        fname = Path(r["input_path"]).stem
        by_sample.setdefault(fname, []).append(r)
    outputs = []
    for sample in selected:
        sid = sample["sample_id"]
        candidates = by_sample.get(sid, [])
        if not candidates:
            raise RuntimeError(f"JarvisIR missing successful output for {sid}")
        r = candidates[-1]
        output_path = Path(r["output_path"]).resolve()
        if not output_path.is_file():
            raise FileNotFoundError(
                f"JarvisIR output missing for {sid}: {output_path}"
            )
        tools = r.get("tools", [])
        outputs.append(
            {
                "sample_id": sid,
                "model": "JarvisIR",
                "original_image": sample["original_image"],
                "final_image": str(output_path),
                "action_path": tools,
                "tool_calls": [{"action": t, "status": "success"} for t in tools],
                "duration_seconds": None,
            }
        )
    if len(outputs) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} JarvisIR outputs, got {len(outputs)}"
        )
    print(f"  Completed JarvisIR: {len(outputs)} samples.", flush=True)
    return outputs


# ---------------------------------------------------------------------------
# Publish tool calls CSV
# ---------------------------------------------------------------------------


def publish_images_and_csv(output_dir: Path, outputs: list[dict[str, Any]]) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for out in outputs:
        src = Path(out["final_image"])
        dst = images_dir / f"{out['sample_id']}.png"
        shutil.copy2(src, dst)
        action_path = out.get("action_path", [])
        tool_count = sum(1 for a in action_path if a != "stop")
        csv_rows.append(
            {
                "sample_id": out["sample_id"],
                "image": f"images/{out['sample_id']}.png",
                "restoration_tool_call_count": tool_count,
            }
        )
    write_csv(
        output_dir / "tool_calls.csv",
        csv_rows,
        ["sample_id", "image", "restoration_tool_call_count"],
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_lora(
    adapter_path: Path,
    base_model: Path,
    tool_config: Path,
    config_dir: Path,
    config_name: str,
) -> None:
    require_file(base_model / "config.json", "Base model config")
    require_file(tool_config, "Tool config")
    require_file(config_dir / f"{config_name}.yaml", "Training config")
    validate_adapter(adapter_path, base_model)


def preflight_jarvisir(python: Path, entrypoint: Path, model_path: Path) -> None:
    require_file(python, "Python executable")
    require_file(entrypoint, "JarvisIR entrypoint")
    require_file(
        model_path / "config.json" if model_path.is_dir() else model_path,
        "JarvisIR model",
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../config/eval",
    config_name="restoration_inference",
)
def main(config: DictConfig) -> int:
    cfg = OmegaConf.to_container(config, resolve=True)
    run_name = str(cfg["run"]["name"])
    validate_run_name(run_name)
    output_root = Path(cfg["paths"]["output_root"]).resolve()
    state_root = Path(cfg["paths"]["state_root"]).resolve()
    log_root = Path(cfg["paths"]["log_root"]).resolve()
    dataset_path = Path(cfg["data"]["parquet_path"]).resolve()
    expected_total = int(cfg["data"]["expected_total_samples"])
    offset = int(cfg["data"]["offset"])
    max_samples = int(cfg["data"]["max_samples"])
    progress_interval = int(cfg["runtime"]["progress_interval_seconds"])
    backend = str(cfg["backend"]["name"])

    if cfg.get("command") == "preflight":
        if backend == "lora_sglang":
            preflight_lora(
                Path(cfg["backend"]["adapter_path"]).resolve(),
                Path(cfg["backend"]["base_model"]).resolve(),
                Path(cfg["backend"]["tool_config"]).resolve(),
                Path(cfg["backend"]["source_config"]["dir"]).resolve(),
                cfg["backend"]["source_config"]["name"],
            )
        else:
            preflight_jarvisir(
                Path(cfg["backend"]["python"]).resolve(),
                Path(cfg["backend"]["entrypoint"]).resolve(),
                Path(cfg["backend"]["model_path"]).resolve(),
            )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "backend": backend,
                    "run_name": run_name,
                    "max_samples": max_samples,
                },
                indent=2,
            )
        )
        return 0

    print(
        f"Restoration eval: backend={backend} run_name={run_name} max_samples={max_samples}",
        flush=True,
    )

    full_manifest = load_validation_manifest(dataset_path, expected_total)
    selected = full_manifest[offset : offset + max_samples]
    print(
        f"  Dataset: {len(full_manifest)} total, selected {len(selected)} (offset={offset})",
        flush=True,
    )

    output_dir = output_root / run_name
    if output_dir.exists() and not cfg.get("run", {}).get("overwrite", False):
        raise FileExistsError(
            f"Output directory exists: {output_dir}. Set run.overwrite=true to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "lora_sglang":
        adapter_path = Path(cfg["backend"]["adapter_path"]).resolve()
        base_model = Path(cfg["backend"]["base_model"]).resolve()
        tool_config = Path(cfg["backend"]["tool_config"]).resolve()
        config_dir = Path(cfg["backend"]["source_config"]["dir"]).resolve()
        config_name = str(cfg["backend"]["source_config"]["name"])
        python = Path(cfg["backend"]["python"]).resolve()
        topology = Topology(
            visible_devices=str(cfg["backend"]["gpu"]["visible_devices"]),
            trainer_gpu_count=int(cfg["backend"]["gpu"]["trainer_gpu_count"]),
            tensor_parallel_size=int(cfg["backend"]["gpu"]["tensor_parallel_size"]),
            tool_device=str(cfg["backend"]["gpu"]["tool_device"]),
        )
        sampling = SamplingConfig(
            temperature=float(cfg["backend"]["sampling"]["temperature"]),
            top_p=float(cfg["backend"]["sampling"]["top_p"]),
            top_k=int(cfg["backend"]["sampling"]["top_k"]),
            do_sample=bool(cfg["backend"]["sampling"]["do_sample"]),
            n=int(cfg["backend"]["sampling"]["n"]),
        )
        ds = DatasetSelectionConfig(shuffle=False, validation_shuffle=False)
        preflight_lora(adapter_path, base_model, tool_config, config_dir, config_name)
        work_dir = state_root / run_name
        outputs = run_lora_inference(
            run_name=run_name,
            adapter_path=adapter_path,
            manifest=selected,
            dataset_path=dataset_path,
            work_dir=work_dir,
            config_dir=config_dir,
            config_name=config_name,
            tool_template=tool_config,
            python=python,
            topology=topology,
            max_samples=max_samples,
            sampling=sampling,
            ds=ds,
            progress_interval=progress_interval,
            resume=cfg.get("run", {}).get("resume", True),
        )
    elif backend == "jarvisir":
        python = Path(cfg["backend"]["python"]).resolve()
        entrypoint = Path(cfg["backend"]["entrypoint"]).resolve()
        model_path = Path(cfg["backend"]["model_path"]).resolve()
        repo_root = Path(cfg["backend"]["repo_root"]).resolve()
        preflight_jarvisir(python, entrypoint, model_path)
        state_dir = state_root / run_name
        outputs = run_jarvisir_inference(
            python=python,
            entrypoint=entrypoint,
            model_path=model_path,
            repo_root=repo_root,
            state_dir=state_dir,
            manifest=selected,
            progress_interval=progress_interval,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    publish_images_and_csv(output_dir, outputs)
    print(f"Output published: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
