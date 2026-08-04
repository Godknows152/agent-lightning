#!/usr/bin/env python3
"""Benchmark configured Fog LoRAs through the real VERL validation path.

Formal runs use physical GPU0/1 for the TP2 SGLang rollout and physical GPU1
for the persistent restoration/IQA runtime. Development smoke runs are isolated
to physical GPU3. The two adapters are evaluated sequentially and their final
images are scored independently with MUSIQ, MANIQA, CLIP-IQA, and LIQE.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
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
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

SCRIPT_PATH = Path(__file__).resolve()
OLD_VERL_DIR = SCRIPT_PATH.parents[2]
EXAMPLE_DIR = SCRIPT_PATH.parents[3]
PROJECT_ROOT = SCRIPT_PATH.parents[5]
BACKEND_ROOT = EXAMPLE_DIR / "verl_backend"
LOCAL_PYDEPS = OLD_VERL_DIR / ".pydeps"
IQA_REPO = PROJECT_ROOT / "External_Tools/iqa_repos/IQA-PyTorch"
BENCHMARK_CONFIG_DIR = OLD_VERL_DIR / "config/eval"
BENCHMARK_CONFIG_NAME = "fog_lora_benchmark"

EXPECTED_TARGET_MODULES = {
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


@dataclass(frozen=True)
class Topology:
    visible_devices: str
    trainer_gpu_count: int
    tensor_parallel_size: int
    tool_device: str
    label: str


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


@dataclass(frozen=True)
class BenchmarkSettings:
    command: str
    resume: bool
    python: Path
    base_model: Path
    dataset: Path
    expected_dataset_samples: int
    max_samples: int
    dataset_selection: DatasetSelectionConfig
    model_adapters: dict[str, Path]
    config_dir: Path
    config_name: str
    tool_config: Path
    benchmark_dir: Path
    work_dir: Path
    sampling: SamplingConfig
    metrics: tuple[str, ...]
    formal_topology: Topology
    formal_sampling_gpus: tuple[int, ...]
    scoring_gpu: int
    smoke_model: str
    smoke_max_samples: int
    smoke_topology: Topology
    score_manifest: Path | None
    score_progress: Path | None
    resolved_config: dict[str, Any]

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.model_adapters)


def absolute_path(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{description} must be an absolute path: {value}")
    return path.resolve()


def topology_from_config(config: DictConfig) -> Topology:
    return Topology(
        visible_devices=str(config.visible_devices),
        trainer_gpu_count=int(config.trainer_gpu_count),
        tensor_parallel_size=int(config.tensor_parallel_size),
        tool_device=str(config.tool_device),
        label=str(config.label),
    )


def settings_from_config(config: DictConfig) -> BenchmarkSettings:
    model_adapters: dict[str, Path] = {}
    for model in config.models:
        name = str(model.name)
        if name in model_adapters:
            raise ValueError(f"Duplicate benchmark model name: {name}")
        model_adapters[name] = absolute_path(str(model.adapter_path), f"LoRA path for {name}")
    if len(model_adapters) < 2:
        raise ValueError(f"At least two benchmark models are required, found {tuple(model_adapters)}")

    metrics = tuple(str(metric) for metric in config.iqa_metrics)
    if metrics != ("musiq", "maniqa", "clipiqa", "liqe"):
        raise ValueError(f"IQA metrics must be MUSIQ, MANIQA, CLIP-IQA, and LIQE, found {metrics}")
    max_samples = int(config.max_samples)
    expected_dataset_samples = int(config.expected_dataset_samples)
    if max_samples <= 0 or max_samples > expected_dataset_samples:
        raise ValueError(
            f"max_samples must be in [1, {expected_dataset_samples}], found {max_samples}"
        )
    dataset_selection = DatasetSelectionConfig(
        shuffle=bool(config.dataset_selection.shuffle),
        validation_shuffle=bool(config.dataset_selection.validation_shuffle),
    )
    if dataset_selection.shuffle or dataset_selection.validation_shuffle:
        raise ValueError("Fog benchmark must disable dataset and validation shuffling to use parquet rows 0-99")
    smoke_model = str(config.smoke.model)
    if smoke_model not in model_adapters:
        raise ValueError(f"smoke.model={smoke_model!r} is not present in configured models")

    score_manifest = config.get("score_manifest")
    score_progress = config.get("score_progress")
    return BenchmarkSettings(
        command=str(config.command),
        resume=bool(config.resume),
        python=absolute_path(str(config.python), "Python executable"),
        base_model=absolute_path(str(config.base_model), "Base model"),
        dataset=absolute_path(str(config.dataset), "Validation dataset"),
        expected_dataset_samples=expected_dataset_samples,
        max_samples=max_samples,
        dataset_selection=dataset_selection,
        model_adapters=model_adapters,
        config_dir=absolute_path(str(config.source_config.dir), "Source config directory"),
        config_name=str(config.source_config.name),
        tool_config=absolute_path(str(config.tool_config), "Tool config"),
        benchmark_dir=absolute_path(str(config.output_dir), "Benchmark output directory"),
        work_dir=absolute_path(str(config.work_dir), "Benchmark work directory"),
        sampling=SamplingConfig(
            temperature=float(config.sampling.temperature),
            top_p=float(config.sampling.top_p),
            top_k=int(config.sampling.top_k),
            do_sample=bool(config.sampling.do_sample),
            n=int(config.sampling.n),
        ),
        metrics=metrics,
        formal_topology=topology_from_config(config.formal_topology),
        formal_sampling_gpus=tuple(int(gpu) for gpu in config.formal_topology.sampling_gpus),
        scoring_gpu=int(config.formal_topology.scoring_gpu),
        smoke_model=smoke_model,
        smoke_max_samples=int(config.smoke.max_samples),
        smoke_topology=topology_from_config(config.smoke.topology),
        score_manifest=absolute_path(str(score_manifest), "IQA score manifest") if score_manifest else None,
        score_progress=absolute_path(str(score_progress), "IQA score progress") if score_progress else None,
        resolved_config=OmegaConf.to_container(config, resolve=True),
    )


def compose_benchmark_config(overrides: list[str] | None = None) -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(BENCHMARK_CONFIG_DIR.resolve())):
        return compose(config_name=BENCHMARK_CONFIG_NAME, overrides=overrides or [])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def normalized_path(value: str | Path) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")


def validate_adapter(adapter_path: Path, base_model: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    require_file(config_path, "LoRA adapter config")
    require_file(weights_path, "LoRA adapter weights")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    if config.get("peft_type") != "LORA":
        errors.append(f"peft_type={config.get('peft_type')!r}")
    if config.get("task_type") != "CAUSAL_LM":
        errors.append(f"task_type={config.get('task_type')!r}")
    if int(config.get("r", -1)) != 16:
        errors.append(f"r={config.get('r')!r}, expected 16")
    if int(config.get("lora_alpha", -1)) != 32:
        errors.append(f"lora_alpha={config.get('lora_alpha')!r}, expected 32")
    actual_targets = set(config.get("target_modules", []))
    if actual_targets != EXPECTED_TARGET_MODULES:
        errors.append(
            f"target_modules missing={sorted(EXPECTED_TARGET_MODULES - actual_targets)}, "
            f"unexpected={sorted(actual_targets - EXPECTED_TARGET_MODULES)}"
        )
    configured_base = Path(config.get("base_model_name_or_path", "")).expanduser().resolve()
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


def load_validation_manifest(dataset_path: Path, expected_sample_count: int) -> list[dict[str, Any]]:
    import pandas as pd

    require_file(dataset_path, "Fog validation parquet")
    frame = pd.read_parquet(dataset_path)
    required_columns = {"prompt", "images", "extra_info", "reward_model"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Validation parquet is missing columns: {sorted(missing)}")
    if len(frame) != expected_sample_count:
        raise ValueError(f"Expected {expected_sample_count} Fog validation samples, found {len(frame)}.")

    rows = []
    for row_index, row in frame.iterrows():
        extra = row["extra_info"]
        sample_id = str(extra.get("sample_id", f"fog-{row_index:06d}"))
        image_path = Path(str(extra["image_path"])).expanduser().resolve()
        require_file(image_path, f"Validation image for {sample_id}")
        create_kwargs = extra.get("tools_kwargs", {}).get("restore_image", {}).get("create_kwargs", {})
        configured_image = normalized_path(create_kwargs.get("image_path", image_path))
        if configured_image != normalized_path(image_path):
            raise ValueError(f"Tool image path mismatch for {sample_id}: {configured_image} != {image_path}")
        prompt_messages = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
        prompt_text = json.dumps(prompt_messages, ensure_ascii=False, sort_keys=True, default=str)
        rows.append(
            {
                "row_index": int(row_index),
                "sample_id": sample_id,
                "original_image": str(image_path),
                "degradation_type": str(extra.get("degradation_type", "fog")),
                "prompt_version": str(extra.get("prompt_version", "")),
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            }
        )
    sample_ids = [row["sample_id"] for row in rows]
    originals = [normalized_path(row["original_image"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Validation sample IDs are not unique.")
    if len(set(originals)) != len(originals):
        raise ValueError("Validation original image paths are not unique.")
    return rows


def parse_nvidia_csv(output: str) -> list[list[str]]:
    return [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]


def query_gpu_processes() -> dict[int, list[dict[str, Any]]]:
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid_to_index = {parts[1]: int(parts[0]) for parts in parse_nvidia_csv(gpu_query.stdout)}
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[int, list[dict[str, Any]]] = {index: [] for index in uuid_to_index.values()}
    for parts in parse_nvidia_csv(process_query.stdout):
        if len(parts) < 4 or parts[0] not in uuid_to_index:
            continue
        result[uuid_to_index[parts[0]]].append(
            {"pid": int(parts[1]), "process_name": parts[2], "used_memory_mib": parts[3]}
        )
    return result


def require_gpus_idle(indices: Iterable[int]) -> None:
    processes = query_gpu_processes()
    busy = {index: processes.get(index, []) for index in indices if processes.get(index)}
    if busy:
        details = "; ".join(
            f"GPU{index}: "
            + ", ".join(f"pid={item['pid']} {item['process_name']} {item['used_memory_mib']}MiB" for item in items)
            for index, items in busy.items()
        )
        raise RuntimeError(f"Required benchmark GPUs are busy: {details}")


def wait_for_gpus_idle(indices: Iterable[int], timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            require_gpus_idle(indices)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2.0)


def create_tool_config(template_path: Path, destination: Path, attempt_dir: Path, topology: Topology) -> Path:
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    tool_config = payload["tools"][0]["config"]
    device = topology.tool_device
    tool_config.update(
        {
            "device": device,
            "worker_devices": [device],
            "model_devices": [device],
            "iqa_devices": [device],
            "iqa_device": device,
            "output_dir": str((attempt_dir / "tool_outputs").resolve()),
            "tool_result_cache_dir": str((attempt_dir / "tool_cache").resolve()),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return destination


def hydra_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def ray_temp_dir(attempt_dir: Path) -> Path:
    """Return an isolated short path that leaves room for Ray's Unix sockets."""
    return Path("/tmp") / f"rb_{attempt_dir.name[-8:]}"


def build_hydra_overrides(
    *,
    adapter_path: Path,
    dataset_path: Path,
    attempt_dir: Path,
    tool_config_path: Path,
    topology: Topology,
    max_samples: int,
    sampling: SamplingConfig,
    dataset_selection: DatasetSelectionConfig,
) -> list[str]:
    logs_dir = attempt_dir / "logs"
    ray_dir = ray_temp_dir(attempt_dir)
    output_dir = attempt_dir / "trainer_output"
    validation_dir = attempt_dir / "validation_generations"
    experiment_name = f"fog_lora_benchmark_{adapter_path.name}_{attempt_dir.name}"
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
        f"data.shuffle={str(dataset_selection.shuffle).lower()}",
        f"data.validation_shuffle={str(dataset_selection.validation_shuffle).lower()}",
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
        f"trainer.project_name={hydra_string('fog_lora_benchmark')}",
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


def compose_config(config_dir: Path, config_name: str, overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # The benchmark itself is a Hydra app, while VERL owns a separate Hydra
    # config tree. The outer context is no longer needed after settings parsing.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=config_name, overrides=overrides)


def validate_composed_config(
    config: Any,
    topology: Topology,
    adapter_path: Path,
    dataset_path: Path,
    max_samples: int,
    sampling: SamplingConfig,
    dataset_selection: DatasetSelectionConfig,
) -> None:
    errors = []
    if Path(str(config.actor_rollout_ref.model.lora_adapter_path)).resolve() != adapter_path.resolve():
        errors.append("LoRA adapter override did not compose correctly")
    if Path(str(config.data.val_files[0])).resolve() != dataset_path.resolve():
        errors.append("validation parquet override did not compose correctly")
    if float(config.actor_rollout_ref.rollout.temperature) != sampling.temperature:
        errors.append("rollout temperature mismatch")
    val_kwargs = config.actor_rollout_ref.rollout.val_kwargs
    expected_val = {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "do_sample": sampling.do_sample,
        "n": sampling.n,
    }
    for key, expected in expected_val.items():
        if val_kwargs.get(key) != expected:
            errors.append(f"val_kwargs.{key}={val_kwargs.get(key)!r}, expected {expected!r}")
    if int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != topology.tensor_parallel_size:
        errors.append("tensor parallel size mismatch")
    if int(config.trainer.n_gpus_per_node) != topology.trainer_gpu_count:
        errors.append("trainer GPU count mismatch")
    if int(config.data.val_max_samples) != max_samples:
        errors.append("validation sample limit mismatch")
    if bool(config.data.shuffle) != dataset_selection.shuffle:
        errors.append("dataset shuffle mismatch")
    if bool(config.data.validation_shuffle) != dataset_selection.validation_shuffle:
        errors.append("validation dataloader shuffle mismatch")
    if bool(config.data.shuffle) or bool(config.data.validation_shuffle):
        errors.append("benchmark must preserve parquet order and use rows 0 through max_samples-1")
    if int(config.ray_kwargs.ray_init.num_gpus) != topology.trainer_gpu_count:
        errors.append("Ray initialization GPU count mismatch")
    if str(config.ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES) != topology.visible_devices:
        errors.append("Ray runtime visible devices mismatch")
    if str(config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_SYNC_CPU_OFFLOAD) != "1":
        errors.append("synchronous FSDP CPU offload is not enabled")
    if not bool(config.trainer.val_before_train) or not bool(config.trainer.val_only):
        errors.append("validation-only mode is not enabled")
    if str(config.trainer.resume_mode) != "disable":
        errors.append("checkpoint resume must be disabled")
    if errors:
        raise ValueError("Invalid composed benchmark config:\n- " + "\n- ".join(errors))


def build_runtime_environment(attempt_dir: Path, topology: Topology) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(LOCAL_PYDEPS), str(BACKEND_ROOT), str(EXAMPLE_DIR)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
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
            "VERL_LOGGING_LEVEL": environment.get("VERL_LOGGING_LEVEL", "WARN"),
            "SWANLAB_MODE": "disabled",
        }
    )
    library_paths = [
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib",
        "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    cudart = "/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
    environment["LD_PRELOAD"] = cudart + (f":{environment['LD_PRELOAD']}" if environment.get("LD_PRELOAD") else "")
    environment.pop("RAY_ADDRESS", None)
    return environment


def terminate_process_group(process_group_id: int, timeout_seconds: float = 20.0) -> None:
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def count_completed_trajectories(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if '"event":"restoration_trajectory"' in line or '"event": "restoration_trajectory"' in line:
                count += 1
    return count


def run_validation_process(
    python: Path,
    config_dir: Path,
    config_name: str,
    overrides: list[str],
    attempt_dir: Path,
    topology: Topology,
    progress_label: str,
    expected_samples: int,
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
    if (attempt_dir / "command.json").exists():
        raise FileExistsError(f"Benchmark attempt already contains a command manifest: {attempt_dir}")
    (attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_json(attempt_dir / "command.json", {"command": command, "topology": topology.__dict__, "created_at": utc_now()})
    log_path = attempt_dir / "benchmark.log"
    trajectory_log = attempt_dir / "logs/restoration_tool_info.log"
    environment = build_runtime_environment(attempt_dir, topology)
    print(f"Launching validation-only run; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            started_at = time.monotonic()
            last_completed = -1
            last_reported_at = 0.0
            while True:
                now = time.monotonic()
                completed = count_completed_trajectories(trajectory_log)
                if completed != last_completed or now - last_reported_at >= 30.0:
                    elapsed_seconds = now - started_at
                    rate_per_minute = completed / elapsed_seconds * 60.0 if elapsed_seconds > 0 else 0.0
                    print(
                        f"[{progress_label}] inference progress: {completed}/{expected_samples}; "
                        f"elapsed={elapsed_seconds:.0f}s; rate={rate_per_minute:.2f} images/min",
                        flush=True,
                    )
                    last_completed = completed
                    last_reported_at = now
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(5.0)
        except BaseException:
            terminate_process_group(process.pid)
            raise
        finally:
            terminate_process_group(process.pid)
    if return_code != 0:
        raise RuntimeError(f"Validation process failed with exit code {return_code}; inspect {log_path}")


def collect_final_outputs(
    validation_manifest: list[dict[str, Any]], trajectory_log: Path, model_name: str
) -> list[dict[str, Any]]:
    trajectory_rows = [row for row in read_jsonl(trajectory_log) if row.get("event") == "restoration_trajectory"]
    by_original: dict[str, dict[str, Any]] = {}
    duplicate_originals = []
    missing_original_count = 0
    for row in trajectory_rows:
        original = row.get("original_image")
        if not original:
            missing_original_count += 1
            continue
        key = normalized_path(original)
        if key in by_original:
            duplicate_originals.append(key)
        by_original[key] = row
    if missing_original_count:
        raise ValueError(f"Found {missing_original_count} trajectory summaries without original_image.")
    if duplicate_originals:
        raise ValueError(f"Duplicate trajectory summaries for {len(set(duplicate_originals))} validation images.")
    expected_originals = {normalized_path(sample["original_image"]) for sample in validation_manifest}
    unexpected_originals = sorted(set(by_original) - expected_originals)
    if unexpected_originals:
        raise ValueError(
            f"Found {len(unexpected_originals)} trajectories outside the configured parquet head; "
            f"first unexpected image: {unexpected_originals[0]}"
        )

    outputs = []
    for sample in validation_manifest:
        original_path = Path(sample["original_image"]).resolve()
        trajectory = by_original.get(normalized_path(original_path))
        fallback_reason = None
        tool_calls: list[dict[str, Any]] = []
        duration_seconds: float | None = None
        if trajectory is None:
            final_path = original_path
            fallback_reason = "missing_trajectory"
            action_path: list[str] = []
            termination_reason = "no_trajectory"
            trajectory_id = None
        else:
            requested_final = Path(str(trajectory.get("final_image") or original_path)).expanduser()
            final_path = requested_final.resolve()
            if not final_path.is_file():
                final_path = original_path
                fallback_reason = "missing_final_image"
            action_path = [str(action) for action in trajectory.get("action_path", [])]
            termination_reason = str(trajectory.get("termination_reason", "unknown"))
            trajectory_id = trajectory.get("trajectory_id")
            duration_value = trajectory.get("duration_seconds")
            duration_seconds = float(duration_value) if duration_value is not None else None
            for call in trajectory.get("tool_calls", []):
                tool_calls.append(
                    {
                        "call_index": call.get("call_index"),
                        "action": call.get("action"),
                        "status": call.get("status"),
                        "restoration_step": call.get("restoration_step"),
                    }
                )
        outputs.append(
            {
                "sample_id": sample["sample_id"],
                "model": model_name,
                "original_image": str(original_path),
                "final_image": str(final_path),
                "used_original_fallback": fallback_reason is not None,
                "fallback_reason": fallback_reason,
                "trajectory_id": trajectory_id,
                "termination_reason": termination_reason,
                "action_path": action_path,
                "tool_call_count": len(action_path),
                "duration_seconds": duration_seconds,
                "tool_calls": tool_calls,
            }
        )
    return outputs


def completed_model_outputs(model_dir: Path, validation_manifest: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    marker_path = model_dir / "latest_success.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    outputs_path = Path(marker["outputs_path"])
    rows = read_jsonl(outputs_path)
    if len(rows) != len(validation_manifest):
        return None
    expected_ids = {row["sample_id"] for row in validation_manifest}
    if {row.get("sample_id") for row in rows} != expected_ids:
        return None
    if any(not Path(str(row.get("final_image", ""))).is_file() for row in rows):
        return None
    return rows


def run_one_model(
    *,
    model_name: str,
    adapter_path: Path,
    validation_manifest: list[dict[str, Any]],
    dataset_path: Path,
    work_dir: Path,
    config_dir: Path,
    config_name: str,
    tool_template: Path,
    python: Path,
    topology: Topology,
    max_samples: int,
    sampling: SamplingConfig,
    dataset_selection: DatasetSelectionConfig,
    resume: bool,
) -> list[dict[str, Any]]:
    model_dir = work_dir / model_name
    if resume:
        completed = completed_model_outputs(model_dir, validation_manifest[:max_samples] if max_samples > 0 else validation_manifest)
        if completed is not None:
            print(f"Skipping completed {model_name} inference ({len(completed)} samples).", flush=True)
            return completed

    attempt_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
    attempt_dir = model_dir / "attempts" / attempt_name
    tool_config_path = attempt_dir / "tool_config.yaml"
    create_tool_config(tool_template, tool_config_path, attempt_dir, topology)
    overrides = build_hydra_overrides(
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        attempt_dir=attempt_dir,
        tool_config_path=tool_config_path,
        topology=topology,
        max_samples=max_samples,
        sampling=sampling,
        dataset_selection=dataset_selection,
    )
    config = compose_config(config_dir, config_name, overrides)
    validate_composed_config(
        config,
        topology,
        adapter_path,
        dataset_path,
        max_samples,
        sampling,
        dataset_selection,
    )
    selected_manifest = validation_manifest if max_samples < 0 else validation_manifest[:max_samples]
    run_validation_process(
        python,
        config_dir,
        config_name,
        overrides,
        attempt_dir,
        topology,
        model_name,
        len(selected_manifest),
    )
    trajectory_log = attempt_dir / "logs/restoration_tool_info.log"
    require_file(trajectory_log, "Restoration trajectory log")
    outputs = collect_final_outputs(selected_manifest, trajectory_log, model_name)
    outputs_path = attempt_dir / "final_outputs.jsonl"
    write_jsonl(outputs_path, outputs)
    marker = {
        "model": model_name,
        "completed_at": utc_now(),
        "attempt_dir": str(attempt_dir.resolve()),
        "outputs_path": str(outputs_path.resolve()),
        "sample_count": len(outputs),
        "fallback_count": sum(bool(row["used_original_fallback"]) for row in outputs),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
    }
    write_json(attempt_dir / "SUCCESS.json", marker)
    write_json(model_dir / "latest_success.json", marker)
    print(f"Completed {model_name} inference: {len(outputs)} samples.", flush=True)
    return outputs


def build_score_manifest(
    model_outputs: dict[str, list[dict[str, Any]]], model_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    records = []
    for model_name in model_names:
        for output in model_outputs[model_name]:
            records.append(
                {
                    "image_id": f"{output['sample_id']}::{model_name}",
                    "sample_id": output["sample_id"],
                    "variant": model_name,
                    "image_path": output["repaired_image"],
                }
            )
    return records


def score_worker(settings: BenchmarkSettings) -> int:
    import importlib
    import packaging
    import packaging.version
    import torch

    packaging.version = packaging.version
    sys.path.insert(0, str(IQA_REPO))
    pyiqa = importlib.import_module("pyiqa")
    if settings.score_manifest is None or settings.score_progress is None:
        raise ValueError("score_manifest and score_progress are required for command=score-worker")
    manifest = read_jsonl(settings.score_manifest)
    if not settings.resume and settings.score_progress.exists():
        settings.score_progress.unlink()
    completed_rows = read_jsonl(settings.score_progress) if settings.resume else []
    completed = {row["image_id"] for row in completed_rows}
    torch.manual_seed(42)
    metrics = {name: pyiqa.create_metric(name, device="cuda:0").eval() for name in settings.metrics}
    for index, record in enumerate(manifest, start=1):
        if record["image_id"] in completed:
            continue
        raw_scores = {}
        for name, metric in metrics.items():
            with torch.inference_mode():
                score = float(metric(record["image_path"]).reshape(-1)[0].item())
            torch.cuda.synchronize()
            if not math.isfinite(score):
                raise ValueError(f"{name} returned a non-finite value for {record['image_path']}: {score}")
            raw_scores[name] = score
        append_jsonl(settings.score_progress, {**record, "scores": raw_scores})
        if index % 25 == 0 or index == len(manifest):
            print(f"IQA scoring progress: {index}/{len(manifest)}", flush=True)
    return 0


def materialize_repaired_images(
    benchmark_dir: Path, model_outputs: dict[str, list[dict[str, Any]]]
) -> None:
    image_root = benchmark_dir / "images"
    for model_name, outputs in model_outputs.items():
        model_dir = image_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            source = Path(output["final_image"])
            destination = model_dir / f"{output['sample_id']}{source.suffix.lower() or '.png'}"
            if not source.is_file():
                raise FileNotFoundError(f"Repaired image is missing: {source}")
            shutil.copy2(source, destination)
            output["repaired_image"] = str(destination.resolve())


def write_simple_outputs(
    benchmark_dir: Path,
    model_outputs: dict[str, list[dict[str, Any]]],
    score_rows: list[dict[str, Any]],
    model_names: tuple[str, ...],
    metrics: tuple[str, ...],
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    scores_by_id = {row["image_id"]: row["scores"] for row in score_rows}
    trajectory_rows = []
    iqa_rows = []
    summary_rows = []
    for model_name in model_names:
        outputs = model_outputs[model_name]
        metric_values = {metric: [] for metric in metrics}
        durations = []
        restoration_call_counts = []
        for output in outputs:
            image_id = f"{output['sample_id']}::{model_name}"
            scores = scores_by_id[image_id]
            for metric in metrics:
                metric_values[metric].append(float(scores[metric]))
            if output.get("duration_seconds") is not None and math.isfinite(output["duration_seconds"]):
                durations.append(float(output["duration_seconds"]))
            action_path = output.get("action_path", [])
            restoration_call_count = sum(action != "stop" for action in action_path)
            restoration_call_counts.append(restoration_call_count)
            trajectory_rows.append(
                {
                    "model": model_name,
                    "sample_id": output["sample_id"],
                    "image": output["repaired_image"],
                    "tool_calls": output.get("tool_calls", []),
                    "action_path": output.get("action_path", []),
                    "restoration_tool_call_count": restoration_call_count,
                    "duration_seconds": output.get("duration_seconds"),
                    "iqa_scores": scores,
                }
            )
            iqa_rows.append(
                {
                    "model": model_name,
                    "sample_id": output["sample_id"],
                    "image": output["repaired_image"],
                    **{metric: float(scores[metric]) for metric in metrics},
                }
            )
        summary_rows.append(
            {
                "model": model_name,
                "sample_count": len(outputs),
                **{f"{metric}_mean": float(np.mean(values)) for metric, values in metric_values.items()},
                "average_restoration_tool_calls": float(np.mean(restoration_call_counts)),
                "average_duration_seconds": float(np.mean(durations)) if durations else None,
            }
        )
    write_jsonl(benchmark_dir / "trajectories.jsonl", trajectory_rows)
    write_csv(benchmark_dir / "iqa_scores.csv", iqa_rows, ["model", "sample_id", "image", *metrics])
    write_csv(
        benchmark_dir / "summary.csv",
        summary_rows,
        [
            "model",
            "sample_count",
            *(f"{metric}_mean" for metric in metrics),
            "average_restoration_tool_calls",
            "average_duration_seconds",
        ],
    )


def run_iqa_scoring(
    *,
    python: Path,
    benchmark_dir: Path,
    work_dir: Path,
    model_outputs: dict[str, list[dict[str, Any]]],
    model_names: tuple[str, ...],
    metrics: tuple[str, ...],
    gpu: int,
    resume: bool,
) -> None:
    state_dir = work_dir / "iqa"
    score_manifest = state_dir / "iqa_manifest.jsonl"
    score_progress = state_dir / "iqa_scores.jsonl"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_score_manifest(model_outputs, model_names)
    write_jsonl(score_manifest, manifest)
    completed = read_jsonl(score_progress) if resume else []
    expected_ids = {row["image_id"] for row in manifest}
    completed_ids = {row.get("image_id") for row in completed}
    invalid_completed = any("scores" not in row for row in completed)
    if (
        len(completed) != len(manifest)
        or completed_ids != expected_ids
        or len(completed_ids) != len(completed)
        or invalid_completed
    ):
        if completed and (completed_ids - expected_ids or len(completed_ids) != len(completed) or invalid_completed):
            score_progress.unlink(missing_ok=True)
        command = [
            str(python),
            str(SCRIPT_PATH),
            "command=score-worker",
            f"score_manifest={hydra_string(score_manifest.resolve())}",
            f"score_progress={hydra_string(score_progress.resolve())}",
            f"resume={str(resume).lower()}",
        ]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
        print(f"Scoring {len(manifest)} repaired images on physical GPU{gpu}.", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    score_rows = read_jsonl(score_progress)
    if len(score_rows) != len(manifest):
        raise RuntimeError(f"Expected {len(manifest)} IQA rows, found {len(score_rows)}.")
    write_simple_outputs(benchmark_dir, model_outputs, score_rows, model_names, metrics)


def preflight(settings: BenchmarkSettings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_file(settings.base_model / "config.json", "Base model config")
    require_file(settings.tool_config, "3-GPU restoration tool config")
    require_file(settings.config_dir / f"{settings.config_name}.yaml", "Fog v4.1.1 3-GPU config")
    validation_manifest = load_validation_manifest(settings.dataset, settings.expected_dataset_samples)
    adapter_metadata = {
        model_name: validate_adapter(adapter_path, settings.base_model)
        for model_name, adapter_path in settings.model_adapters.items()
    }
    adapter_hashes = [metadata["weights_sha256"] for metadata in adapter_metadata.values()]
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise ValueError("Two or more configured LoRA adapters have byte-identical weights.")
    formal_config = None
    first_model = settings.model_names[0]
    with tempfile.TemporaryDirectory(prefix="fog-lora-benchmark-preflight-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for topology, model_name, max_samples in (
            (settings.formal_topology, first_model, settings.max_samples),
            (settings.smoke_topology, settings.smoke_model, settings.smoke_max_samples),
        ):
            attempt_dir = temporary_root / topology.label
            tool_config_path = create_tool_config(
                settings.tool_config,
                attempt_dir / "tool_config.yaml",
                attempt_dir,
                topology,
            )
            overrides = build_hydra_overrides(
                adapter_path=settings.model_adapters[model_name],
                dataset_path=settings.dataset,
                attempt_dir=attempt_dir,
                tool_config_path=tool_config_path,
                topology=topology,
                max_samples=max_samples,
                sampling=settings.sampling,
                dataset_selection=settings.dataset_selection,
            )
            config = compose_config(settings.config_dir, settings.config_name, overrides)
            validate_composed_config(
                config,
                topology,
                settings.model_adapters[model_name],
                settings.dataset,
                max_samples,
                settings.sampling,
                settings.dataset_selection,
            )
            if topology == settings.formal_topology:
                formal_config = config
    assert formal_config is not None
    metadata = {
        "created_at": utc_now(),
        "dataset": {
            "path": str(settings.dataset),
            "sha256": sha256_file(settings.dataset),
            "source_sample_count": len(validation_manifest),
            "evaluation_sample_count": settings.max_samples,
            "selection": f"parquet_rows_0_through_{settings.max_samples - 1}",
            "shuffle": settings.dataset_selection.shuffle,
            "validation_shuffle": settings.dataset_selection.validation_shuffle,
            "prompt_versions": dict(Counter(row["prompt_version"] for row in validation_manifest)),
        },
        "base_model": str(settings.base_model),
        "adapters": adapter_metadata,
        "source_config": {
            "path": str((settings.config_dir / f"{settings.config_name}.yaml").resolve()),
            "sha256": sha256_file(settings.config_dir / f"{settings.config_name}.yaml"),
            "tool_config_path": str(settings.tool_config),
            "tool_config_sha256": sha256_file(settings.tool_config),
        },
        "sampling": settings.sampling.__dict__,
        "inference": {
            "mode": "validation_only",
            "backend": str(formal_config.actor_rollout_ref.rollout.name),
            "dtype": str(formal_config.actor_rollout_ref.rollout.dtype),
            "tensor_parallel_size": settings.formal_topology.tensor_parallel_size,
            "visible_devices": settings.formal_topology.visible_devices,
            "sampling_gpus": list(settings.formal_sampling_gpus),
            "restoration_and_iqa_device": settings.formal_topology.tool_device,
            "final_iqa_scoring_gpu": settings.scoring_gpu,
            "max_prompt_length": int(formal_config.data.max_prompt_length),
            "max_response_length": int(formal_config.data.max_response_length),
            "max_image_pixels": int(formal_config.data.max_image_pixels),
            "max_user_turns": int(formal_config.actor_rollout_ref.rollout.multi_turn.max_user_turns),
            "max_assistant_turns": int(formal_config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns),
            "max_generated_response_length": int(
                formal_config.actor_rollout_ref.rollout.multi_turn.max_generated_response_length
            ),
            "max_num_seqs": int(formal_config.actor_rollout_ref.rollout.max_num_seqs),
            "enforce_eager": bool(formal_config.actor_rollout_ref.rollout.enforce_eager),
        },
        "iqa_metrics": list(settings.metrics),
        "output_directory": str(settings.benchmark_dir),
        "hydra_config": settings.resolved_config,
    }
    return validation_manifest, metadata


def benchmark_outputs_complete(
    benchmark_dir: Path, expected_samples_per_model: int, model_names: tuple[str, ...]
) -> bool:
    trajectories = read_jsonl(benchmark_dir / "trajectories.jsonl")
    scores = read_csv(benchmark_dir / "iqa_scores.csv")
    summaries = read_csv(benchmark_dir / "summary.csv")
    expected_total = expected_samples_per_model * len(model_names)
    if len(trajectories) != expected_total or len(scores) != expected_total or len(summaries) != len(model_names):
        return False
    expected_pairs = {
        (model_name, f"fog-{index:06d}")
        for model_name in model_names
        for index in range(expected_samples_per_model)
    }
    trajectory_pairs = {(row.get("model"), row.get("sample_id")) for row in trajectories}
    score_pairs = {(row.get("model"), row.get("sample_id")) for row in scores}
    if trajectory_pairs != expected_pairs or score_pairs != expected_pairs:
        return False
    if {row.get("model") for row in summaries} != set(model_names):
        return False
    return all(Path(str(row.get("image", ""))).is_file() for row in trajectories)


def benchmark_outputs_reusable(
    benchmark_dir: Path,
    expected_samples_per_model: int,
    model_names: tuple[str, ...],
    current_metadata: dict[str, Any],
) -> bool:
    parameter_path = benchmark_dir / "inference_parameters.json"
    if not parameter_path.is_file():
        return False
    try:
        existing = json.loads(parameter_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if existing.get("status") != "complete":
        return False

    existing_dataset = existing.get("dataset", {})
    current_dataset = current_metadata["dataset"]
    dataset_contract_keys = (
        "path",
        "sha256",
        "evaluation_sample_count",
        "selection",
        "shuffle",
        "validation_shuffle",
    )
    if any(existing_dataset.get(key) != current_dataset.get(key) for key in dataset_contract_keys):
        return False
    for model_name in model_names:
        existing_adapter = existing.get("adapters", {}).get(model_name, {})
        current_adapter = current_metadata["adapters"][model_name]
        if existing_adapter.get("path") != current_adapter.get("path"):
            return False
        if existing_adapter.get("weights_sha256") != current_adapter.get("weights_sha256"):
            return False
    return benchmark_outputs_complete(benchmark_dir, expected_samples_per_model, model_names)


def write_inference_parameters(path: Path, metadata: dict[str, Any], **status: Any) -> None:
    write_json(path, {**metadata, **status})


def migrate_legacy_results(settings: BenchmarkSettings) -> int:
    legacy_results = settings.benchmark_dir / "results"
    legacy_trajectories = legacy_results / "trajectories.jsonl"
    legacy_scores = legacy_results / "iqa_scores.jsonl"
    legacy_images = legacy_results / "images"
    require_file(legacy_trajectories, "Legacy trajectory output")
    require_file(legacy_scores, "Legacy IQA output")
    if not legacy_images.is_dir():
        raise FileNotFoundError(f"Legacy repaired images are missing: {legacy_images}")

    destination_images = settings.benchmark_dir / "images"
    if destination_images.exists():
        raise FileExistsError(f"Destination image directory already exists: {destination_images}")
    shutil.move(str(legacy_images), str(destination_images))

    trajectory_rows = read_jsonl(legacy_trajectories)
    score_rows = read_jsonl(legacy_scores)
    model_outputs: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in settings.model_names}
    for row in trajectory_rows:
        model_name = str(row["model"])
        sample_id = str(row["sample_id"])
        candidates = list((destination_images / model_name).glob(f"{sample_id}.*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one repaired image for {model_name}/{sample_id}, found {len(candidates)}")
        model_outputs[model_name].append(
            {
                "model": model_name,
                "sample_id": sample_id,
                "repaired_image": str(candidates[0].resolve()),
                "tool_calls": row.get("tool_calls", []),
                "action_path": row.get("action_path", []),
                "duration_seconds": row.get("duration_seconds"),
            }
        )
    for model_name in settings.model_names:
        model_outputs[model_name].sort(key=lambda row: row["sample_id"])
    for row in score_rows:
        model_name = str(row["variant"])
        sample_id = str(row["sample_id"])
        image = next((destination_images / model_name).glob(f"{sample_id}.*"))
        row["image_path"] = str(image.resolve())

    write_simple_outputs(
        settings.benchmark_dir,
        model_outputs,
        score_rows,
        settings.model_names,
        settings.metrics,
    )
    validation_manifest, metadata = preflight(settings)
    write_inference_parameters(
        settings.benchmark_dir / "inference_parameters.json",
        metadata,
        status="complete",
        migrated_from_legacy_layout=True,
        migrated_at=utc_now(),
    )
    if not benchmark_outputs_complete(settings.benchmark_dir, settings.max_samples, settings.model_names):
        raise RuntimeError("Migrated benchmark output validation failed; legacy directories were retained.")

    for obsolete in (legacy_results, settings.benchmark_dir / "inference", settings.benchmark_dir / ".state"):
        if obsolete.exists():
            shutil.rmtree(obsolete)
    print(f"Migrated benchmark outputs: {settings.benchmark_dir}", flush=True)
    return 0


def run_formal(settings: BenchmarkSettings) -> int:
    # Formal evaluation intentionally starts without checking GPU occupancy.
    # The caller owns the GPU topology decision and may co-locate this run with
    # an existing process on GPU1.
    validation_manifest, metadata = preflight(settings)
    selected_manifest = validation_manifest[: settings.max_samples]
    settings.benchmark_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = settings.benchmark_dir / "inference_parameters.json"
    if settings.resume and benchmark_outputs_reusable(
        settings.benchmark_dir,
        settings.max_samples,
        settings.model_names,
        metadata,
    ):
        write_inference_parameters(parameter_path, metadata, status="complete", reused_existing_results=True)
        print(f"Benchmark already complete: {settings.benchmark_dir}", flush=True)
        return 0
    write_inference_parameters(parameter_path, metadata, status="running", started_at=utc_now())

    model_outputs = {}
    for model_name, adapter_path in settings.model_adapters.items():
        model_outputs[model_name] = run_one_model(
            model_name=model_name,
            adapter_path=adapter_path,
            validation_manifest=selected_manifest,
            dataset_path=settings.dataset,
            work_dir=settings.work_dir,
            config_dir=settings.config_dir,
            config_name=settings.config_name,
            tool_template=settings.tool_config,
            python=settings.python,
            topology=settings.formal_topology,
            max_samples=settings.max_samples,
            sampling=settings.sampling,
            dataset_selection=settings.dataset_selection,
            resume=settings.resume,
        )
    materialize_repaired_images(settings.benchmark_dir, model_outputs)
    run_iqa_scoring(
        python=settings.python,
        benchmark_dir=settings.benchmark_dir,
        work_dir=settings.work_dir,
        model_outputs=model_outputs,
        model_names=settings.model_names,
        metrics=settings.metrics,
        gpu=settings.scoring_gpu,
        resume=settings.resume,
    )
    if not benchmark_outputs_complete(settings.benchmark_dir, settings.max_samples, settings.model_names):
        raise RuntimeError(f"Final benchmark output validation failed: {settings.benchmark_dir}")
    write_inference_parameters(parameter_path, metadata, status="complete", completed_at=utc_now())
    if settings.work_dir.is_dir():
        shutil.rmtree(settings.work_dir)
    print(f"Benchmark complete: {settings.benchmark_dir}", flush=True)
    return 0


def run_smoke(settings: BenchmarkSettings) -> int:
    smoke_gpu_indices = tuple(int(value) for value in settings.smoke_topology.visible_devices.split(","))
    require_gpus_idle(smoke_gpu_indices)
    validation_manifest, _metadata = preflight(settings)
    smoke_dir = settings.work_dir / "development_smoke"
    outputs = run_one_model(
        model_name=settings.smoke_model,
        adapter_path=settings.model_adapters[settings.smoke_model],
        validation_manifest=validation_manifest,
        dataset_path=settings.dataset,
        work_dir=smoke_dir,
        config_dir=settings.config_dir,
        config_name=settings.config_name,
        tool_template=settings.tool_config,
        python=settings.python,
        topology=settings.smoke_topology,
        max_samples=settings.smoke_max_samples,
        sampling=settings.sampling,
        dataset_selection=settings.dataset_selection,
        resume=settings.resume,
    )
    print(f"GPU3 smoke completed: {outputs[0]['final_image']}", flush=True)
    return 0


@hydra.main(version_base=None, config_path="../../config/eval", config_name="fog_lora_benchmark")
def main(config: DictConfig) -> int:
    settings = settings_from_config(config)
    if settings.command == "score-worker":
        return score_worker(settings)
    if settings.command == "preflight":
        validation_manifest, metadata = preflight(settings)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "validation_samples": len(validation_manifest),
                    "adapter_hashes": {
                        model: item["weights_sha256"] for model, item in metadata["adapters"].items()
                    },
                    "evaluation_samples": settings.max_samples,
                    "formal_topology": settings.formal_topology.__dict__,
                },
                indent=2,
            )
        )
        return 0
    if settings.command == "migrate-results":
        return migrate_legacy_results(settings)
    if settings.command == "smoke":
        return run_smoke(settings)
    if settings.command != "run":
        raise ValueError(f"Unsupported command: {settings.command}")
    return run_formal(settings)


if __name__ == "__main__":
    raise SystemExit(main())
