# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image Restoration Tool for verl multi-turn rollout.

This tool wraps the RestorationToolkit located in restoration_tools/agent_tools/.
It self-contains IQA-based reward computation (no separate interactions layer
is required), and is designed for use with the standard verl ToolAgentLoop and
the hermes tool-call format.

Key design decisions:
- IQA scoring (QAlign / MANIQA / MUSIQ / CLIPIQA / NIQE) is done inside execute()
  and the reward is returned directly as tool_reward_score.
- Termination is controlled by YAML config (max_user_turns / max_assistant_turns)
  rather than a hard-stop mechanism inside the tool.
- The tool returns a feedback text message after each step so the model can
  reason about its next action.

Supported restoration actions:
- real_esrgan: Super-resolution / deblurring / denoising / compression artifact removal
- scunet: High-quality denoising
- retinexformer_fivek: Low-light enhancement
- hvicidnet: Low-light / exposure correction
- lightdiff: Low-light enhancement (diffusion model)
- turbo_rain: Fast deraining
- s2former: Rain streak removal
- idt: Deraining / raindrop removal
- ridcp: Dehazing
- kanet: Dehazing
- turbo_snow: Desnowing
- snowmaster: Advanced desnowing
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Optional
from uuid import uuid4

import torch
import yaml
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Keep runtime diagnostics on the console according to VERL_LOGGING_LEVEL.
logger.propagate = False
_console_level = os.getenv("VERL_LOGGING_LEVEL", "WARN").upper()
logger.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(getattr(logging, _console_level, logging.WARNING))
_console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_console_handler)

# Write only completed trajectory summaries to restoration_tool_info.log. Each
# message is one JSON object on one line, so concurrent trajectories do not
# interleave their individual tool-call diagnostics.
_log_dir = os.getenv("VERL_LOG_DIR", "/tmp")
os.makedirs(_log_dir, exist_ok=True)
trajectory_logger = logging.getLogger(f"{__name__}.trajectory")
trajectory_logger.propagate = False
trajectory_logger.setLevel(logging.INFO)
_trajectory_file_handler = logging.FileHandler(
    os.path.join(_log_dir, "restoration_tool_info.log"),
    mode="a",
    encoding="utf-8",
)
_trajectory_file_handler.setLevel(logging.INFO)
_trajectory_file_handler.setFormatter(logging.Formatter("%(message)s"))
trajectory_logger.addHandler(_trajectory_file_handler)

# Add restoration_tools/agent_tools to sys.path for importing RestorationToolkit
AGENT_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent / "restoration_tools" / "agent_tools"
if AGENT_TOOLS_PATH.exists() and str(AGENT_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_PATH))

# Add submodule paths (Retinexformer, SCUNet, etc.)
_RESTORATION_TOOLS_PATH = AGENT_TOOLS_PATH.parent
_submodules = [
    "Retinexformer",
    "HVICIDNet",
    "LightenDiffusion",
    "SCUNet",
    "ESRGAN",
    "IDT",
    "RIDCP",
    "KANet",
    "S2Former",
    "SnowMaster",
    "img2img_turbo",
]
for _submod in _submodules:
    _submod_path = _RESTORATION_TOOLS_PATH / _submod
    if _submod_path.exists() and str(_submod_path) not in sys.path:
        sys.path.insert(0, str(_submod_path))

# Allowed restoration actions (including stop)
ALLOWED_ACTIONS = {
    "real_esrgan",
    "scunet",
    "retinexformer_fivek",
    "hvicidnet",
    "lightdiff",
    "turbo_rain",
    "s2former",
    "idt",
    "ridcp",
    "kanet",
    "turbo_snow",
    "snowmaster",
    "nafnet_denoise",
    "focalnet_dehaze",
    "focalnet_desnow",
    "mb_taylorformer_dehaze",
    "stop",
}

# Degradation type → recommended action affinity map.
# Values in [0, 1]: 1.0 = primary recommendation, 0.8 = strong, 0.5 = moderate.
# When affinity_bonus_scale > 0, choosing a high-affinity action yields a bonus
# on top of the IQA-based step reward, guiding the model to learn the mapping
# between degradation types and their specialized restoration tools.
DEGRADATION_ACTION_AFFINITY: dict[str, dict[str, float]] = {
    "night": {"retinexformer_fivek": 1.0, "hvicidnet": 1.0, "lightdiff": 0.8},
    "low_light": {"retinexformer_fivek": 1.0, "hvicidnet": 1.0, "lightdiff": 0.8},
    "rain": {"s2former": 1.0, "turbo_rain": 1.0, "idt": 0.8},
    "rain_streak": {"s2former": 1.0, "turbo_rain": 1.0, "idt": 0.8},
    "rain_drop": {"idt": 1.0, "turbo_rain": 0.8, "s2former": 0.6},
    "rain_drive": {"turbo_rain": 1.0, "idt": 0.8, "s2former": 0.6},
    "fog": {"ridcp": 1.0, "kanet": 1.0},
    "snow": {"turbo_snow": 1.0, "snowmaster": 1.0},
}

# Action groups that share the repeat penalty within a trajectory. Derain and
# generic restoration tools intentionally fall back to their individual actions.
TOOL_REPEAT_TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "dehaze": ("ridcp", "kanet", "focalnet_dehaze", "mb_taylorformer_dehaze"),
    "desnow": ("focalnet_desnow", "s2former", "turbo_snow"),
    "low_light": ("retinexformer_fivek", "hvicidnet", "lightdiff"),
}
TOOL_REPEAT_TYPE_BY_ACTION: dict[str, str] = {
    action: group_name for group_name, actions in TOOL_REPEAT_TYPE_GROUPS.items() for action in actions
}

# Legacy IQA metric weights per degradation type (QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE).
# Prefer loading a data-driven map from local training data via ``iqa_weight_map_path``.
SCORE_WEIGHT_MAP: dict[str, list[float]] = {
    "night": [2.0 / 9, 2.0 / 9, 0.0, 2.0 / 9, 3.0 / 9],
    "low_light": [2.0 / 9, 2.0 / 9, 0.0, 2.0 / 9, 3.0 / 9],
    "rain": [1.0 / 5, 1.25 / 5, 1.0 / 5, 0.75 / 5, 1.0 / 5],
    "rain_streak": [1.0 / 5, 1.25 / 5, 1.0 / 5, 0.75 / 5, 1.0 / 5],
    "rain_drop": [0.0, 0.5 / 3, 0.0, 1.25 / 3, 1.25 / 3],
    "rain_drive": [0.5 / 4, 1.5 / 4, 1.0 / 4, 1.0 / 4, 0.0],
    "snow": [1.5 / 5, 0.75 / 5, 1.0 / 5, 0.75 / 5, 1.0 / 5],
    "fog": [1.5 / 5, 0.5 / 5, 1.5 / 5, 0.5 / 5, 1.0 / 5],
}
DEFAULT_WEIGHT: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2]
FAILURE_REWARD = -5.0
INVALID_ACTION_REWARD = -10.0
REPEAT_ACTION_PENALTY = 0.2
REPEAT_LOW_GAIN_PENALTY = 0.8
REPEAT_LOW_GAIN_THRESHOLD = 0.05
TOOL_CALL_COST = 0.12
STOP_MIN_STEP = 3
STOP_IQA_DELTA_THRESHOLD = 0.25
STOP_EARLY_PENALTY = -1.0
STOP_RECENT_REWARD_WINDOW = 2
STOP_RECENT_REWARD_THRESHOLD = 0.25
REWARD_MODE_STEP_MIXED_V1 = "step_mixed_v1"
REWARD_MODE_FINAL_IQA_V2 = "final_iqa_v2"
REWARD_MODE_MARGINAL_EFFICIENCY_V1 = "marginal_efficiency_v1"
SUPPORTED_REWARD_MODES = {
    REWARD_MODE_STEP_MIXED_V1,
    REWARD_MODE_FINAL_IQA_V2,
    REWARD_MODE_MARGINAL_EFFICIENCY_V1,
}

# Module-level caches
_toolkit_instance = None
_runtime_pool_instance = None
_runtime_pool_config_key = None
_iqa_instances: dict[tuple[str, bool, str | None, str | None, str | None], Any] = {}


def _uniform_score_weight(size: int) -> list[float]:
    if size <= 0:
        raise ValueError("IQA weight vector must contain at least one value")
    return [1.0 / size] * size


def _normalize_score_weight_vector(weights: list[float], fallback: list[float] | None = None) -> list[float]:
    if not weights:
        raise ValueError("IQA weight vector must contain at least one value")
    tensor = torch.tensor(weights, dtype=torch.float32)
    tensor = torch.clamp(tensor, min=0.0)
    total = float(tensor.sum().item())
    if total <= 0.0:
        if fallback is not None and len(fallback) == len(weights):
            return list(fallback)
        return _uniform_score_weight(len(weights))
    return [float(value) for value in (tensor / total).tolist()]


def _load_score_weight_map(weight_map_path: str) -> dict[str, list[float]]:
    with open(weight_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics_payload = payload.get("metrics")
    if isinstance(metrics_payload, dict) and metrics_payload:
        weights = []
        for metric_name, metric_config in metrics_payload.items():
            if not isinstance(metric_config, dict) or "weight" not in metric_config:
                raise ValueError(f"Metric '{metric_name}' in {weight_map_path} is missing a weight")
            weights.append(float(metric_config["weight"]))
        default_weight = _normalize_score_weight_vector(weights)
        aliases = [
            "__default__",
            "default",
            "night",
            "low_light",
            "rain",
            "rain_streak",
            "rain_drop",
            "rain_drive",
            "snow",
            "fog",
        ]
        return {alias: list(default_weight) for alias in aliases}

    weights_payload = payload.get("weights", payload)
    if not isinstance(weights_payload, dict):
        raise ValueError(f"Invalid weight map payload in {weight_map_path}")

    normalized_map = {}
    for degradation_type, weights in weights_payload.items():
        if not isinstance(weights, list):
            raise ValueError(f"Weight entry for degradation '{degradation_type}' must be a list, got {type(weights)}")
        normalized_map[degradation_type] = _normalize_score_weight_vector(weights)
    return normalized_map


def _resolve_path(value: str | None, *, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _load_tool_runtime_config(
    tool_registry_path: str | None,
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, str]]:
    """Load canonical runtimes and model-facing action mappings from tools.yaml."""

    if not tool_registry_path:
        actions = set(ALLOWED_ACTIONS) | {"stop"}
        return actions, {}, {action: action for action in actions}

    registry_path = Path(tool_registry_path).expanduser().resolve()
    if not registry_path.is_file():
        logger.warning("Tool registry path does not exist: %s", registry_path)
        actions = set(ALLOWED_ACTIONS) | {"stop"}
        return actions, {}, {action: action for action in actions}

    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"tool registry must be a mapping: {registry_path}")

    actions = {"stop"}
    model_to_runtime = {"stop": "stop"}
    candidate_runtimes: dict[str, dict[str, Any]] = {}
    for item in payload.get("tools", []):
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        model_name = str(item.get("model_name", name)).strip()
        if not model_name:
            raise ValueError(f"tool {name!r} has an empty model_name")
        if model_name in model_to_runtime:
            raise ValueError(f"duplicate model-facing restoration action: {model_name}")
        actions.add(name)
        model_to_runtime[model_name] = name
        runtime = item.get("runtime")
        if isinstance(runtime, dict) and runtime.get("adapter") == "candidate":
            candidate_runtimes[name] = dict(runtime)
    return actions, candidate_runtimes, model_to_runtime


def get_toolkit(
    device: str = "cuda",
    models: list = None,
    preload: bool = True,
    auto_unload: bool = False,
    model_devices: list[str] | None = None,
    model_device_map: dict[str, str] | None = None,
):
    """Lazy load and cache the RestorationToolkit instance."""
    global _toolkit_instance
    if _toolkit_instance is None:
        try:
            from restoration_tools.agent_tools import RestorationToolkit

            _toolkit_instance = RestorationToolkit(
                models=models,
                device=device,
                load_iqa=False,
                preload=preload,
                auto_unload=auto_unload,
                model_devices=model_devices,
                model_device_map=model_device_map,
            )
            logger.info(
                f"RestorationToolkit initialized on {device} " f"(preload={preload}, auto_unload={auto_unload})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RestorationToolkit: {e}")
            raise
    return _toolkit_instance


def get_iqa_scorer(
    device: str = "cuda",
    normalize_scores: bool = False,
    normalization_stats_path: str | None = None,
    qalign_path: str | None = None,
    metric_config_path: str | None = None,
):
    """Lazy load and cache the IQAScore instance."""
    global _iqa_instances
    cache_key = (str(device), normalize_scores, normalization_stats_path, qalign_path, metric_config_path)
    if cache_key not in _iqa_instances:
        try:
            from iqa_reward import IQAScore

            _iqa_instances[cache_key] = IQAScore(
                device=device,
                qalign_path=qalign_path,
                normalize_scores=normalize_scores,
                normalization_stats_path=normalization_stats_path,
                metric_config_path=metric_config_path,
            )
            logger.info(
                f"IQAScore initialized on {device} "
                f"(normalize_scores={normalize_scores}, stats_path={normalization_stats_path}, "
                f"metric_config_path={metric_config_path})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize IQAScore: {e}")
            raise
    return _iqa_instances[cache_key]


def _as_device_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass
class _CachedToolResult:
    result: dict[str, Any]
    iqa_scores: list[float]
    cache_key: str


class _ToolResultDiskCache:
    """Bounded disk cache for deterministic restoration tool outputs."""

    def __init__(
        self,
        *,
        enabled: bool,
        cache_dir: str,
        signature: dict[str, Any],
        max_entries: int,
        max_size_bytes: int,
        ttl_seconds: float,
        cleanup_interval: int,
    ):
        self.enabled = bool(enabled) and max_entries > 0 and max_size_bytes > 0
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.signature = signature
        self.max_entries = int(max_entries)
        self.max_size_bytes = int(max_size_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self.cleanup_interval = max(1, int(cleanup_interval))
        self._ops_since_cleanup = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup(force=True)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _ToolResultDiskCache._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_ToolResultDiskCache._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _hash_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, path)

    def _entry_dir(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key[:2] / cache_key

    def _build_key(self, action: str, input_path: str) -> tuple[str, str]:
        input_hash = self._hash_file(input_path)
        payload = {
            "version": 1,
            "action": action,
            "input_sha256": input_hash,
            "signature": self.signature,
        }
        key_material = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(key_material).hexdigest(), input_hash

    def _remove_entry(self, entry_dir: Path) -> None:
        try:
            shutil.rmtree(entry_dir, ignore_errors=True)
            parent = entry_dir.parent
            if parent != self.cache_dir and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception as e:
            logger.debug("Failed to remove restoration tool cache entry %s: %s", entry_dir, e)

    def get(self, action: str, input_path: str, output_dir: str) -> _CachedToolResult | None:
        if not self.enabled:
            return None
        try:
            cache_key, _ = self._build_key(action, input_path)
            entry_dir = self._entry_dir(cache_key)
            metadata_path = entry_dir / "metadata.json"
            image_path = entry_dir / "image.png"
            if not metadata_path.is_file() or not image_path.is_file():
                return None

            with open(metadata_path, encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
            now = time.time()
            if self.ttl_seconds > 0 and now - float(metadata.get("created_at", now)) > self.ttl_seconds:
                self._remove_entry(entry_dir)
                return None

            output_path = Path(output_dir) / f"{Path(input_path).stem}_{action}_{uuid4().hex[:8]}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_path)

            metadata["last_access"] = now
            metadata["hits"] = int(metadata.get("hits", 0)) + 1
            self._write_json_atomic(metadata_path, metadata)
            self._ops_since_cleanup += 1
            self.cleanup()

            result = dict(metadata.get("result") or {})
            result["output_path"] = str(output_path)
            result_metadata = dict(result.get("metadata") or {})
            result_metadata["tool_result_cache"] = {"hit": True, "key": cache_key}
            result["metadata"] = result_metadata
            result["cache_hit"] = True
            return _CachedToolResult(
                result=result,
                iqa_scores=[float(score) for score in metadata.get("iqa_scores", [])],
                cache_key=cache_key,
            )
        except Exception as e:
            logger.warning("Restoration tool result cache lookup failed for %s on %s: %s", action, input_path, e)
            return None

    def put(self, action: str, input_path: str, result: dict[str, Any], iqa_scores: list[float]) -> str | None:
        if not self.enabled:
            return None
        output_path = result.get("output_path")
        if not output_path or not os.path.isfile(output_path):
            return None
        try:
            cache_key, input_hash = self._build_key(action, input_path)
            entry_dir = self._entry_dir(cache_key)
            entry_dir.mkdir(parents=True, exist_ok=True)

            image_path = entry_dir / "image.png"
            tmp_image_path = entry_dir / f"image.png.{os.getpid()}.{uuid4().hex}.tmp"
            shutil.copy2(output_path, tmp_image_path)
            os.replace(tmp_image_path, image_path)

            cached_result = {
                key: self._json_safe(value) for key, value in result.items() if key not in {"output_path", "cache_hit"}
            }
            metadata = {
                "version": 1,
                "cache_key": cache_key,
                "action": action,
                "input_sha256": input_hash,
                "created_at": time.time(),
                "last_access": time.time(),
                "hits": 0,
                "signature": self.signature,
                "iqa_scores": [float(score) for score in iqa_scores],
                "result": cached_result,
            }
            self._write_json_atomic(entry_dir / "metadata.json", metadata)
            self._ops_since_cleanup += 1
            self.cleanup()
            return cache_key
        except Exception as e:
            logger.warning("Restoration tool result cache write failed for %s on %s: %s", action, input_path, e)
            return None

    def cleanup(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and self._ops_since_cleanup < self.cleanup_interval:
            return
        self._ops_since_cleanup = 0

        now = time.time()
        entries: list[tuple[float, int, Path]] = []
        total_size = 0
        try:
            shard_dirs = [path for path in self.cache_dir.iterdir() if path.is_dir()]
        except FileNotFoundError:
            return

        for shard_dir in shard_dirs:
            for entry_dir in list(shard_dir.iterdir()):
                if not entry_dir.is_dir():
                    continue
                metadata_path = entry_dir / "metadata.json"
                image_path = entry_dir / "image.png"
                if not metadata_path.is_file() or not image_path.is_file():
                    self._remove_entry(entry_dir)
                    continue
                try:
                    with open(metadata_path, encoding="utf-8") as file_obj:
                        metadata = json.load(file_obj)
                    created_at = float(metadata.get("created_at", 0.0))
                    last_access = float(metadata.get("last_access", created_at))
                    if self.ttl_seconds > 0 and now - created_at > self.ttl_seconds:
                        self._remove_entry(entry_dir)
                        continue
                    size_bytes = metadata_path.stat().st_size + image_path.stat().st_size
                except Exception:
                    self._remove_entry(entry_dir)
                    continue
                total_size += size_bytes
                entries.append((last_access, size_bytes, entry_dir))

        if len(entries) <= self.max_entries and total_size <= self.max_size_bytes:
            return

        entries.sort(key=lambda item: item[0])
        while entries and (len(entries) > self.max_entries or total_size > self.max_size_bytes):
            _, size_bytes, entry_dir = entries.pop(0)
            self._remove_entry(entry_dir)
            total_size -= size_bytes


@dataclass
class _RestorationRuntimeWorker:
    index: int
    device: str
    iqa_device: str
    toolkit: Any
    iqa: Any | None


class _RestorationRuntimePool:
    """One full restoration+IQA runtime replica per configured GPU."""

    def __init__(
        self,
        *,
        worker_devices: list[str],
        iqa_devices: list[str],
        models: list[str] | None,
        preload: bool,
        auto_unload: bool,
        use_iqa: bool,
        normalize_iqa_scores: bool,
        iqa_stats_path: str | None,
        iqa_qalign_path: str | None,
        iqa_metric_config_path: str | None,
    ):
        if not worker_devices:
            raise ValueError("worker_devices must contain at least one device")

        from restoration_tools.agent_tools import RestorationToolkit

        self.workers: list[_RestorationRuntimeWorker] = []
        self._available_worker_indices: Queue[int] = Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=len(worker_devices),
            thread_name_prefix="restoration-runtime",
        )

        resolved_iqa_devices = iqa_devices or worker_devices
        for index, device in enumerate(worker_devices):
            iqa_device = resolved_iqa_devices[index % len(resolved_iqa_devices)]
            toolkit = RestorationToolkit(
                models=models,
                device=device,
                load_iqa=False,
                preload=False,
                auto_unload=auto_unload,
                model_devices=[device],
                model_device_map=None,
            )
            iqa = None
            if use_iqa:
                iqa = get_iqa_scorer(
                    device=iqa_device,
                    normalize_scores=normalize_iqa_scores,
                    normalization_stats_path=iqa_stats_path,
                    qalign_path=iqa_qalign_path,
                    metric_config_path=iqa_metric_config_path,
                )
            worker = _RestorationRuntimeWorker(
                index=index,
                device=device,
                iqa_device=iqa_device,
                toolkit=toolkit,
                iqa=iqa,
            )
            self.workers.append(worker)
            self._available_worker_indices.put(index)

        if preload:
            self.preload_models()

        logger.info(
            "Restoration runtime pool initialized with workers: %s",
            [
                {"index": worker.index, "device": worker.device, "iqa_device": worker.iqa_device}
                for worker in self.workers
            ],
        )

    def _run_with_worker(self, fn):
        worker_index = self._available_worker_indices.get()
        worker = self.workers[worker_index]
        try:
            logger.info(
                "Dispatching restoration request to worker %s (tool=%s, iqa=%s)",
                worker.index,
                worker.device,
                worker.iqa_device,
            )
            return fn(worker)
        finally:
            self._available_worker_indices.put(worker_index)

    async def run(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_with_worker, fn)

    def preload_models(self) -> None:
        # Model construction touches process-global torch/diffusers/import state.
        # Keep preload serialized; after models are resident, execute() still
        # dispatches inference concurrently across runtime workers.
        for worker in self.workers:
            logger.info(
                "Preloading restoration models on worker %s (%s)",
                worker.index,
                worker.device,
            )
            worker.toolkit.load_models()
        logger.info("Preloaded restoration models on %d runtime workers", len(self.workers))

    def unload_all_models(self) -> None:
        futures = [self._executor.submit(worker.toolkit.unload_all_models) for worker in self.workers]
        for future in futures:
            future.result()
        logger.info("Unloaded restoration models on %d runtime workers", len(self.workers))


def get_runtime_pool(
    *,
    worker_devices: list[str],
    iqa_devices: list[str],
    models: list[str] | None,
    preload: bool,
    auto_unload: bool,
    use_iqa: bool,
    normalize_iqa_scores: bool,
    iqa_stats_path: str | None,
    iqa_qalign_path: str | None,
    iqa_metric_config_path: str | None,
) -> _RestorationRuntimePool:
    """Lazy load and cache the multi-GPU restoration runtime pool."""
    global _runtime_pool_instance, _runtime_pool_config_key

    models_key = tuple(models or [])
    config_key = (
        tuple(worker_devices),
        tuple(iqa_devices),
        models_key,
        bool(auto_unload),
        bool(use_iqa),
        bool(normalize_iqa_scores),
        iqa_stats_path,
        iqa_qalign_path,
        iqa_metric_config_path,
    )
    if _runtime_pool_instance is None or _runtime_pool_config_key != config_key:
        if _runtime_pool_instance is not None:
            try:
                _runtime_pool_instance.unload_all_models()
            except Exception as e:
                logger.warning("Failed to unload previous restoration runtime pool: %s", e)
        _runtime_pool_instance = _RestorationRuntimePool(
            worker_devices=worker_devices,
            iqa_devices=iqa_devices,
            models=models,
            preload=preload,
            auto_unload=auto_unload,
            use_iqa=use_iqa,
            normalize_iqa_scores=normalize_iqa_scores,
            iqa_stats_path=iqa_stats_path,
            iqa_qalign_path=iqa_qalign_path,
            iqa_metric_config_path=iqa_metric_config_path,
        )
        _runtime_pool_config_key = config_key
    return _runtime_pool_instance


def _load_restoration_tool_runtime_config(tool_config_path: str) -> dict[str, Any] | None:
    """Load runtime config for RestorationTool from tool_config yaml."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(tool_config_path)
        for tool_item in cfg.get("tools", []):
            if tool_item.get("class_name") == "verl.tools.restoration_tool.RestorationTool":
                return dict(tool_item.get("config", {}))
    except Exception as e:
        logger.warning(f"Failed to load tool config from {tool_config_path}: {e}")
    return None


def preload_restoration_models_for_sampling(tool_config_path: str) -> bool:
    """Preload all restoration models at sampling stage start.

    Returns True if preload path was executed (or models already loaded), False otherwise.
    """
    runtime_cfg = _load_restoration_tool_runtime_config(tool_config_path)
    if runtime_cfg is None:
        return False

    # Phase-managed mode: keep models resident during rollout; unload as a batch afterwards.
    device = runtime_cfg.get("device", "cuda")
    models = runtime_cfg.get("models", None)
    worker_devices = _as_device_list(runtime_cfg.get("worker_devices", None))
    iqa_devices = _as_device_list(runtime_cfg.get("iqa_devices", None))
    project_root = Path(__file__).resolve().parent.parent.parent
    iqa_stats_path = runtime_cfg.get("iqa_stats_path", None)
    iqa_qalign_path = runtime_cfg.get("iqa_qalign_path", None)
    iqa_metric_config_path = runtime_cfg.get("iqa_metric_config_path", None)
    if iqa_stats_path and not os.path.isabs(iqa_stats_path):
        iqa_stats_path = str((project_root / iqa_stats_path).resolve())
    if iqa_qalign_path and not os.path.isabs(iqa_qalign_path):
        iqa_qalign_path = str((project_root / iqa_qalign_path).resolve())
    if iqa_metric_config_path and not os.path.isabs(iqa_metric_config_path):
        iqa_metric_config_path = str((project_root / iqa_metric_config_path).resolve())
    if worker_devices:
        pool = get_runtime_pool(
            worker_devices=worker_devices,
            iqa_devices=iqa_devices,
            models=models,
            preload=False,
            auto_unload=False,
            use_iqa=bool(runtime_cfg.get("use_iqa", True)),
            normalize_iqa_scores=bool(runtime_cfg.get("normalize_iqa_scores", False)),
            iqa_stats_path=iqa_stats_path,
            iqa_qalign_path=iqa_qalign_path,
            iqa_metric_config_path=iqa_metric_config_path,
        )
        pool.preload_models()
        logger.info("Preloaded replicated restoration runtime pool for sampling stage")
        return True

    model_devices = runtime_cfg.get("model_devices", None)
    model_device_map = runtime_cfg.get("model_device_map", None)
    toolkit = get_toolkit(
        device=device,
        models=models,
        preload=False,
        auto_unload=False,
        model_devices=model_devices,
        model_device_map=model_device_map,
    )
    toolkit.auto_unload = False
    toolkit.load_models()
    logger.info("Preloaded all restoration models for sampling stage")
    return True


def keep_restoration_models_loaded_between_sampling_steps(tool_config_path: str) -> bool:
    """Return whether restoration models should remain resident between GRPO steps."""
    runtime_cfg = _load_restoration_tool_runtime_config(tool_config_path)
    return bool(runtime_cfg and runtime_cfg.get("keep_models_loaded_between_sampling_steps", False))


def unload_restoration_models_after_sampling() -> bool:
    """Unload all restoration models at sampling stage end."""
    unloaded = False
    global _toolkit_instance, _runtime_pool_instance
    if _toolkit_instance is not None:
        _toolkit_instance.unload_all_models()
        logger.info("Unloaded all restoration models after sampling stage")
        unloaded = True
    if _runtime_pool_instance is not None:
        _runtime_pool_instance.unload_all_models()
        logger.info("Unloaded replicated restoration runtime pool after sampling stage")
        unloaded = True
    return unloaded


class RestorationTool(BaseTool):
    """A tool for iterative image restoration / degradation removal.

    Computes IQA-based step rewards inside ``execute()`` and returns them
    directly to the verl ToolAgentLoop as ``tool_reward_score``.
    No external interactions layer is required.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        self.device = config.get("device", "cuda")
        self.iqa_device = config.get("iqa_device", self.device)
        self.worker_devices = _as_device_list(config.get("worker_devices", None))
        self.iqa_devices = _as_device_list(config.get("iqa_devices", None))
        self.use_parallel_workers = bool(self.worker_devices)
        self.preload_models = config.get("models", None)
        self.model_devices = config.get("model_devices", [self.device])
        self.model_device_map = config.get("model_device_map", None)
        self.output_dir = config.get("output_dir", "/tmp/verl_restoration")
        self.preload = config.get("preload", True)
        # Disable per-tool-call auto-unload mode. We use phase-managed load/unload:
        # preload once at sampling start, unload once after sampling.
        configured_auto_unload = bool(config.get("auto_unload", False))
        if configured_auto_unload:
            logger.warning("auto_unload=true is ignored in phase-managed mode; forcing auto_unload=false")
        self.auto_unload = False
        self.use_iqa = config.get("use_iqa", True)
        self.normalize_iqa_scores = bool(config.get("normalize_iqa_scores", False))
        self.iqa_stats_path = config.get("iqa_stats_path", None)
        self.iqa_qalign_path = config.get("iqa_qalign_path", None)
        self.iqa_metric_config_path = config.get("iqa_metric_config_path", None)
        self.iqa_weight_map_path = config.get("iqa_weight_map_path", None)
        self.reward_mode = str(config.get("reward_mode", REWARD_MODE_STEP_MIXED_V1))
        if self.reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode!r}; " f"expected one of {sorted(SUPPORTED_REWARD_MODES)}"
            )
        self.alpha = float(config.get("alpha", 0.9))  # marginal-improvement weight
        self.beta = 1.0 - self.alpha  # identity-improvement weight
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.tool_call_cost = float(config.get("tool_call_cost", TOOL_CALL_COST))
        if self.tool_call_cost < 0.0:
            raise ValueError(f"tool_call_cost must be non-negative, got {self.tool_call_cost}")
        self.final_iqa_reward_scale = float(config.get("final_iqa_reward_scale", self.reward_scale))
        self.final_iqa_regression_penalty_scale = float(config.get("final_iqa_regression_penalty_scale", 1.0))
        self.final_iqa_step_penalty = float(config.get("final_iqa_step_penalty", 0.0))
        self.affinity_bonus_scale = float(config.get("affinity_bonus_scale", 0.0))
        self.repeat_action_penalty = float(config.get("repeat_action_penalty", REPEAT_ACTION_PENALTY))
        self.repeat_low_gain_penalty = float(config.get("repeat_low_gain_penalty", REPEAT_LOW_GAIN_PENALTY))
        self.repeat_low_gain_threshold = float(config.get("repeat_low_gain_threshold", REPEAT_LOW_GAIN_THRESHOLD))
        self.stop_min_step = int(config.get("stop_min_step", STOP_MIN_STEP))
        self.stop_iqa_delta_threshold = float(config.get("stop_iqa_delta_threshold", STOP_IQA_DELTA_THRESHOLD))
        self.stop_early_penalty = float(config.get("stop_early_penalty", STOP_EARLY_PENALTY))
        self.stop_recent_reward_window = int(config.get("stop_recent_reward_window", STOP_RECENT_REWARD_WINDOW))
        self.stop_recent_reward_threshold = float(
            config.get("stop_recent_reward_threshold", STOP_RECENT_REWARD_THRESHOLD)
        )

        example_root = Path(__file__).resolve().parents[3]
        default_external_tools_root = example_root.parents[1] / "External_Tools"
        self.tool_registry_path = _resolve_path(
            config.get("tool_registry_path", str(example_root / "config" / "tools.yaml")),
            base_dir=example_root,
        )
        self.external_tools_root = _resolve_path(
            config.get("external_tools_root", str(default_external_tools_root)),
            base_dir=example_root,
        )
        self.restoration_entrypoint = _resolve_path(
            config.get("restoration_entrypoint", str(example_root / "tool_runtime" / "restoration_entrypoint.py")),
            base_dir=example_root,
        )
        self.candidate_timeout_seconds = float(config.get("candidate_timeout_seconds", 600.0))
        (
            self.allowed_actions,
            self.candidate_tool_runtimes,
            self.model_to_runtime_actions,
        ) = _load_tool_runtime_config(self.tool_registry_path)
        if not self.allowed_actions:
            self.allowed_actions = set(ALLOWED_ACTIONS)
        self.allowed_actions.add("stop")
        self.runtime_to_model_actions = {
            runtime_action: model_action
            for model_action, runtime_action in self.model_to_runtime_actions.items()
        }

        project_root = Path(__file__).resolve().parent.parent.parent
        if self.iqa_stats_path and not os.path.isabs(self.iqa_stats_path):
            self.iqa_stats_path = str((project_root / self.iqa_stats_path).resolve())
        if self.iqa_qalign_path and not os.path.isabs(self.iqa_qalign_path):
            self.iqa_qalign_path = str((project_root / self.iqa_qalign_path).resolve())
        if self.iqa_metric_config_path and not os.path.isabs(self.iqa_metric_config_path):
            self.iqa_metric_config_path = str((project_root / self.iqa_metric_config_path).resolve())
        if self.iqa_weight_map_path and not os.path.isabs(self.iqa_weight_map_path):
            self.iqa_weight_map_path = str((project_root / self.iqa_weight_map_path).resolve())
        if self.normalize_iqa_scores and not (self.iqa_stats_path or self.iqa_metric_config_path):
            raise ValueError("normalize_iqa_scores=true requires iqa_stats_path or iqa_metric_config_path")

        self.score_weight_map = {
            degradation_type: _normalize_score_weight_vector(weights)
            for degradation_type, weights in SCORE_WEIGHT_MAP.items()
        }
        if self.iqa_weight_map_path:
            self.score_weight_map = _load_score_weight_map(self.iqa_weight_map_path)
        elif self.iqa_metric_config_path:
            self.score_weight_map = _load_score_weight_map(self.iqa_metric_config_path)
        self.default_weight = (
            self.score_weight_map.get("__default__")
            or self.score_weight_map.get("default")
            or _normalize_score_weight_vector(DEFAULT_WEIGHT)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self._toolkit = None
        self._iqa = None
        self._runtime_pool = None
        cache_max_size_gb = float(config.get("tool_result_cache_max_size_gb", 20.0))
        cache_ttl_hours = float(config.get("tool_result_cache_ttl_hours", 24.0))
        self.tool_result_cache = _ToolResultDiskCache(
            enabled=bool(config.get("enable_tool_result_cache", False)),
            cache_dir=str(config.get("tool_result_cache_dir") or Path(self.output_dir) / ".tool_result_cache"),
            signature=self._build_tool_result_cache_signature(),
            max_entries=int(config.get("tool_result_cache_max_entries", 5000)),
            max_size_bytes=int(cache_max_size_gb * 1024**3),
            ttl_seconds=cache_ttl_hours * 3600,
            cleanup_interval=int(config.get("tool_result_cache_cleanup_interval", 256)),
        )

        logger.info(
            f"RestorationTool initialized: device={self.device}, iqa_device={self.iqa_device}, "
            f"worker_devices={self.worker_devices}, iqa_devices={self.iqa_devices}, "
            f"use_iqa={self.use_iqa}, normalize_iqa_scores={self.normalize_iqa_scores}, "
            f"reward_mode={self.reward_mode}, "
            f"alpha={self.alpha}, reward_scale={self.reward_scale}, "
            f"tool_call_cost={self.tool_call_cost}, "
            f"final_iqa_reward_scale={self.final_iqa_reward_scale}, "
            f"final_iqa_regression_penalty_scale={self.final_iqa_regression_penalty_scale}, "
            f"final_iqa_step_penalty={self.final_iqa_step_penalty}, "
            f"affinity_bonus_scale={self.affinity_bonus_scale}, "
            f"repeat_action_penalty={self.repeat_action_penalty}, "
            f"repeat_low_gain_penalty={self.repeat_low_gain_penalty}, "
            f"stop_min_step={self.stop_min_step}, "
            f"model_devices={self.model_devices}, "
            f"metric_config_path={self.iqa_metric_config_path}, "
            f"weight_map_path={self.iqa_weight_map_path}, "
            f"tool_registry_path={self.tool_registry_path}, "
            f"tool_result_cache_enabled={self.tool_result_cache.enabled}, "
            f"tool_result_cache_dir={self.tool_result_cache.cache_dir}, "
            f"tool_result_cache_max_entries={self.tool_result_cache.max_entries}, "
            f"tool_result_cache_max_size_gb={cache_max_size_gb}, "
            f"tool_result_cache_ttl_hours={cache_ttl_hours}, "
            f"candidate_actions={sorted(self.candidate_tool_runtimes)}"
        )

    def _path_signature(self, path: str | None, *, hash_file: bool = False) -> dict[str, Any] | None:
        if path is None:
            return None
        signature: dict[str, Any] = {"path": str(path)}
        if hash_file and os.path.isfile(path):
            try:
                signature["sha256"] = _ToolResultDiskCache._hash_file(path)
            except Exception as e:
                logger.debug("Failed to hash signature file %s: %s", path, e)
        return signature

    def _build_tool_result_cache_signature(self) -> dict[str, Any]:
        return {
            "tool_registry": self._path_signature(self.tool_registry_path, hash_file=True),
            "candidate_tool_runtimes": _ToolResultDiskCache._json_safe(self.candidate_tool_runtimes),
            "preload_models": _ToolResultDiskCache._json_safe(self.preload_models),
            "use_iqa": self.use_iqa,
            "normalize_iqa_scores": self.normalize_iqa_scores,
            "iqa_stats": self._path_signature(self.iqa_stats_path, hash_file=True),
            "iqa_qalign": self._path_signature(self.iqa_qalign_path, hash_file=False),
            "iqa_metric_config": self._path_signature(self.iqa_metric_config_path, hash_file=True),
        }

    @property
    def toolkit(self):
        if self._toolkit is None:
            self._toolkit = get_toolkit(
                device=self.device,
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload,
                model_devices=self.model_devices,
                model_device_map=self.model_device_map,
            )
        return self._toolkit

    @property
    def iqa(self):
        if self._iqa is None and self.use_iqa:
            self._iqa = get_iqa_scorer(
                device=self.iqa_device,
                normalize_scores=self.normalize_iqa_scores,
                normalization_stats_path=self.iqa_stats_path,
                qalign_path=self.iqa_qalign_path,
                metric_config_path=self.iqa_metric_config_path,
            )
        return self._iqa

    @property
    def runtime_pool(self):
        if not self.use_parallel_workers:
            raise RuntimeError("runtime_pool requested but worker_devices is not configured")
        if self._runtime_pool is None:
            self._runtime_pool = get_runtime_pool(
                worker_devices=self.worker_devices,
                iqa_devices=self.iqa_devices,
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload,
                use_iqa=self.use_iqa,
                normalize_iqa_scores=self.normalize_iqa_scores,
                iqa_stats_path=self.iqa_stats_path,
                iqa_qalign_path=self.iqa_qalign_path,
                iqa_metric_config_path=self.iqa_metric_config_path,
            )
        return self._runtime_pool

    def _run_candidate_action(self, action: str, input_path: str, output_dir: str, device: str) -> dict[str, Any]:
        runtime = self.candidate_tool_runtimes[action]
        if self.external_tools_root is None or self.restoration_entrypoint is None:
            raise RuntimeError("candidate tools require external_tools_root and restoration_entrypoint")

        external_tools_root = Path(self.external_tools_root)
        output_path = Path(output_dir) / f"{Path(input_path).stem}_{action}_{uuid4().hex[:8]}.png"
        repo = external_tools_root / str(runtime["repo"])
        checkpoint = external_tools_root / str(runtime["checkpoint"])
        cmd = [
            sys.executable,
            str(self.restoration_entrypoint),
            "--adapter",
            "candidate",
            "--model",
            str(runtime["model"]),
            "--external-tools-root",
            str(external_tools_root),
            "--repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            str(device),
        ]
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.candidate_timeout_seconds,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"candidate action {action} failed with code {completed.returncode}: {detail[-2000:]}")

        metadata: dict[str, Any] = {}
        for line in completed.stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                try:
                    metadata = json.loads(line.removeprefix("RESULT_JSON="))
                except json.JSONDecodeError:
                    metadata = {}
                break
        if not output_path.is_file():
            raise RuntimeError(f"candidate action {action} did not produce an image: {output_path}")
        return {
            "output_path": str(output_path),
            "adapter": "candidate",
            "model": str(runtime["model"]),
            "metadata": metadata,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _zero_iqa_scores(self) -> list[float]:
        return [0.0] * len(self.default_weight)

    def _get_iqa_scores(self, image_path: str) -> list[float]:
        """Compute IQA scores for an image in the configured metric order."""
        if not self.use_iqa:
            return self._zero_iqa_scores()
        try:
            scores = self.iqa.get_iqa_score(image_path)
            return list(scores)
        except Exception as e:
            logger.warning(f"IQA scoring failed for {image_path}: {e}")
            return self._zero_iqa_scores()

    def _get_iqa_scores_on_worker(self, worker: _RestorationRuntimeWorker, image_path: str) -> list[float]:
        """Compute IQA scores using the scorer attached to a runtime worker."""
        if not self.use_iqa or worker.iqa is None:
            return self._zero_iqa_scores()
        try:
            scores = worker.iqa.get_iqa_score(image_path)
            return list(scores)
        except Exception as e:
            logger.warning(
                "IQA scoring failed for %s on worker %s (%s): %s",
                image_path,
                worker.index,
                worker.iqa_device,
                e,
            )
            return self._zero_iqa_scores()

    async def _aget_iqa_scores(self, image_path: str) -> list[float]:
        """Async IQA scoring that uses the replicated runtime pool when enabled."""
        if self.use_parallel_workers:
            return await self.runtime_pool.run(lambda worker: self._get_iqa_scores_on_worker(worker, image_path))
        return self._get_iqa_scores(image_path)

    def _repeat_type_key(self, action: str) -> str:
        """Return the repeat-penalty grouping key for an action."""
        return TOOL_REPEAT_TYPE_BY_ACTION.get(action, action)

    def _count_trajectory_type_repeats(self, actions_history: list[str], action: str) -> int:
        """Count prior actions in the trajectory that share the current action's repeat type."""
        repeat_type_key = self._repeat_type_key(action)
        return sum(
            1 for previous_action in actions_history if self._repeat_type_key(previous_action) == repeat_type_key
        )

    def _calculate_repeat_penalty(self, marginal: float, repeat_count: int) -> float:
        """Penalize same-type repeats only when their marginal IQA gain is low."""
        if repeat_count <= 0 or marginal > self.repeat_low_gain_threshold:
            return 0.0
        return (self.repeat_action_penalty + self.repeat_low_gain_penalty) * repeat_count

    def _calculate_reward(
        self,
        prev_scores: list[float],
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
        action: str,
        actions_history: list[str],
        degradation_type: str | None = None,
        best_identity_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Compute reward and diagnostics for a restoration step.

        Args:
            degradation_type: Optional degradation category (e.g. 'fog', 'rain_streak').
                Used to compute an affinity bonus when the action matches the degradation.
        """
        if self.reward_mode == REWARD_MODE_FINAL_IQA_V2:
            return self._calculate_final_iqa_reward_v2(
                prev_scores=prev_scores,
                curr_scores=curr_scores,
                identity_scores=identity_scores,
                weights=weights,
                action=action,
                actions_history=actions_history,
                best_identity_delta=best_identity_delta,
            )
        if self.reward_mode == REWARD_MODE_MARGINAL_EFFICIENCY_V1:
            return self._calculate_marginal_efficiency_reward_v1(
                prev_scores=prev_scores,
                curr_scores=curr_scores,
                identity_scores=identity_scores,
                weights=weights,
                action=action,
                actions_history=actions_history,
                best_identity_delta=best_identity_delta,
            )

        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        mixed = self.alpha * marginal + self.beta * identity
        base_reward = mixed * self.reward_scale

        # Affinity bonus: reward choosing degradation-specific tools over generic ones.
        # Only awarded once per trajectory — the first time a high-affinity tool is chosen.
        # Subsequent affinity-matched actions receive no bonus so the model does not
        # learn to simply repeat the same specialized tool for extra reward.
        affinity_bonus = 0.0
        if degradation_type and self.affinity_bonus_scale > 0:
            affinity_map = DEGRADATION_ACTION_AFFINITY.get(degradation_type, {})
            affinity_score = affinity_map.get(action, 0.0)
            if affinity_score > 0:
                # Check whether an affinity bonus was already given in this trajectory
                already_given = any(affinity_map.get(prev, 0.0) > 0 for prev in actions_history)
                if not already_given:
                    affinity_bonus = self.affinity_bonus_scale * affinity_score

        repeat_tool_type_key = self._repeat_type_key(action)
        repeat_count = self._count_trajectory_type_repeats(actions_history, action)
        repeat_penalty = self._calculate_repeat_penalty(marginal, repeat_count)

        reward = float(torch.clamp(torch.tensor(base_reward - repeat_penalty + affinity_bonus), -10.0, 10.0).item())
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "affinity_bonus": float(affinity_bonus),
            "consecutive_action_count": float(repeat_count + 1),
            "same_type_action_count": float(repeat_count + 1),
            "repeat_tool_type_key": repeat_tool_type_key,
            "best_identity_delta": float(max(best_identity_delta, identity)),
            "best_improvement": float(max(0.0, identity - best_identity_delta)),
            "regression_penalty": 0.0,
            "step_penalty": 0.0,
            "tool_call_cost": 0.0,
        }

    def _calculate_marginal_efficiency_reward_v1(
        self,
        prev_scores: list[float],
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
        action: str,
        actions_history: list[str],
        best_identity_delta: float,
    ) -> dict[str, Any]:
        """Reward marginal IQA improvement while charging for each successful tool call.

        Before clipping and repeat penalties, summing the reward over a trajectory yields
        ``reward_scale * (final_score - initial_score) - tool_call_cost * tool_calls``.
        The original-image IQA delta is retained only as a diagnostic and is not
        included in the per-step base reward.
        """
        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        base_reward = marginal * self.reward_scale

        repeat_tool_type_key = self._repeat_type_key(action)
        repeat_count = self._count_trajectory_type_repeats(actions_history, action)
        repeat_penalty = self._calculate_repeat_penalty(marginal, repeat_count)

        reward = float(
            torch.clamp(
                torch.tensor(base_reward - self.tool_call_cost - repeat_penalty),
                -10.0,
                10.0,
            ).item()
        )
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "affinity_bonus": 0.0,
            "consecutive_action_count": float(repeat_count + 1),
            "same_type_action_count": float(repeat_count + 1),
            "repeat_tool_type_key": repeat_tool_type_key,
            "best_identity_delta": float(max(best_identity_delta, identity)),
            "best_improvement": float(max(0.0, identity - best_identity_delta)),
            "regression_penalty": 0.0,
            "step_penalty": float(self.tool_call_cost),
            "tool_call_cost": float(self.tool_call_cost),
        }

    def _calculate_final_iqa_reward_v2(
        self,
        prev_scores: list[float],
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
        action: str,
        actions_history: list[str],
        best_identity_delta: float,
    ) -> dict[str, Any]:
        """Reward improvements to the trajectory-best final IQA score.

        The summed v2 reward tracks the best weighted IQA improvement achieved
        so far, instead of paying every marginal step independently. This keeps
        the objective aligned with the final restored image quality and avoids
        rewarding extra tool calls that do not produce a new best image.
        """
        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        best_improvement = max(0.0, identity - best_identity_delta)
        regression = max(0.0, best_identity_delta - identity)
        base_reward = best_improvement * self.final_iqa_reward_scale
        regression_penalty = regression * self.final_iqa_regression_penalty_scale

        repeat_tool_type_key = self._repeat_type_key(action)
        repeat_count = self._count_trajectory_type_repeats(actions_history, action)
        repeat_penalty = self._calculate_repeat_penalty(marginal, repeat_count)

        reward = float(
            torch.clamp(
                torch.tensor(base_reward - regression_penalty - self.final_iqa_step_penalty - repeat_penalty),
                -10.0,
                10.0,
            ).item()
        )
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "affinity_bonus": 0.0,
            "consecutive_action_count": float(repeat_count + 1),
            "same_type_action_count": float(repeat_count + 1),
            "repeat_tool_type_key": repeat_tool_type_key,
            "best_identity_delta": float(max(best_identity_delta, identity)),
            "best_improvement": float(best_improvement),
            "regression_penalty": float(regression_penalty),
            "step_penalty": float(self.final_iqa_step_penalty),
            "tool_call_cost": 0.0,
        }

    def _calculate_identity_delta(
        self,
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
    ) -> float:
        """Compute weighted IQA improvement over the original degraded image."""
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)
        return float(((curr_t - iden_t) * w_t).sum().item())

    @staticmethod
    def _weighted_aggregate_score(scores: list[float], weights: list[float]) -> float:
        """Return the weighted aggregate IQA score used by the reward."""
        return float(sum(float(score) * float(weight) for score, weight in zip(scores, weights, strict=False)))

    @staticmethod
    def _append_trajectory_call(instance: dict[str, Any], action: str, **details: Any) -> None:
        """Append one structured tool call to an in-memory trajectory."""
        calls = instance.setdefault("trajectory_calls", [])
        calls.append(
            {
                "call_index": len(calls) + 1,
                "action": action,
                **details,
            }
        )

    def _build_trajectory_summary(self, instance_id: str, instance: dict[str, Any]) -> dict[str, Any]:
        """Build the single JSON-serializable record emitted for a trajectory."""
        calls = list(instance.get("trajectory_calls", []))
        scores_history = instance.get("scores_history", [])
        initial_scores = list(instance.get("identity_scores", scores_history[0] if scores_history else []))
        final_scores = list(scores_history[-1]) if scores_history else initial_scores
        weights = list(instance.get("weights", []))
        restoration_rewards = [
            float(call.get("reward", 0.0))
            for call in calls
            if call.get("action") != "stop" and call.get("status", "success") == "success"
        ]
        stop_rewards = [float(call.get("reward", 0.0)) for call in calls if call.get("action") == "stop"]
        call_rewards = [float(call.get("reward", 0.0)) for call in calls]
        if not calls:
            call_rewards = [float(value) for value in instance.get("rewards_history", [])]
        action_path = [str(call.get("action", "")) for call in calls]
        if not action_path:
            action_path = [str(action) for action in instance.get("actions_history", [])]
        finished_at = time.time()
        started_at = float(instance.get("trajectory_started_at", finished_at))

        return {
            "event": "restoration_trajectory",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished_at)),
            "trajectory_id": instance_id,
            "degradation_type": instance.get("degradation_type"),
            "original_image": instance.get("original_image"),
            "final_image": instance.get("current_image"),
            "reward_mode": self.reward_mode,
            "termination_reason": "stop" if stop_rewards else "rollout_end_without_stop",
            "action_path": action_path,
            "restoration_step_count": int(instance.get("step", len(instance.get("actions_history", [])))),
            "tool_call_count": len(calls),
            "initial_iqa_scores": initial_scores,
            "initial_aggregate_score": self._weighted_aggregate_score(initial_scores, weights),
            "final_iqa_scores": final_scores,
            "final_aggregate_score": self._weighted_aggregate_score(final_scores, weights),
            "final_identity_delta": self._calculate_identity_delta(final_scores, initial_scores, weights),
            "best_identity_delta": float(instance.get("best_identity_delta", 0.0)),
            "restoration_reward_sum": float(sum(restoration_rewards)),
            "stop_reward": float(sum(stop_rewards)),
            "total_tool_reward": float(sum(call_rewards)),
            "failed_tool_call_count": sum(1 for call in calls if call.get("status") == "error"),
            "total_repeat_penalty": float(sum(float(call.get("repeat_penalty", 0.0)) for call in calls)),
            "total_tool_call_cost": float(sum(float(call.get("tool_call_cost", 0.0)) for call in calls)),
            "duration_seconds": max(0.0, finished_at - started_at),
            "tool_calls": calls,
        }

    def _log_trajectory_summary(self, instance_id: str, instance: dict[str, Any]) -> None:
        """Emit exactly one compact JSON line for a completed trajectory."""
        summary = self._build_trajectory_summary(instance_id, instance)
        trajectory_logger.info(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), default=str))

    def _calculate_stop_reward(
        self,
        step: int,
        identity_delta: float,
        recent_rewards: list[float],
    ) -> dict[str, float | bool]:
        """Return a neutral eligible-stop reward or the configured early-stop penalty."""
        recent_reward_mean = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
        plateau = bool(recent_rewards) and recent_reward_mean <= self.stop_recent_reward_threshold
        good_enough = identity_delta >= self.stop_iqa_delta_threshold

        if self.reward_mode == REWARD_MODE_MARGINAL_EFFICIENCY_V1:
            reward = 0.0
        elif step < self.stop_min_step:
            reward = self.stop_early_penalty
        else:
            reward = 0.0

        return {
            "reward": float(torch.clamp(torch.tensor(reward), -10.0, 10.0).item()),
            "recent_reward_mean": recent_reward_mean,
            "plateau": plateau,
            "good_enough": good_enough,
        }

    def _generate_feedback(
        self,
        action: str,
        step: int,
        reward: float,
        actions_history: list[str],
        marginal: float,
        identity_delta: float,
        same_type_action_count: int,
        repeat_tool_type_key: str | None = None,
        degradation_type: str | None = None,
        best_identity_delta: float | None = None,
        best_improvement: float | None = None,
        regression_penalty: float = 0.0,
        step_penalty: float = 0.0,
    ) -> str:
        """Generate human-readable feedback for the model's next turn.

        Args:
            degradation_type: Optional degradation category for affinity hints.
        """
        model_action = self.runtime_to_model_actions.get(action, action)
        model_history = [self.runtime_to_model_actions.get(item, item) for item in actions_history]
        history_str = " → ".join(model_history) if model_history else "none"
        repeat_tool_type_key = repeat_tool_type_key or action

        def _repeat_feedback_line(final_iqa_mode: bool) -> str:
            if repeat_tool_type_key == action:
                prefix = f"Trajectory uses of '{model_action}': {same_type_action_count}."
            else:
                prefix = f"Trajectory uses of {repeat_tool_type_key} tools: {same_type_action_count}."
            suffix = (
                " Reusing the same tool type without a new best IQA is discouraged."
                if final_iqa_mode
                else " Reusing the same tool type without clear gains is discouraged."
            )
            return prefix + suffix

        if self.reward_mode == REWARD_MODE_MARGINAL_EFFICIENCY_V1:
            break_even_gain = self.tool_call_cost / self.reward_scale if self.reward_scale > 0.0 else float("inf")
            lines = [
                f"Step {step}: Applied '{model_action}'.",
                f"Step reward: {reward:.4f}",
                f"Weighted marginal improvement: {marginal:.4f}",
                f"Tool-call cost: {self.tool_call_cost:.4f}",
                f"Break-even marginal improvement: {break_even_gain:.4f}",
                f"Current improvement over original image: {identity_delta:.4f}",
                f"Action history: {history_str}",
            ]
            if same_type_action_count > 1:
                lines.append(_repeat_feedback_line(final_iqa_mode=False))
            if marginal > break_even_gain:
                lines.append("This action's IQA gain exceeded its tool-call cost.")
            else:
                lines.append("This action's IQA gain did not cover its tool-call cost.")
            return "\n".join(lines)

        if self.reward_mode == REWARD_MODE_FINAL_IQA_V2:
            best_identity_delta = identity_delta if best_identity_delta is None else best_identity_delta
            best_improvement = 0.0 if best_improvement is None else best_improvement
            lines = [
                f"Step {step}: Applied '{model_action}'.",
                f"Step reward: {reward:.4f}",
                f"Current improvement over original image: {identity_delta:.4f}",
                f"Trajectory-best improvement over original image: {best_identity_delta:.4f}",
                f"New best-IQA gain from this action: {best_improvement:.4f}",
                f"Weighted marginal improvement: {marginal:.4f}",
                f"Action history: {history_str}",
            ]
            if regression_penalty > 0.0:
                lines.append(
                    f"This action fell below the trajectory-best IQA "
                    f"(regression penalty: {regression_penalty:.4f})."
                )
            if step_penalty > 0.0:
                lines.append(f"Each extra tool call has a step cost of {step_penalty:.4f}.")
            if same_type_action_count > 1:
                lines.append(_repeat_feedback_line(final_iqa_mode=True))
            if best_improvement > 0.0:
                lines.append(
                    "This action set a new trajectory-best IQA. Continue only if another "
                    "tool is likely to raise the best score further; otherwise stop."
                )
            elif step >= self.stop_min_step:
                lines.append(
                    "This action did not improve the trajectory-best IQA. Prefer stopping "
                    "or switching to a different targeted operation instead of repeating it."
                )
            else:
                lines.append(
                    "Continue exploring only if the next action is likely to improve the " "trajectory-best IQA."
                )
            return "\n".join(lines)

        lines = [
            f"Step {step}: Applied '{model_action}'.",
            f"Step reward: {reward:.4f}",
            f"Weighted marginal improvement: {marginal:.4f}",
            f"Improvement over original image: {identity_delta:.4f}",
            f"Action history: {history_str}",
        ]
        if same_type_action_count > 1:
            lines.append(_repeat_feedback_line(final_iqa_mode=False))
        # NOTE: degradation-type affinity hints are intentionally omitted from the
        # feedback text.  The model must learn to diagnose the degradation and
        # choose appropriate tools on its own — revealing the degradation type or
        # recommending specific actions would shortcut that learning process.
        if step >= self.stop_min_step and marginal <= self.repeat_low_gain_threshold:
            lines.append("Recent gains are small. Consider stopping now or switch to a different targeted operation.")
        elif step >= self.stop_min_step:
            lines.append(
                "You have completed several restoration steps. "
                "If gains keep shrinking, prefer stopping over repeating the same action."
            )
        else:
            lines.append("Continue with the next restoration action or stop if the image looks good.")
        return "\n".join(lines)

    async def create(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        image_path: Optional[str] = None,
        degradation_type: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: Optional instance identifier (generated if None).
            original_image: Path to the original degraded image.
            image_path: Alias for original_image (for dataset compatibility).
            degradation_type: Degradation category, e.g. 'fog', 'night', 'snow'.
                              Used to select IQA metric weights.
        """
        if original_image is None:
            original_image = kwargs.get("create_kwargs", {}).get("original_image", None)
        if image_path is None:
            image_path = kwargs.get("create_kwargs", {}).get("image_path", None)
        if degradation_type is None:
            degradation_type = kwargs.get("create_kwargs", {}).get("degradation_type", None)
        if original_image is None and image_path is not None:
            original_image = image_path

        if instance_id is None:
            instance_id = str(uuid4())

        instance_output_dir = os.path.join(self.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)

        weights = self.score_weight_map.get(degradation_type, self.default_weight)

        # Compute identity (original) IQA scores. In replicated mode this is
        # dispatched to whichever GPU worker is free, so batch create() calls
        # can score originals in parallel.
        identity_scores = await self._aget_iqa_scores(original_image) if original_image else self._zero_iqa_scores()

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "processed_images": [],
            "actions_history": [],
            "scores_history": [identity_scores],
            "rewards_history": [],
            "marginals_history": [],
            "best_identity_delta": 0.0,
            "identity_scores": identity_scores,
            "weights": weights,
            "degradation_type": degradation_type,
            "step": 0,
            "output_dir": instance_output_dir,
            "trajectory_started_at": time.time(),
            "trajectory_calls": [],
        }

        logger.info(
            f"Created restoration instance {instance_id} for: {original_image} "
            f"(degradation={degradation_type}, identity_scores={identity_scores})"
        )
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a restoration action and return an IQA-based step reward.

        Args:
            instance_id: The instance identifier returned by ``create``.
            parameters: Dict with one model-facing ``action`` schema value.

        Returns:
            (ToolResponse, step_reward, metrics_dict)
        """
        model_action = str(parameters.get("action", "")).strip()
        action = self.model_to_runtime_actions.get(model_action)

        allowed_actions = getattr(self, "allowed_actions", ALLOWED_ACTIONS)
        instance = self._instance_dict.get(instance_id)
        if action is None or action not in allowed_actions:
            allowed_model_actions = sorted(self.model_to_runtime_actions)
            error_msg = f"Invalid action '{model_action}'. Allowed: {', '.join(allowed_model_actions)}"
            if instance is not None:
                self._append_trajectory_call(
                    instance,
                    model_action,
                    status="error",
                    restoration_step=int(instance.get("step", 0)),
                    reward=INVALID_ACTION_REWARD,
                    error="invalid_action",
                )
            logger.warning(error_msg)
            return (
                ToolResponse(text=error_msg),
                INVALID_ACTION_REWARD,
                {
                    "model_action": model_action,
                    "error": "invalid_action",
                    "skip_tool_call_reward": True,
                },
            )

        if instance is None:
            error_msg = f"Instance {instance_id} not found"
            logger.error(error_msg)
            return (
                ToolResponse(text=error_msg),
                FAILURE_REWARD,
                {"error": "instance_not_found", "skip_tool_call_reward": True},
            )

        if action == "stop":
            step = instance["step"]
            curr_scores = instance["scores_history"][-1]
            identity_scores = instance["identity_scores"]
            weights = instance["weights"]
            identity_delta = self._calculate_identity_delta(curr_scores, identity_scores, weights)
            recent_rewards = instance["rewards_history"][-self.stop_recent_reward_window :]
            stop_info = self._calculate_stop_reward(step, identity_delta, recent_rewards)
            reward = float(stop_info["reward"])
            instance["rewards_history"].append(reward)
            self._append_trajectory_call(
                instance,
                "stop",
                status="stopped",
                restoration_step=step,
                reward=reward,
                identity_delta=identity_delta,
                best_identity_delta=float(instance.get("best_identity_delta", identity_delta)),
                recent_reward_mean=float(stop_info["recent_reward_mean"]),
                plateau=bool(stop_info["plateau"]),
                good_enough=bool(stop_info["good_enough"]),
            )
            logger.info(
                f"Instance {instance_id}: stop action at step {step}, "
                f"identity_delta={identity_delta:.4f}, recent_reward_mean={stop_info['recent_reward_mean']:.4f}, "
                f"plateau={stop_info['plateau']}, good_enough={stop_info['good_enough']}, reward={reward}"
            )
            reason_parts = []
            if stop_info["good_enough"]:
                reason_parts.append("quality is already good enough")
            if stop_info["plateau"]:
                reason_parts.append("recent gains are small")
            reason_text = "; ".join(reason_parts) if reason_parts else "more improvement is still possible"
            return (
                ToolResponse(text=f"Restoration stopped after {step} step(s): {reason_text}."),
                reward,
                {
                    "action": "stop",
                    "model_action": model_action,
                    "step": step,
                    "reward_mode": self.reward_mode,
                    "identity_delta": identity_delta,
                    "best_identity_delta": float(instance.get("best_identity_delta", identity_delta)),
                    "recent_reward_mean": stop_info["recent_reward_mean"],
                    "plateau": stop_info["plateau"],
                    "good_enough": stop_info["good_enough"],
                    "skip_tool_call_reward": True,
                },
            )

        current_image = instance["current_image"]
        output_dir = instance["output_dir"]

        try:
            logger.info(f"Instance {instance_id}: applying '{action}' to {current_image}")
            cache_hit = False
            cache_key = None
            cached_result = self.tool_result_cache.get(action, current_image, output_dir)
            if cached_result is not None and cached_result.iqa_scores:
                result = cached_result.result
                curr_scores = cached_result.iqa_scores
                cache_hit = True
                cache_key = cached_result.cache_key
                logger.info(
                    "Instance %s: cache hit for '%s' on %s (key=%s)",
                    instance_id,
                    action,
                    current_image,
                    cache_key,
                )
            else:
                if action in self.candidate_tool_runtimes:
                    if self.use_parallel_workers:

                        def _restore_and_score(worker: _RestorationRuntimeWorker):
                            result = self._run_candidate_action(action, current_image, output_dir, worker.device)
                            output_path = result.get("output_path")
                            if output_path and os.path.exists(output_path):
                                curr_scores = self._get_iqa_scores_on_worker(worker, output_path)
                            else:
                                curr_scores = self._zero_iqa_scores()
                            return result, curr_scores, worker.index, worker.device, worker.iqa_device

                        result, curr_scores, worker_index, worker_device, worker_iqa_device = (
                            await self.runtime_pool.run(_restore_and_score)
                        )
                        logger.info(
                            "Instance %s: worker %s completed candidate '%s' (tool=%s, iqa=%s)",
                            instance_id,
                            worker_index,
                            action,
                            worker_device,
                            worker_iqa_device,
                        )
                    else:
                        result = self._run_candidate_action(action, current_image, output_dir, self.device)
                        curr_scores = None
                elif self.use_parallel_workers:

                    def _restore_and_score(worker: _RestorationRuntimeWorker):
                        result = worker.toolkit.process_image(
                            tools=[action],
                            img_path=current_image,
                            output_dir=output_dir,
                            is_identify=True,
                        )
                        output_path = result.get("output_path")
                        if output_path and os.path.exists(output_path):
                            curr_scores = self._get_iqa_scores_on_worker(worker, output_path)
                        else:
                            curr_scores = self._zero_iqa_scores()
                        return result, curr_scores, worker.index, worker.device, worker.iqa_device

                    result, curr_scores, worker_index, worker_device, worker_iqa_device = await self.runtime_pool.run(
                        _restore_and_score
                    )
                    logger.info(
                        "Instance %s: worker %s completed '%s' (tool=%s, iqa=%s)",
                        instance_id,
                        worker_index,
                        action,
                        worker_device,
                        worker_iqa_device,
                    )
                else:
                    result = self.toolkit.process_image(
                        tools=[action],
                        img_path=current_image,
                        output_dir=output_dir,
                        is_identify=True,
                    )
                    curr_scores = None

            output_path = result.get("output_path")
            if not output_path or not os.path.exists(output_path):
                error_msg = "Restoration failed: no output generated"
                self._append_trajectory_call(
                    instance,
                    action,
                    status="error",
                    restoration_step=int(instance.get("step", 0)),
                    reward=FAILURE_REWARD,
                    error="restoration_failed",
                )
                logger.error(error_msg)
                return (
                    ToolResponse(text=error_msg),
                    FAILURE_REWARD,
                    {"error": "restoration_failed", "skip_tool_call_reward": True},
                )

            # Compute IQA scores for the new image
            if curr_scores is None:
                curr_scores = self._get_iqa_scores(output_path)
            if not cache_hit:
                cache_key = self.tool_result_cache.put(action, current_image, result, curr_scores)
            prev_scores = instance["scores_history"][-1]
            identity_scores = instance["identity_scores"]
            weights = instance["weights"]

            reward_info = self._calculate_reward(
                prev_scores,
                curr_scores,
                identity_scores,
                weights,
                action=action,
                actions_history=instance["actions_history"],
                degradation_type=instance.get("degradation_type"),
                best_identity_delta=float(instance.get("best_identity_delta", 0.0)),
            )
            reward = float(reward_info["reward"])

            # Update instance state
            instance["processed_images"].append((action, output_path))
            instance["actions_history"].append(action)
            instance["current_image"] = output_path
            instance["step"] += 1
            instance["scores_history"].append(curr_scores)
            instance["rewards_history"].append(reward)
            instance["marginals_history"].append(float(reward_info["marginal"]))
            instance["best_identity_delta"] = float(reward_info["best_identity_delta"])

            identity_delta = self._calculate_identity_delta(curr_scores, identity_scores, weights)
            aggregate_score = float(
                (torch.tensor(curr_scores, dtype=torch.float32) * torch.tensor(weights, dtype=torch.float32))
                .sum()
                .item()
            )

            # Generate feedback text
            feedback = self._generate_feedback(
                action=action,
                step=instance["step"],
                reward=reward,
                actions_history=instance["actions_history"],
                marginal=float(reward_info["marginal"]),
                identity_delta=identity_delta,
                same_type_action_count=int(reward_info["same_type_action_count"]),
                repeat_tool_type_key=str(reward_info["repeat_tool_type_key"]),
                degradation_type=instance.get("degradation_type"),
                best_identity_delta=float(reward_info["best_identity_delta"]),
                best_improvement=float(reward_info["best_improvement"]),
                regression_penalty=float(reward_info["regression_penalty"]),
                step_penalty=float(reward_info["step_penalty"]),
            )

            # Build response with restored image
            pil_img = None
            try:
                from PIL import Image as PILImage

                pil_img = PILImage.open(output_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load result image with PIL: {e}")

            response = ToolResponse(
                image=[pil_img] if pil_img is not None else None,
                text=feedback,
            )

            logger.info(
                f"Instance {instance_id}: '{action}' done, step={instance['step']}, "
                f"reward={reward:.4f}, marginal={reward_info['marginal']:.4f}, "
                f"identity={reward_info['identity']:.4f}, "
                f"best_identity_delta={reward_info['best_identity_delta']:.4f}, "
                f"repeat_penalty={reward_info['repeat_penalty']:.4f}, "
                f"affinity_bonus={reward_info['affinity_bonus']:.4f}, "
                f"cache_hit={cache_hit}, "
                f"output={output_path}"
            )
            metrics = {
                "action": action,
                "model_action": model_action,
                "step": instance["step"],
                "reward": reward,
                "reward_mode": self.reward_mode,
                "base_reward": reward_info["base_reward"],
                "marginal": reward_info["marginal"],
                "aggregate_score": aggregate_score,
                "identity_delta": identity_delta,
                "best_identity_delta": reward_info["best_identity_delta"],
                "best_improvement": reward_info["best_improvement"],
                "regression_penalty": reward_info["regression_penalty"],
                "step_penalty": reward_info["step_penalty"],
                "tool_call_cost": reward_info["tool_call_cost"],
                "repeat_penalty": reward_info["repeat_penalty"],
                "affinity_bonus": reward_info["affinity_bonus"],
                "consecutive_action_count": int(reward_info["consecutive_action_count"]),
                "same_type_action_count": int(reward_info["same_type_action_count"]),
                "repeat_tool_type_key": reward_info["repeat_tool_type_key"],
                "iqa_scores": curr_scores,
                "input_path": current_image,
                "output_path": output_path,
                "tool_result_cache_hit": cache_hit,
            }
            if cache_key is not None:
                metrics["tool_result_cache_key"] = cache_key
            self._append_trajectory_call(
                instance,
                action,
                status="success",
                restoration_step=instance["step"],
                reward=reward,
                base_reward=float(reward_info["base_reward"]),
                marginal_iqa_gain=float(reward_info["marginal"]),
                aggregate_score=aggregate_score,
                identity_delta=identity_delta,
                best_identity_delta=float(reward_info["best_identity_delta"]),
                best_improvement=float(reward_info["best_improvement"]),
                regression_penalty=float(reward_info["regression_penalty"]),
                step_penalty=float(reward_info["step_penalty"]),
                tool_call_cost=float(reward_info["tool_call_cost"]),
                repeat_penalty=float(reward_info["repeat_penalty"]),
                affinity_bonus=float(reward_info["affinity_bonus"]),
                repeat_tool_type_key=str(reward_info["repeat_tool_type_key"]),
                same_type_action_count=int(reward_info["same_type_action_count"]),
                iqa_scores=[float(score) for score in curr_scores],
                cache_hit=cache_hit,
                input_image=current_image,
                output_image=output_path,
            )
            return response, reward, metrics

        except Exception as e:
            error_msg = f"Restoration error during '{action}': {e}"
            self._append_trajectory_call(
                instance,
                action,
                status="error",
                restoration_step=int(instance.get("step", 0)),
                reward=FAILURE_REWARD,
                error=type(e).__name__,
                error_message=str(e),
            )
            logger.exception(error_msg)
            return (
                ToolResponse(text=error_msg),
                FAILURE_REWARD,
                {"error": str(e), "skip_tool_call_reward": True},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return cumulative reward for the trajectory (sum of step rewards)."""
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return 0.0
        return float(sum(instance.get("rewards_history", [])))

    async def release(self, instance_id: str, **kwargs) -> None:
        instance = self._instance_dict.pop(instance_id, None)
        if instance is not None:
            try:
                self._log_trajectory_summary(instance_id, instance)
            except Exception:
                logger.exception("Failed to write trajectory summary for restoration instance %s", instance_id)
            logger.info(f"Released restoration instance {instance_id}")

    def get_instance_state(self, instance_id: str) -> Optional[dict]:
        return self._instance_dict.get(instance_id)
