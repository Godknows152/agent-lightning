#!/usr/bin/env python3
"""Export a VERL FSDP actor checkpoint as a standard PEFT LoRA adapter.

Typical usage from the repository root:

    /home/LXJ/anaconda3/envs/verl/bin/python \
      examples/image_restoration_multi_agent/old_verl_grpo/scripts/export_LoRA/export_fsdp_checkpoint_to_lora.py \
      examples/image_restoration_multi_agent/old_verl_grpo/outputs/fog/v4.1.2/4gpu/0802/global_step_130 \
      examples/image_restoration_multi_agent/old_verl_grpo/outputs/fog/LoRA/v4.1.2/0802_step130

The checkpoint argument may point either to ``global_step_N`` or directly to
its ``actor`` directory. Only LoRA tensors are materialized, so the script does
not write a merged copy of the base model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_BASE_MODEL = "/home/LXJ/Python_Projects/Models/Qwen3.5-9B"
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
LORA_KEY_RE = re.compile(r"\.lora_([AB])\.weight$")


@dataclass
class ShardedParameter:
    global_shape: tuple[int, ...]
    placement: str
    shard_dim: int | None
    pieces: list[torch.Tensor] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LoRA weights from a VERL FSDP global_step checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to global_step_N or global_step_N/actor.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Exact destination for adapter_config.json and adapter_model.safetensors.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model path or Hugging Face model ID written to adapter_config.json.",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        help="Optional existing adapter directory or adapter_config.json to reuse as metadata.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help=f"Override LoRA alpha; otherwise use the reused config or {DEFAULT_LORA_ALPHA}.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        help=f"Override LoRA dropout; otherwise use the reused config or {DEFAULT_LORA_DROPOUT}.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Dtype of exported adapter tensors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing adapter_config.json/adapter_model.safetensors in output_dir.",
    )
    return parser.parse_args()


def resolve_actor_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    actor_dir = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    required = (actor_dir / "fsdp_config.json", actor_dir / "huggingface" / "config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Invalid actor checkpoint; missing: " + ", ".join(missing))
    return actor_dir


def load_fsdp_config(actor_dir: Path) -> tuple[int, int]:
    with (actor_dir / "fsdp_config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    fsdp_version = int(config.get("FSDP_version", -1))
    world_size = int(config.get("world_size", 0))
    if fsdp_version not in (1, 2):
        raise ValueError(f"Unsupported FSDP_version={fsdp_version}; expected 1 or 2")
    if world_size < 1:
        raise ValueError(f"Invalid world_size={world_size}")

    missing_ranks = [
        rank
        for rank in range(world_size)
        if not (actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
    ]
    if missing_ranks:
        raise FileNotFoundError(f"Missing model shards for ranks: {missing_ranks}")
    return fsdp_version, world_size


def normalize_lora_key(key: str) -> str:
    normalized = key.replace(".default.weight", ".weight")
    if not LORA_KEY_RE.search(normalized):
        raise ValueError(f"Unsupported LoRA parameter name: {key}")
    return normalized


def detach_copy(tensor: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=output_dtype).contiguous().clone()


def tensor_layout(value: Any, world_size: int) -> tuple[str, int | None, tuple[int, ...], torch.Tensor]:
    if hasattr(value, "placements") and hasattr(value, "_local_tensor"):
        placements = tuple(value.placements)
        if len(placements) != 1:
            raise NotImplementedError(f"Only one-dimensional FSDP placement is supported, got {placements}")

        placement = placements[0]
        if placement.is_shard():
            return "shard", int(placement.dim), tuple(value.shape), value._local_tensor
        if placement.is_replicate():
            return "replicate", None, tuple(value.shape), value._local_tensor
        raise NotImplementedError(f"Unsupported DTensor placement: {placement}")

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor or DTensor, got {type(value).__name__}")
    if world_size == 1:
        return "replicate", None, tuple(value.shape), value

    # This matches VERL's fallback for legacy non-DTensor FSDP state dicts.
    return "legacy_shard", 0, (), value


def extract_lora_state(actor_dir: Path, world_size: int, output_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    parameters: dict[str, ShardedParameter] = {}
    expected_keys: set[str] | None = None

    for rank in range(world_size):
        shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"[{rank + 1}/{world_size}] Reading {shard_path}", flush=True)
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
        raw_keys = {key for key in state_dict if "lora_" in key}
        if not raw_keys:
            raise ValueError(f"No LoRA parameters found in {shard_path}")
        if expected_keys is None:
            expected_keys = raw_keys
        elif raw_keys != expected_keys:
            missing = sorted(expected_keys - raw_keys)
            extra = sorted(raw_keys - expected_keys)
            raise ValueError(f"LoRA key mismatch at rank {rank}: missing={missing[:3]}, extra={extra[:3]}")

        for raw_key in sorted(raw_keys):
            output_key = normalize_lora_key(raw_key)
            placement, shard_dim, global_shape, local_tensor = tensor_layout(state_dict[raw_key], world_size)
            parameter = parameters.get(output_key)
            if parameter is None:
                parameter = ShardedParameter(global_shape, placement, shard_dim)
                parameters[output_key] = parameter
            elif (parameter.global_shape, parameter.placement, parameter.shard_dim) != (
                global_shape,
                placement,
                shard_dim,
            ):
                raise ValueError(f"Inconsistent shard metadata for {output_key} at rank {rank}")

            if placement != "replicate" or rank == 0:
                parameter.pieces.append(detach_copy(local_tensor, output_dtype))

        del state_dict
        gc.collect()

    merged: dict[str, torch.Tensor] = {}
    for key, parameter in sorted(parameters.items()):
        if parameter.placement == "replicate":
            tensor = parameter.pieces[0]
        else:
            if len(parameter.pieces) != world_size:
                raise ValueError(f"Expected {world_size} pieces for {key}, got {len(parameter.pieces)}")
            tensor = torch.cat(parameter.pieces, dim=parameter.shard_dim).contiguous()

        if parameter.global_shape and tuple(tensor.shape) != parameter.global_shape:
            raise ValueError(
                f"Merged shape mismatch for {key}: got {tuple(tensor.shape)}, expected {parameter.global_shape}"
            )
        merged[key] = tensor

    return merged


def inspect_adapter_state(state_dict: dict[str, torch.Tensor]) -> tuple[int, list[str]]:
    ranks: set[int] = set()
    target_modules: set[str] = set()
    adapter_sides: dict[str, set[str]] = {}

    for key, tensor in state_dict.items():
        match = LORA_KEY_RE.search(key)
        if match is None or tensor.ndim != 2:
            raise ValueError(f"Invalid LoRA tensor: {key} shape={tuple(tensor.shape)}")
        side = match.group(1)
        rank = tensor.shape[0] if side == "A" else tensor.shape[1]
        ranks.add(int(rank))
        module_prefix = key[: match.start()]
        target_modules.add(module_prefix.rsplit(".", 1)[-1])
        adapter_sides.setdefault(module_prefix, set()).add(side)

    incomplete = sorted(prefix for prefix, sides in adapter_sides.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(f"LoRA A/B pair is incomplete for: {incomplete[:3]}")
    if len(ranks) != 1:
        raise ValueError(f"A single global LoRA rank is required, found: {sorted(ranks)}")
    return ranks.pop(), sorted(target_modules)


def resolve_adapter_config_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "adapter_config.json" if path.is_dir() else path


def write_adapter_config(
    staging_dir: Path,
    adapter_config: Path | None,
    base_model: str,
    rank: int,
    target_modules: list[str],
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> dict[str, Any]:
    if adapter_config is not None:
        config_path = resolve_adapter_config_path(adapter_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Adapter config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            config_data = json.load(f)
        configured_rank = int(config_data.get("r", 0))
        if configured_rank != rank:
            raise ValueError(f"Adapter config rank {configured_rank} does not match checkpoint rank {rank}")
        config_data["base_model_name_or_path"] = base_model
        config_data["target_modules"] = target_modules
        config_data["inference_mode"] = True
        if lora_alpha is not None:
            config_data["lora_alpha"] = lora_alpha
        if lora_dropout is not None:
            config_data["lora_dropout"] = lora_dropout
        with (staging_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha if lora_alpha is not None else DEFAULT_LORA_ALPHA,
            lora_dropout=lora_dropout if lora_dropout is not None else DEFAULT_LORA_DROPOUT,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=True,
        )
        config.base_model_name_or_path = base_model
        config.save_pretrained(staging_dir)
        with (staging_dir / "adapter_config.json").open(encoding="utf-8") as f:
            config_data = json.load(f)
        config_data["target_modules"] = target_modules
        with (staging_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    return config_data


def validate_staged_adapter(staging_dir: Path, rank: int, expected_keys: set[str], dtype: torch.dtype) -> None:
    config = PeftConfig.from_pretrained(staging_dir)
    if int(getattr(config, "r", 0)) != rank:
        raise ValueError(f"Staged adapter config has rank={getattr(config, 'r', None)}, expected {rank}")

    weights_path = staging_dir / "adapter_model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != expected_keys:
            raise ValueError("Staged safetensors keys do not match the extracted adapter")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.dtype != dtype:
                raise ValueError(f"Unexpected dtype for {key}: {tensor.dtype}, expected {dtype}")


def publish_adapter(staging_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        os.replace(staging_dir / filename, output_dir / filename)


def check_output_files(output_dir: Path, overwrite: bool) -> None:
    output_files = [output_dir / "adapter_config.json", output_dir / "adapter_model.safetensors"]
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing adapter files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()
    actor_dir = resolve_actor_dir(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir in (actor_dir, actor_dir.parent):
        raise ValueError("output_dir must not be the actor or global_step checkpoint directory")
    check_output_files(output_dir, args.overwrite)

    fsdp_version, world_size = load_fsdp_config(actor_dir)
    output_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Actor checkpoint: {actor_dir}")
    print(f"FSDP version/world size: {fsdp_version}/{world_size}")
    print(f"Output directory: {output_dir}")

    state_dict = extract_lora_state(actor_dir, world_size, output_dtype)
    rank, target_modules = inspect_adapter_state(state_dict)
    print(f"Extracted {len(state_dict)} tensors; rank={rank}; targets={','.join(target_modules)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.export-", dir=output_dir.parent))
    try:
        config_data = write_adapter_config(
            staging_dir=staging_dir,
            adapter_config=args.adapter_config,
            base_model=args.base_model,
            rank=rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        save_file(state_dict, staging_dir / "adapter_model.safetensors", metadata={"format": "pt"})
        validate_staged_adapter(staging_dir, rank, set(state_dict), output_dtype)
        publish_adapter(staging_dir, output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    weights_path = output_dir / "adapter_model.safetensors"
    print("Export complete")
    print(f"  adapter_config: {output_dir / 'adapter_config.json'}")
    print(f"  adapter_weights: {weights_path} ({weights_path.stat().st_size / 1024**2:.2f} MiB)")
    print(f"  base_model: {config_data.get('base_model_name_or_path')}")


if __name__ == "__main__":
    main()
