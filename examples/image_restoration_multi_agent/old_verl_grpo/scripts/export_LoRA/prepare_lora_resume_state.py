"""Reshard FSDP1 LoRA Adam states while loading model weights from a PEFT adapter.

Supports the per-trainable-leaf FSDP wrapping used by this restoration backend.
Rejects mixed/uneven parameter shards rather than guessing their layout.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

from export_fsdp_checkpoint_to_lora import load_fsdp_config, resolve_actor_dir


def prepare_resume_state(checkpoint: Path, destination: Path, target_world_size: int) -> dict:
    actor = resolve_actor_dir(checkpoint)
    source = actor.parent
    fsdp_version, source_world_size = load_fsdp_config(actor)
    if fsdp_version != 1 or target_world_size < 1 or source_world_size % target_world_size:
        raise ValueError("Only FSDP1 consolidation to a divisor of the source world size is supported")
    if destination.exists() or destination.name != source.name:
        raise ValueError("Destination must be a new directory with the same global_step_N name")
    states = [
        torch.load(actor / f"optim_world_size_{source_world_size}_rank_{rank}.pt", weights_only=False,
                   map_location="cpu", mmap=True)
        for rank in range(source_world_size)
    ]
    model = torch.load(actor / f"model_world_size_{source_world_size}_rank_0.pt", weights_only=False,
                       map_location="cpu", mmap=True)
    lora_shapes = {name: tuple(tensor.shape) for name, tensor in model.items() if "lora_" in name}
    del model
    groups = states[0]["param_groups"]
    parameter_ids = [pid for group in groups for pid in group["params"]]
    active_ids = [pid for pid in parameter_ids if states[0]["state"][pid]]
    if len(active_ids) != len(lora_shapes):
        raise ValueError("Optimizer parameters do not match the per-leaf LoRA wrapping")
    for state in states:
        if state["param_groups"] != groups or state["state"].keys() != states[0]["state"].keys():
            raise ValueError("Optimizer parameter order/groups differ across source ranks")

    outputs = [{"state": {}, "param_groups": copy.deepcopy(groups)} for _ in range(target_world_size)]
    parameter_map = dict(zip(active_ids, lora_shapes, strict=True))
    for pid in parameter_ids:
        rank_states = [state["state"][pid] for state in states]
        if any(value.keys() != rank_states[0].keys() for value in rank_states):
            raise ValueError(f"Optimizer slots differ for parameter {pid}")
        for output in outputs:
            output["state"][pid] = {}
        for slot, value in rank_states[0].items():
            values = [state[slot] for state in rank_states]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Unsupported optimizer slot {pid}/{slot}")
            if value.ndim == 0:
                if any(not torch.equal(value, other) for other in values):
                    raise ValueError(f"Scalar optimizer state differs for {pid}/{slot}")
                pieces = [value.clone() for _ in outputs]
            else:
                if slot not in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} or value.ndim != 1:
                    raise ValueError(f"Unsupported optimizer moment layout for {pid}/{slot}")
                expected = torch.Size(lora_shapes[parameter_map[pid]]).numel()
                if any(other.shape != value.shape or other.dtype != value.dtype for other in values):
                    raise ValueError(f"Uneven optimizer shards for {pid}/{slot}")
                if value.numel() * source_world_size != expected or expected % target_world_size:
                    raise ValueError(f"Optimizer shard size disagrees with LoRA parameter {pid}")
                full = torch.cat(values)
                pieces = [piece.clone() for piece in full.chunk(target_world_size)]
                if not torch.equal(torch.cat(pieces), full):
                    raise ValueError(f"Optimizer moment round-trip failed for {pid}/{slot}")
            for output, piece in zip(outputs, pieces, strict=True):
                output["state"][pid][slot] = piece

    destination.mkdir(parents=True)
    dest_actor = destination / "actor"
    dest_actor.mkdir()
    for rank, output in enumerate(outputs):
        torch.save(output, dest_actor / f"optim_world_size_{target_world_size}_rank_{rank}.pt")
        shutil.copy2(actor / f"extra_state_world_size_{source_world_size}_rank_{rank}.pt",
                     dest_actor / f"extra_state_world_size_{target_world_size}_rank_{rank}.pt")
    shutil.copy2(source / "data.pt", destination / "data.pt")
    metadata = {
        "source_checkpoint": str(source), "source_world_size": source_world_size,
        "target_world_size": target_world_size, "lora_parameter_count": len(active_ids),
        "optimizer_lr": [group["lr"] for group in groups],
        "load_contents": ["optimizer", "extra"],
        "model_weights": "Load the exported PEFT adapter via actor_rollout_ref.model.lora_adapter_path",
        "parameter_names": parameter_map,
    }
    (destination / "conversion_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    metadata = prepare_resume_state(args.checkpoint.resolve(), args.destination.resolve(), args.world_size)
    print(json.dumps({k: v for k, v in metadata.items() if k != "parameter_names"}, ensure_ascii=False, indent=2))
