"""Validate a completed two-GPU checkpoint and its original SwanLab identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_resume_checkpoint(output_dir: str, experiment_name: str) -> str | None:
    """Resolve the last published checkpoint, failing closed on missing identity/files."""
    root = Path(output_dir).resolve()
    latest = root / "latest_checkpointed_iteration.txt"
    marker = root / ".swanlab_experiment.json"
    if not latest.exists():
        if marker.exists() or any(root.glob("global_step_*")):
            raise ValueError("Existing run has no published checkpoint; refusing an implicit fresh restart.")
        return None
    step = int(latest.read_text().strip())
    if step < 1:
        raise ValueError("Checkpoint step must be positive")
    checkpoint = root / f"global_step_{step}"
    required = [checkpoint / "data.pt", checkpoint / "actor/fsdp_config.json"]
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            required.append(checkpoint / "actor" / f"{kind}_world_size_2_rank_{rank}.pt")
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    identity = json.loads(marker.read_text())
    if identity.get("experiment_name") != experiment_name:
        raise ValueError("Checkpoint SwanLab experiment name differs from launcher configuration")
    if not isinstance(identity.get("run_id"), str) or not identity["run_id"].strip():
        raise ValueError("Missing SwanLab run ID; refusing to create a replacement experiment")
    return str(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("experiment_name")
    args = parser.parse_args()
    print(resolve_resume_checkpoint(args.output_dir, args.experiment_name) or "")


def validate_native_resume(config) -> None:
    """Validate published checkpoint files before Ray allocates any workers."""
    trainer = config.trainer
    root = Path(trainer.default_local_dir).resolve()
    mode = trainer.resume_mode
    if mode == "disable":
        if (root / "latest_checkpointed_iteration.txt").exists():
            raise ValueError("resume_mode=disable requires a new output directory")
        return
    if mode == "auto":
        latest = root / "latest_checkpointed_iteration.txt"
        if not latest.exists():
            if any(root.glob("global_step_*")) or (root / ".swanlab_experiment.json").exists():
                raise ValueError("Existing run has no published checkpoint; refusing an implicit fresh restart")
            return
        step = int(latest.read_text().strip())
        if step < 1:
            raise ValueError("Checkpoint step must be positive")
        checkpoint = root / f"global_step_{step}"
    elif mode == "resume_path":
        checkpoint = Path(trainer.resume_from_path).resolve()
        if not checkpoint.name.startswith("global_step_") or not checkpoint.name[12:].isdigit():
            raise ValueError("resume_from_path must name a global_step_<number> checkpoint")
    else:
        raise ValueError(f"Unsupported resume_mode: {mode}")
    size = int(trainer.n_gpus_per_node) * int(trainer.nnodes)
    required = [checkpoint / "data.pt", checkpoint / "actor/fsdp_config.json"]
    required += [checkpoint / "actor" / f"{kind}_world_size_{size}_rank_{rank}.pt"
                 for rank in range(size) for kind in ("model", "optim", "extra_state")]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete checkpoint: {missing}")
    if "swanlab" in trainer.logger:
        marker = checkpoint.parent / ".swanlab_experiment.json"
        if not marker.exists():
            raise ValueError("Missing checkpoint SwanLab identity")
        identity = json.loads(marker.read_text())
        if identity.get("experiment_name") != trainer.experiment_name or not identity.get("run_id"):
            raise ValueError("Checkpoint SwanLab identity differs from launcher configuration")
        output_marker = root / marker.name
        if not output_marker.exists() or json.loads(output_marker.read_text()) != identity:
            raise ValueError("Checkpoint and output directory must have the same SwanLab identity")
