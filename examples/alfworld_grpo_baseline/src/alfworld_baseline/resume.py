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
