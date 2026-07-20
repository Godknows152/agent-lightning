"""Resolve standardized old-VERL output and SwanLab experiment names.

CLI usage:
    python resolve_training_run_name.py --expert rain --output-root /path/to/outputs

The command prints four tab-separated fields for the Bash launcher:
experiment name, output directory, SwanLab log directory, and the checkpoint
path selected for resume (or ``-`` for a fresh run).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunNaming:
    """Resolved names and paths for one training launch."""

    experiment_name: str
    output_dir: Path
    swanlab_log_dir: Path
    resume_from_path: Path | None


def _checkpoint_step(path: Path) -> int | None:
    prefix = "global_step_"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    try:
        return int(path.name.removeprefix(prefix))
    except ValueError:
        return None


def find_latest_checkpoint(path: Path) -> Path | None:
    """Return the latest valid ``global_step_*`` checkpoint under ``path``."""

    path = path.expanduser()
    if _checkpoint_step(path) is not None:
        return path.resolve()
    if not path.is_dir():
        return None

    tracker_path = path / "latest_checkpointed_iteration.txt"
    if tracker_path.is_file():
        try:
            tracked_step = int(tracker_path.read_text(encoding="utf-8").strip())
        except ValueError:
            tracked_step = -1
        tracked_checkpoint = path / f"global_step_{tracked_step}"
        if tracked_step >= 0 and tracked_checkpoint.is_dir():
            return tracked_checkpoint.resolve()

    checkpoints = [(step, child) for child in path.iterdir() if (step := _checkpoint_step(child)) is not None]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1].resolve()


def resolve_run_naming(
    *,
    expert: str,
    output_root: Path,
    now: datetime | None = None,
    resume_mode: str = "auto",
    resume_from_path: Path | None = None,
    experiment_name: str | None = None,
    output_dir: Path | None = None,
) -> RunNaming:
    """Resolve the shared output directory and SwanLab experiment name."""

    if resume_mode not in {"auto", "disable", "resume_path"}:
        raise ValueError(f"Unsupported resume mode: {resume_mode!r}")

    current_time = now or datetime.now()
    base_name = f"{expert}_{current_time:%m%d}"
    output_root = output_root.expanduser().resolve()
    explicit_output_dir = output_dir.expanduser().resolve() if output_dir is not None else None
    fresh_output_dir = explicit_output_dir or output_root / base_name
    continuation_output_dir = explicit_output_dir or output_root / f"{base_name}_续"

    checkpoint: Path | None = None
    if resume_from_path is not None:
        checkpoint = find_latest_checkpoint(resume_from_path)
        if checkpoint is None:
            raise ValueError(f"No valid checkpoint found at: {resume_from_path}")
    elif resume_mode == "resume_path":
        raise ValueError("resume_mode='resume_path' requires resume_from_path")
    elif resume_mode == "auto":
        # Prefer an already-created continuation directory when restarting a
        # continuation, then fall back to today's fresh-run directory.
        checkpoint = find_latest_checkpoint(continuation_output_dir)
        if checkpoint is None:
            checkpoint = find_latest_checkpoint(fresh_output_dir)

    is_continuation = checkpoint is not None
    resolved_name = experiment_name or f"{base_name}{'_续' if is_continuation else ''}"
    resolved_output_dir = explicit_output_dir or output_root / resolved_name
    return RunNaming(
        experiment_name=resolved_name,
        output_dir=resolved_output_dir,
        swanlab_log_dir=resolved_output_dir / "swanlab",
        resume_from_path=checkpoint,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=("fog", "low_light", "rain", "snow"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume-mode", choices=("auto", "disable", "resume_path"), default="auto")
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        naming = resolve_run_naming(
            expert=args.expert,
            output_root=args.output_root,
            resume_mode=args.resume_mode,
            resume_from_path=args.resume_from_path,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    fields = (
        naming.experiment_name,
        str(naming.output_dir),
        str(naming.swanlab_log_dir),
        str(naming.resume_from_path) if naming.resume_from_path is not None else "-",
    )
    if any("\t" in field or "\n" in field for field in fields):
        raise ValueError("Resolved run names and paths must not contain tabs or newlines")
    print("\t".join(fields))


if __name__ == "__main__":
    main()
