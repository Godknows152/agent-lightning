"""Copy-based deterministic worker for workflow validation."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from schemas import ExecutionStatus, RestorationResult


class CopyRestorationWorker:
    """Represent a restoration action by copying its input to a step-specific path."""

    def __init__(self, fail_actions: set[str] | None = None) -> None:
        self.fail_actions = fail_actions or set()

    def restore(self, action: str, input_path: str, output_dir: str, step_index: int) -> RestorationResult:
        """Copy the input image or return a configured deterministic failure."""

        started = time.perf_counter()
        source = Path(input_path)
        if action in self.fail_actions:
            return RestorationResult(
                status=ExecutionStatus.FAILED,
                worker=action,
                input_path=str(source),
                output_path=None,
                latency_seconds=time.perf_counter() - started,
                error=f"scripted worker failure for action {action}",
            )
        if not source.is_file():
            return RestorationResult(
                status=ExecutionStatus.FAILED,
                worker=action,
                input_path=str(source),
                output_path=None,
                latency_seconds=time.perf_counter() - started,
                error=f"input image does not exist: {source}",
            )

        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix or ".img"
        destination = destination_dir / f"step_{step_index:03d}_{action}{suffix}"
        shutil.copy2(source, destination)
        return RestorationResult(
            status=ExecutionStatus.SUCCESS,
            worker=action,
            input_path=str(source),
            output_path=str(destination),
            latency_seconds=time.perf_counter() - started,
            error=None,
        )
