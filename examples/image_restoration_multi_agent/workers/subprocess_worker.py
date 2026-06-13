"""Real restoration worker backed by an isolated `verl` Python process."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from config import SubprocessSettings
from schemas import ExecutionStatus, RestorationResult
from subprocess_utils import parse_result_json, resolve_python_command, validate_image_file
from tool_registry import ToolRegistry


class SubprocessRestorationWorker:
    """Invoke one registered model in the isolated restoration environment."""

    def __init__(self, settings: SubprocessSettings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self.python_command = resolve_python_command(settings.environment_name, settings.python_executable)
        self.entrypoint = Path(settings.entrypoint).expanduser().resolve()
        self.external_tools_root = Path(settings.external_tools_root).expanduser().resolve()

    def restore(self, action: str, input_path: str, output_dir: str, step_index: int) -> RestorationResult:
        """Run one action and publish its output only after image validation."""

        started = time.perf_counter()
        source = Path(input_path).expanduser().resolve()
        destination_dir = Path(output_dir).expanduser().resolve()
        destination = destination_dir / f"step_{step_index:03d}_{action}.png"
        partial = destination_dir / f".step_{step_index:03d}_{action}.partial.png"
        try:
            validate_image_file(source)
            tool = self.registry.get_tool(action)
            if tool.runtime is None:
                raise ValueError(f"tool {action} has no real runtime configuration")
            if not self.entrypoint.is_file():
                raise FileNotFoundError(f"restoration entrypoint does not exist: {self.entrypoint}")
            if not self.external_tools_root.is_dir():
                raise FileNotFoundError(f"external tools directory does not exist: {self.external_tools_root}")
            destination_dir.mkdir(parents=True, exist_ok=True)
            partial.unlink(missing_ok=True)

            command = [
                *self.python_command,
                str(self.entrypoint),
                "--adapter",
                tool.runtime.adapter,
                "--model",
                tool.runtime.model,
                "--external-tools-root",
                str(self.external_tools_root),
                "--input",
                str(source),
                "--output",
                str(partial),
                "--device",
                self.settings.device,
            ]
            if tool.runtime.repo:
                command.extend(["--repo", str((self.external_tools_root / tool.runtime.repo).resolve())])
            if tool.runtime.checkpoint:
                command.extend(["--checkpoint", str((self.external_tools_root / tool.runtime.checkpoint).resolve())])

            environment = os.environ.copy()
            environment.setdefault("HF_HUB_OFFLINE", "1")
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
                env=environment,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"restoration subprocess exited with {completed.returncode}: {detail[-2000:]}")
            process_result = parse_result_json(completed.stdout)
            if process_result.get("status") != "success":
                raise RuntimeError(f"restoration subprocess reported failure: {process_result}")
            validate_image_file(partial)
            os.replace(partial, destination)
            process_result["output_path"] = str(destination)
            return RestorationResult(
                status=ExecutionStatus.SUCCESS,
                worker=action,
                input_path=str(source),
                output_path=str(destination),
                latency_seconds=time.perf_counter() - started,
                error=None,
                metadata={
                    **process_result,
                    "environment_name": self.settings.environment_name,
                    "device": self.settings.device,
                    "checkpoint": tool.runtime.checkpoint,
                },
            )
        except Exception as error:
            partial.unlink(missing_ok=True)
            return RestorationResult(
                status=ExecutionStatus.FAILED,
                worker=action,
                input_path=str(source),
                output_path=None,
                latency_seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
                metadata={
                    "environment_name": self.settings.environment_name,
                    "device": self.settings.device,
                },
            )
