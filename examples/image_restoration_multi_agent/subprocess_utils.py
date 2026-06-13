"""Helpers for invoking model code in an isolated conda environment."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

RESULT_PREFIX = "RESULT_JSON="


def resolve_python_command(environment_name: str, python_executable: str | None) -> list[str]:
    """Resolve an explicit Python binary or fall back to `conda run`."""

    override = os.getenv("IMAGE_RESTORATION_PYTHON")
    candidate = override or python_executable
    if candidate:
        executable = Path(candidate).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"isolated Python executable does not exist: {executable}")
        return [str(executable)]

    conda_executable = shutil.which("conda")
    if conda_executable is None:
        raise FileNotFoundError("conda is unavailable and no isolated Python executable was configured")
    return [conda_executable, "run", "-n", environment_name, "python"]


def parse_result_json(stdout: str) -> dict[str, Any]:
    """Extract the final machine-readable record from subprocess stdout."""

    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line.removeprefix(RESULT_PREFIX))
            if not isinstance(payload, dict):
                raise ValueError("subprocess result must be a JSON object")
            return cast(dict[str, Any], payload)
    raise ValueError("subprocess did not emit a RESULT_JSON record")


def validate_image_file(path: Path) -> None:
    """Raise when a path is missing, empty, or not a readable RGB image."""

    from PIL import Image

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"image file is missing or empty: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").load()
