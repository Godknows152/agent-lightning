"""Verify that ALFWorld uses one explicitly selected native veRL checkout."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def native_verl_root() -> Path:
    return Path(os.environ.get("ALFWORLD_VERL_ROOT", "/home/LXJ/Python_Projects/verl")).resolve()


def assert_native_verl() -> Path:
    """Reject cached or mixed imports instead of silently testing another backend."""
    expected = native_verl_root() / "verl"
    if not (expected / "__init__.py").is_file():
        raise RuntimeError(f"Native veRL checkout is missing: {expected}")
    importlib.import_module("verl")
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == "verl" or name.startswith("verl.")):
            continue
        paths = list(getattr(module, "__path__", ()))
        filename = getattr(module, "__file__", None)
        if filename:
            paths.append(filename)
        for path in paths:
            if not Path(path).resolve().is_relative_to(expected):
                raise RuntimeError(
                    f"ALFWorld backend mismatch: {name} loaded from {path}; expected {expected}. "
                    "Start a fresh process with ALFWORLD_VERL_ROOT at the front of PYTHONPATH."
                )
    return expected
