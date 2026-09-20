"""Explicit ALFWorld prompt profiles keyed by training prompt contract."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType

_MODULES = {
    "qwen25": "alfworld_baseline.prompts_gigpo",
    "qwen35": "alfworld_baseline.prompts_gigpo",
    "qwen35_v7": "alfworld_baseline.prompts_qwen35",
    "gigpo": "alfworld_baseline.prompts_gigpo",
}


def get_prompt_profile(name: str) -> ModuleType:
    """Load one supported prompt profile without falling back silently."""

    try:
        return import_module(_MODULES[name])
    except KeyError as exc:
        raise ValueError(f"Unknown ALFWorld prompt profile: {name}. Expected one of {sorted(_MODULES)}") from exc
