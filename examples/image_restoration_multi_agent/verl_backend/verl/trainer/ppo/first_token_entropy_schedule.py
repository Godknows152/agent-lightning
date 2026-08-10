"""Schedules for the legal-action first-token entropy coefficient."""

from __future__ import annotations

import math
from typing import Optional

SUPPORTED_SCHEDULES = frozenset({"constant", "wsd_cosine"})


def get_first_token_entropy_coeff(
    *,
    step: int,
    total_steps: int,
    start: float,
    end: Optional[float] = None,
    schedule: str = "constant",
    ramp_ratio: float = 0.05,
    stable_end_ratio: float = 0.20,
    decay_end_ratio: float = 0.85,
) -> float:
    """Return the coefficient for one absolute training step.

    ``wsd_cosine`` follows a short ramp, a high-exploration stable phase, and
    a half-cosine decay to ``end``. Ratios are measured against the complete
    training run, so checkpoint resumes continue at the same schedule point.
    ``constant`` preserves the legacy fixed-coefficient behavior.
    """

    if not isinstance(step, int):
        raise TypeError(f"step must be an int, got {type(step).__name__}")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(f"unsupported first-token entropy schedule: {schedule!r}")

    for name, value in (
        ("start", start),
        ("ramp_ratio", ramp_ratio),
        ("stable_end_ratio", stable_end_ratio),
        ("decay_end_ratio", decay_end_ratio),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if start < 0:
        raise ValueError("start must be non-negative")

    resolved_end = start if end is None else float(end)
    if not math.isfinite(resolved_end) or resolved_end < 0:
        raise ValueError("end must be finite and non-negative")
    if schedule == "wsd_cosine" and resolved_end > start:
        raise ValueError("end must not exceed start for wsd_cosine")
    if not 0.0 <= ramp_ratio <= stable_end_ratio <= decay_end_ratio <= 1.0:
        raise ValueError("schedule ratios must satisfy 0 <= ramp_ratio <= stable_end_ratio <= decay_end_ratio <= 1")
    if schedule == "constant" or total_steps == 1:
        return float(start)

    bounded_step = min(max(step, 1), total_steps)
    progress = (bounded_step - 1) / float(total_steps - 1)

    if ramp_ratio > 0.0 and progress < ramp_ratio:
        return float(start * progress / ramp_ratio)
    if progress < stable_end_ratio:
        return float(start)
    if progress < decay_end_ratio:
        decay_progress = (progress - stable_end_ratio) / (decay_end_ratio - stable_end_ratio)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return float(resolved_end + (start - resolved_end) * cosine)
    return float(resolved_end)
