"""GiGPO-repository GRPO scores: episode outcome plus only this decision's cost."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STEP_PENALTY = -0.2
STEP_PENALTIES = {
    "none": 0.0,
    "no_action": STEP_PENALTY,
    "invalid_action": STEP_PENALTY,
    "repeated_action": 0.0,
}
PENALTY_COUNTERS = {
    "no_action": "alfworld_no_tool_call_penalty_count",
    "invalid_action": "alfworld_invalid_tool_call_penalty_count",
    "repeated_action": "alfworld_repeated_action_penalty_count",
}


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> float:
    """Score one decision; never sum penalties from other decisions.

    Missing and invalid actions cost -0.2, including in successful episodes.
    Repeated valid actions remain identifiable in telemetry but cost zero.
    """
    if data_source != "alfworld":
        raise ValueError(
            f"ALFWorld step reward received unexpected data_source={data_source!r}"
        )
    if extra_info is None or "alfworld_step_penalty_kind" not in extra_info:
        raise ValueError("Step GRPO requires the decision's own penalty metadata")
    reason = extra_info.get("alfworld_terminal_reason")
    if reason not in {
        "success",
        "environment_timeout",
        "decision_limit",
        "env_failure",
        "no_tool_call",
    }:
        raise ValueError(f"Step GRPO requires a completed episode, got {reason!r}")
    kind = extra_info["alfworld_step_penalty_kind"]
    if kind not in {"none", *PENALTY_COUNTERS}:
        raise ValueError(f"Unknown ALFWorld step penalty kind: {kind!r}")
    outcome = 10.0 if reason == "success" else 0.0
    return outcome + STEP_PENALTIES[kind]
