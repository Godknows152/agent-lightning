"""ALFWorld-only rollout metrics for the isolated trainer entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np


_PENALTY_COUNT_FIELDS = {
    "alfworld_no_tool_call_penalty_count": "alfworld_penalty/no_action_count",
    "alfworld_invalid_tool_call_penalty_count": "alfworld_penalty/invalid_action_count",
    "alfworld_repeated_action_penalty_count": "alfworld_penalty/repeated_action_count",
}


def _sum_count_field(value: Any) -> int:
    """Sum per-trajectory scalar counters from a VERL non-tensor field."""
    if value is None:
        return 0
    values = np.asarray(value, dtype=object).reshape(-1)
    total = 0
    for item in values:
        if item is None:
            continue
        try:
            total += int(item)
        except (TypeError, ValueError):
            # Be tolerant of a nested/object value produced by an older batch
            # collation path while keeping malformed telemetry non-fatal.
            nested = np.asarray(item, dtype=object).reshape(-1)
            total += sum(int(nested_item) for nested_item in nested if nested_item is not None)
    return total


def compute_alfworld_rollout_metrics(batch: Any) -> dict[str, int | float]:
    """Report terminal reasons, penalty counts, and valid action counts.

    The agent loop records one counter per trajectory for each category. The
    repeated-action counter counts only consecutive valid repeats. The
    trainer aggregates those counters over the rollout batch for SwanLab.
    Legacy tool-call input fields are retained for stored trajectory compatibility,
    but public metrics use action names only, without duplicate aliases. No
    generic restoration penalty series are emitted here.
    """

    non_tensor_batch = batch.non_tensor_batch
    reasons = np.asarray(non_tensor_batch.get("alfworld_terminal_reason", []), dtype=object).reshape(-1)
    # Keep the five buckets exhaustive for older/malformed batches.
    reasons = np.where(np.isin(reasons, ["success", "environment_timeout", "decision_limit", "no_tool_call", "env_failure"]), reasons, "env_failure")
    termination_counts = {
        "success": int(np.count_nonzero(reasons == "success")),
        "environment_timeout": int(np.count_nonzero(reasons == "environment_timeout")),
        "decision_limit": int(np.count_nonzero(reasons == "decision_limit")),
        "no_tool_call": int(np.count_nonzero(reasons == "no_tool_call")),
        "env_failure": int(np.count_nonzero(reasons == "env_failure")),
    }
    result: dict[str, int | float] = {
        f"alfworld_termination/{name}_count": count
        for name, count in termination_counts.items()
    }
    result["alfworld_termination/total_trajectories"] = int(len(reasons))

    for field, metric_name in _PENALTY_COUNT_FIELDS.items():
        result[metric_name] = _sum_count_field(non_tensor_batch.get(field))
    categories = np.asarray(non_tensor_batch.get("alfworld_no_action_category", []), dtype=object).reshape(-1)
    if len(categories) != len(reasons):
        categories = np.full(len(reasons), "", dtype=object)
    no_action_total = _sum_count_field(non_tensor_batch.get("alfworld_no_tool_call_penalty_count"))
    category_names = ("thinking_unclosed", "tool_call_format", "overlong_thinking", "other")
    category_counts = {
        name: int(np.count_nonzero(categories == name))
        for name in category_names
    }
    # Older batches may lack the detail field; keep the four categories
    # exhaustive without double-counting an already classified trajectory.
    classified = sum(category_counts.values())
    category_counts["other"] += no_action_total - classified
    result.update({
        f"alfworld_penalty/no_action/{name}_count": count
        for name, count in category_counts.items()
    })

    valid_tool_calls = np.asarray(
        non_tensor_batch.get("alfworld_valid_tool_call_count", 0), dtype=np.int64
    ).reshape(-1)
    result.update(
        {
            "alfworld/valid_action_count/min": int(valid_tool_calls.min()),
            "alfworld/valid_action_count/max": int(valid_tool_calls.max()),
            "alfworld/valid_action_count/mean": float(valid_tool_calls.mean()),
        }
    )
    return result
