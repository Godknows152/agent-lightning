"""ALFWorld-only rollout metrics for the isolated trainer entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np


_PENALTY_COUNT_FIELDS = {
    "alfworld_no_tool_call_penalty_count": "alfworld_penalty/no_tool_call_count",
    "alfworld_invalid_tool_call_penalty_count": "alfworld_penalty/invalid_tool_call_count",
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
    """Report terminal reasons, penalty counts, and valid tool-call counts.

    The agent loop records one counter per trajectory for each category. The
    trainer aggregates those counters over the rollout batch for SwanLab. No
    generic restoration penalty series are emitted here.
    """

    non_tensor_batch = batch.non_tensor_batch
    reasons = non_tensor_batch.get("alfworld_terminal_reason")
    if reasons is None:
        done_count = 0
        max_steps_count = 0
    else:
        reasons = np.asarray(reasons, dtype=object).reshape(-1)
        done_count = int(np.count_nonzero(reasons == "done"))
        max_steps_count = int(np.count_nonzero(reasons == "max_steps"))

    result: dict[str, int | float] = {
        "alfworld_termination/done_count": done_count,
        "alfworld_termination/max_steps_count": max_steps_count,
    }
    for field, metric_name in _PENALTY_COUNT_FIELDS.items():
        result[metric_name] = _sum_count_field(non_tensor_batch.get(field))

    valid_tool_calls = np.asarray(
        non_tensor_batch.get("alfworld_valid_tool_call_count", 0), dtype=np.int64
    ).reshape(-1)
    result.update(
        {
            "alfworld/valid_tool_call_count/min": int(valid_tool_calls.min()),
            "alfworld/valid_tool_call_count/max": int(valid_tool_calls.max()),
            "alfworld/valid_tool_call_count/mean": float(valid_tool_calls.mean()),
        }
    )
    return result
