"""ALFWorld-only rollout penalty metrics for the isolated trainer entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np


_REASONS = {
    "no_tool_call": "no_tool_call_count",
    "malformed_tool_call_xml": "malformed_tool_call_count",
    "invalid_json_arguments": "format_error_count",
    "invalid_arguments_schema": "format_error_count",
    "format_error": "format_error_count",
    "invalid_action": "invalid_action_count",
    "invalid_restoration_action": "invalid_action_count",
    "unknown_tool_name": "unknown_tool_count",
}


def compute_alfworld_penalty_metrics(batch: Any) -> dict[str, float | int]:
    """Count explicit per-occurrence ALFWorld protocol penalties in a batch."""

    counts = {name: 0 for name in sorted(set(_REASONS.values()))}
    total_value = 0.0
    records = batch.non_tensor_batch.get("penalty_records")
    if records is not None:
        array = np.asarray(records, dtype=object)
        for trajectory in array.reshape(-1):
            if not isinstance(trajectory, (list, tuple, np.ndarray)):
                continue
            for record in trajectory:
                if not isinstance(record, dict):
                    continue
                reason = str(record.get("reason", ""))
                metric_name = _REASONS.get(reason)
                if metric_name is None:
                    continue
                try:
                    occurrences = max(1, int(record.get("occurrences", 1)))
                    value = float(record.get("value", 0.0))
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    continue
                counts[metric_name] += occurrences
                total_value += value * occurrences

    result: dict[str, float | int] = {
        f"alfworld_penalty/{name}": int(value) for name, value in counts.items()
    }
    result["alfworld_penalty/total_count"] = int(sum(counts.values()))
    result["alfworld_penalty/total_value"] = float(total_value)
    return result
