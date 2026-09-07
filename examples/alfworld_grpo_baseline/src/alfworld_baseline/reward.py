"""ALFWorld reward adapter for the isolated old-VERL baseline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _first(value: Any) -> Any:
    """Unwrap the one-item arrays used by VERL non-tensor fields."""
    while isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None
        value = value[0]
    return value


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> float:
    """Return the episode reward emitted by ALFWorldTool.

    ALFWorld emits sparse environment rewards (normally 1 on a solved task and
    0 otherwise). The agent loop additionally appends exactly one ``-0.1``
    penalty for each decision that either has no parsed tool call or contains a
    parsed but invalid tool call. The returned score is therefore the native
    environment reward plus the two allowed protocol penalties.
    """
    if data_source != "alfworld":
        raise ValueError(f"ALFWorld reward received unexpected data_source={data_source!r}")
    info: Mapping[str, Any] = extra_info or {}
    rewards = info.get("tool_rewards", [])
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)):
        return float(sum(float(item) for item in rewards))
    reward = _first(rewards)
    return float(reward or 0.0)
