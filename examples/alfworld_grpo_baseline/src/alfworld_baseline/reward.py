"""ALFWorld reward adapter for the isolated old-VERL baseline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REPEATED_ACTION_PENALTY = -0.1
REPEATED_ACTION_PENALTY_INCREMENT = 0.05


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

    ALFWorld emits sparse environment rewards (10 on a solved task and 0 otherwise).
    Each decision without a parseable action appends a ``-2`` penalty and continues
    within the decision budget, retaining previous rewards. Complete but invalid XML
    decisions receive ``-2`` and may continue. Repeated valid actions are counted
    across the whole trajectory. The k-th repeated action costs
    ``0.1 + 0.05 * (k - 1)``, summed over all repeats regardless of action identity.
    This aggregate repeated-action penalty is gated off when the trajectory receives
    the terminal success reward.
    """
    if data_source != "alfworld":
        raise ValueError(f"ALFWorld reward received unexpected data_source={data_source!r}")
    info: Mapping[str, Any] = extra_info or {}
    rewards = info.get("tool_rewards", [])
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)):
        base_score = float(sum(float(item) for item in rewards))
        reward_values = [float(item) for item in rewards]
    else:
        reward = _first(rewards)
        base_score = float(reward or 0.0)
        reward_values = [base_score]

    terminal_reason = _first(info.get("alfworld_terminal_reason"))
    success = terminal_reason == "success" or any(value >= 10.0 for value in reward_values)
    repeat_count = _first(info.get("alfworld_repeated_action_penalty_count", 0))
    try:
        repeat_count = max(0, int(repeat_count or 0))
    except (TypeError, ValueError):
        repeat_count = 0

    # Apply the aggregate penalty exactly once, after the whole trajectory is
    # available. Successful trajectories are exempt from repeated-action costs.
    repeated_penalty = 0.0 if success else (
        REPEATED_ACTION_PENALTY * repeat_count
        - REPEATED_ACTION_PENALTY_INCREMENT * repeat_count * (repeat_count - 1) / 2
    )
    return base_score + repeated_penalty
