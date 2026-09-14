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

    ALFWorld emits sparse environment rewards (10 on a solved task and 0 otherwise).
    The first decision without a parseable action ends the trajectory and appends a
    one-time ``-5`` penalty, retaining previous rewards. Complete but invalid XML
    decisions receive ``-0.1`` and may continue. Repeated valid actions are counted
    across the whole trajectory: every occurrence after the first costs ``0.1``.
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
    repeated_penalty = 0.0 if success else -0.1 * repeat_count
    return base_score + repeated_penalty
