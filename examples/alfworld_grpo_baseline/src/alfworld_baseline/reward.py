"""ALFWorld reward adapter for the isolated old-VERL baseline."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REPEATED_ACTION_PENALTY = -0.1


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
    Each decision without a parseable action appends a ``-0.2`` penalty and continues
    within the decision budget, retaining previous rewards. Complete but invalid XML
    decisions receive ``-0.2`` and may continue. Both penalties accumulate into
    one episode score. Each valid execution of a command already executed in
    this trajectory costs -0.1, whether consecutive or not. The first execution
    is free. Successful trajectories are exempt from repeated-action costs;
    their missing/invalid-action penalties still apply.
    Opt-in thinking/action consistency and branch-entropy eligibility bonuses are accumulated once per decision
    by the agent loop and added once to this episode score, on success or failure.
    """
    if data_source != "alfworld":
        raise ValueError(f"ALFWorld reward received unexpected data_source={data_source!r}")
    info: Mapping[str, Any] = extra_info or {}
    rewards = info.get("tool_rewards", [])
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)):
        base_score = float(sum(float(item) for item in rewards))
    else:
        reward = _first(rewards)
        base_score = float(reward or 0.0)
    repeat_count = _first(info.get("alfworld_repeated_action_penalty_count", 0))
    try:
        repeat_count = max(0, int(repeat_count or 0))
    except (TypeError, ValueError):
        repeat_count = 0

    # Apply the aggregate penalty exactly once, after the whole trajectory is
    # available. Per-step tool_rewards do not contain repeated-action costs.
    success = _first(info.get("alfworld_terminal_reason")) == "success"
    repeated_penalty = 0.0 if success else REPEATED_ACTION_PENALTY * repeat_count
    thinking_bonus = float(_first(info.get("alfworld_thinking_action_reward", 0.0)) or 0.0)
    branch_bonus = float(_first(info.get("alfworld_branch_entropy_eligible_reward", 0.0)) or 0.0)
    return base_score + repeated_penalty + thinking_bonus + branch_bonus
