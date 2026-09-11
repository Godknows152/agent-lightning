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
    0 otherwise). The first decision without a parseable action ends the
    trajectory and appends a one-time ``-5`` penalty, retaining previous rewards.
    Complete but invalid XML decisions receive ``-0.1`` and may continue.
    Qwen3.5 v7 validates post-thinking XML with one tool and one action parameter;
    legacy profiles retain their own parsers. The returned score is therefore the native
    environment reward plus the two protocol penalties and a mutually
    exclusive repeated-action penalty for valid decisions: starting from the
    second consecutive execution of an identical command, each repeat costs
    0.1. A different command or a protocol failure breaks the streak;
    nonconsecutive repetitions are not penalized.
    """
    if data_source != "alfworld":
        raise ValueError(f"ALFWorld reward received unexpected data_source={data_source!r}")
    info: Mapping[str, Any] = extra_info or {}
    rewards = info.get("tool_rewards", [])
    if isinstance(rewards, Sequence) and not isinstance(rewards, (str, bytes, bytearray)):
        return float(sum(float(item) for item in rewards))
    reward = _first(rewards)
    return float(reward or 0.0)
