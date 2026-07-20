"""Serializable LangGraph state for one image-restoration trajectory."""

from __future__ import annotations

from typing import Any, TypedDict, cast

GRAPH_SCHEMA_VERSION = 1


class RestorationGraphState(TypedDict, total=False):
    """JSON-compatible channels shared by the parent graph and expert subgraphs."""

    schema_version: int
    task: dict[str, Any]
    trajectory_id: str
    input_image: str
    output_dir: str
    diagnosis: dict[str, Any]
    original_evaluation: dict[str, Any]
    trajectory: dict[str, Any]
    pending_decision: dict[str, Any] | None
    pending_action: str | None
    pending_restoration: dict[str, Any] | None
    pending_evaluation: dict[str, Any] | None
    pending_error: str | None
    result: dict[str, Any]


def require_mapping(state: RestorationGraphState, key: str) -> dict[str, Any]:
    """Read one required mapping channel with a runtime invariant check."""

    value: object = state.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"graph state is missing mapping channel: {key}")
    return cast(dict[str, Any], value)


def require_text(state: RestorationGraphState, key: str) -> str:
    """Read one required text channel with a runtime invariant check."""

    value: object = state.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"graph state is missing text channel: {key}")
    return value
