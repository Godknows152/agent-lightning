"""Pure conditional routing functions for the restoration state graph."""

from __future__ import annotations

from typing import Literal

from schemas import ExecutionStatus, ExpertName, RestorationResult, RestorationTrajectoryState
from tool_registry import STOP_ACTION

from .state import RestorationGraphState, require_mapping


def _trajectory(state: RestorationGraphState) -> RestorationTrajectoryState:
    return RestorationTrajectoryState.model_validate(require_mapping(state, "trajectory"))


def route_expert(state: RestorationGraphState) -> ExpertName:
    """Route once from the diagnosis result to one stable expert subgraph."""

    return _trajectory(state).expert_name


def route_before_decision(
    state: RestorationGraphState,
    *,
    max_steps: int,
) -> Literal["decide", "max_steps", "finish"]:
    """Enforce the business step limit before requesting another action."""

    trajectory = _trajectory(state)
    if trajectory.terminated:
        return "finish"
    if len(trajectory.steps) >= max_steps:
        return "max_steps"
    return "decide"


def route_action(state: RestorationGraphState) -> Literal["finish", "stop", "restore"]:
    """Dispatch a validated action or finish an invalid decision."""

    if _trajectory(state).terminated:
        return "finish"
    return "stop" if state.get("pending_action") == STOP_ACTION else "restore"


def route_restoration(state: RestorationGraphState) -> Literal["failed", "score"]:
    """Send successful worker results to IQA and failures to recording."""

    payload = state.get("pending_restoration")
    if payload is None:
        raise ValueError("route_restoration requires pending_restoration")
    result = RestorationResult.model_validate(payload)
    if result.status == ExecutionStatus.SUCCESS and result.output_path is not None:
        return "score"
    return "failed"


def route_evaluation(state: RestorationGraphState) -> Literal["failed", "commit"]:
    """Accept only successful IQA results."""

    payload = state.get("pending_evaluation")
    if payload is None:
        raise ValueError("route_evaluation requires pending_evaluation")
    status = payload.get("status")
    return "commit" if status == ExecutionStatus.SUCCESS.value else "failed"


def route_after_feedback(state: RestorationGraphState) -> Literal["continue", "finish"]:
    """Loop only while the same expert trajectory remains active."""

    return "finish" if _trajectory(state).terminated else "continue"
