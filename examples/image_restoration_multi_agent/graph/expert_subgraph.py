"""Reusable fixed-identity expert subgraph builder."""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportArgumentType=false

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from schemas import ExpertName

from .nodes import RestorationGraphNodes
from .routers import (
    route_action,
    route_after_feedback,
    route_before_decision,
    route_evaluation,
    route_restoration,
)
from .state import RestorationGraphState


def _no_op(state: RestorationGraphState) -> RestorationGraphState:
    del state
    return {}


def build_expert_subgraph(
    nodes: RestorationGraphNodes,
    expert_name: ExpertName,
    *,
    max_steps: int,
) -> Any:
    """Compile one subgraph bound to exactly one expert identity."""

    builder = StateGraph(RestorationGraphState)
    builder.add_node("precheck", _no_op)
    builder.add_node("decide_action", nodes.decide_action(expert_name))
    builder.add_node("validate_action", nodes.validate_action)
    builder.add_node("record_stop", nodes.record_stop)
    builder.add_node("execute_restoration", nodes.execute_restoration)
    builder.add_node("record_worker_failure", nodes.record_worker_failure)
    builder.add_node("score_candidate", nodes.score_candidate)
    builder.add_node("record_evaluation_failure", nodes.record_evaluation_failure)
    builder.add_node("commit_successful_step", nodes.commit_successful_step)
    builder.add_node("record_max_steps", nodes.record_max_steps)

    builder.add_edge(START, "precheck")
    builder.add_conditional_edges(
        "precheck",
        partial(route_before_decision, max_steps=max_steps),
        {"decide": "decide_action", "max_steps": "record_max_steps", "finish": END},
    )
    builder.add_edge("decide_action", "validate_action")
    builder.add_conditional_edges(
        "validate_action",
        route_action,
        {"finish": END, "stop": "record_stop", "restore": "execute_restoration"},
    )
    builder.add_edge("record_stop", END)
    builder.add_conditional_edges(
        "execute_restoration",
        route_restoration,
        {"failed": "record_worker_failure", "score": "score_candidate"},
    )
    builder.add_conditional_edges(
        "score_candidate",
        route_evaluation,
        {"failed": "record_evaluation_failure", "commit": "commit_successful_step"},
    )
    builder.add_conditional_edges(
        "record_worker_failure",
        route_after_feedback,
        {"continue": "precheck", "finish": END},
    )
    builder.add_conditional_edges(
        "record_evaluation_failure",
        route_after_feedback,
        {"continue": "precheck", "finish": END},
    )
    builder.add_conditional_edges(
        "commit_successful_step",
        route_after_feedback,
        {"continue": "precheck", "finish": END},
    )
    builder.add_edge("record_max_steps", END)
    return builder.compile(name=expert_name.value)
