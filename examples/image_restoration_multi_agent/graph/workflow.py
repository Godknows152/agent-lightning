"""Parent LangGraph builder and deterministic invocation facade."""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from schemas import ExpertName, RestorationTask, WorkflowResult

from .expert_subgraph import build_expert_subgraph
from .nodes import RestorationGraphNodes
from .routers import route_expert
from .runtime import RestorationGraphRuntime
from .state import GRAPH_SCHEMA_VERSION, RestorationGraphState


class LangGraphImageRestorationWorkflow:
    """Run one no-model restoration workflow through a compiled StateGraph."""

    def __init__(
        self,
        runtime: RestorationGraphRuntime,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.checkpointer = checkpointer or InMemorySaver()
        self.nodes = RestorationGraphNodes(runtime)
        self.graph = self._build().compile(checkpointer=self.checkpointer, name="image_restoration_workflow")

    def _build(self) -> StateGraph[RestorationGraphState]:
        builder = StateGraph(RestorationGraphState)
        builder.add_node("prepare_input", self.nodes.prepare_input)
        builder.add_node("diagnose", self.nodes.diagnose)
        builder.add_node("score_original", self.nodes.score_original)
        builder.add_node("initialize_trajectory", self.nodes.initialize_trajectory)
        builder.add_node("finalize", self.nodes.finalize)

        for expert_name in ExpertName:
            builder.add_node(
                expert_name.value,
                build_expert_subgraph(
                    self.nodes,
                    expert_name,
                    max_steps=self.runtime.settings.max_steps,
                ),
            )

        builder.add_edge(START, "prepare_input")
        builder.add_edge("prepare_input", "diagnose")
        builder.add_edge("diagnose", "score_original")
        builder.add_edge("score_original", "initialize_trajectory")
        builder.add_conditional_edges(
            "initialize_trajectory",
            route_expert,
            {expert_name: expert_name.value for expert_name in ExpertName},
        )
        for expert_name in ExpertName:
            builder.add_edge(expert_name.value, "finalize")
        builder.add_edge("finalize", END)
        return builder

    def invoke(
        self,
        task: RestorationTask | dict[str, Any],
        *,
        trajectory_id: str,
        thread_id: str | None = None,
    ) -> WorkflowResult:
        """Validate input, run the graph, and return the existing workflow contract."""

        parsed_task = task if isinstance(task, RestorationTask) else RestorationTask.model_validate(task)
        resolved_thread_id = thread_id or trajectory_id
        config: RunnableConfig = {
            "configurable": {"thread_id": resolved_thread_id},
            "recursion_limit": max(64, self.runtime.settings.max_steps * 12),
        }
        output = self.graph.invoke(
            {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "task": parsed_task.model_dump(mode="json"),
                "trajectory_id": trajectory_id,
            },
            config,
        )
        result = output.get("result")
        if not isinstance(result, dict):
            raise ValueError("compiled restoration graph did not produce a result")
        return WorkflowResult.model_validate(result)

    def get_checkpoint(self, thread_id: str) -> RestorationGraphState:
        """Return the latest JSON-compatible state snapshot for one thread."""

        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return cast(RestorationGraphState, dict(snapshot.values))

    def node_names(self) -> set[str]:
        """Expose the compiled parent graph structure for regression tests."""

        return set(self.graph.get_graph().nodes)
