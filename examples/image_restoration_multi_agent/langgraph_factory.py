"""Build no-model LangGraph workflows with isolated deterministic dependencies."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from agents import ReplayExpertAgent, ScriptedDiagnosisAgent
from config import ExampleConfig
from evaluators import ScriptedEvaluator
from graph import LangGraphImageRestorationWorkflow, RestorationGraphRuntime
from schemas import DEGRADATION_TO_EXPERT, ExpertName, RestorationTask
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker


class DeterministicLangGraphFactory:
    """Create one checkpointed no-model graph per deterministic task."""

    def __init__(self, config: ExampleConfig, tool_registry: ToolRegistry) -> None:
        self.config = config
        self.tool_registry = tool_registry

    def build(
        self,
        task: RestorationTask,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> LangGraphImageRestorationWorkflow:
        """Construct fresh scripted diagnosis, replay experts, worker, and evaluator."""

        routed_expert = DEGRADATION_TO_EXPERT[task.degradation_type]
        experts = {
            expert_name: ReplayExpertAgent.from_actions(
                expert_name,
                task.scripted_actions if expert_name == routed_expert else ["stop"],
                self.tool_registry,
            )
            for expert_name in ExpertName
        }
        runtime = RestorationGraphRuntime(
            settings=self.config.workflow,
            tool_registry=self.tool_registry,
            diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type, task.visual_evidence),
            experts=experts,
            worker=CopyRestorationWorker(set(task.fail_actions)),
            evaluator=ScriptedEvaluator(
                task.score_sequence,
                improvement_epsilon=self.config.workflow.improvement_epsilon,
                fail_indices=set(task.fail_evaluation_indices),
            ),
        )
        return LangGraphImageRestorationWorkflow(runtime, checkpointer=checkpointer)
