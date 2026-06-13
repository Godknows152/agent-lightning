"""Build isolated deterministic workflow components for each rollout."""

from __future__ import annotations

from typing import Mapping

from agents import ScriptedDiagnosisAgent, ScriptedExpertAgent
from config import ExampleConfig, RealExampleConfig
from controller import ExpertAgent, ImageRestorationController
from evaluators import PyiqaSubprocessEvaluator, ScriptedEvaluator
from schemas import DEGRADATION_TO_EXPERT, ExpertName, RealRestorationTask, RestorationTask
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker, SubprocessRestorationWorker


class DeterministicControllerFactory:
    """Create per-task controllers so scripted cursors never leak across rollouts."""

    def __init__(self, config: ExampleConfig, tool_registry: ToolRegistry) -> None:
        self.config = config
        self.tool_registry = tool_registry

    def build(self, task: RestorationTask) -> ImageRestorationController:
        """Construct one controller and all deterministic collaborators."""

        routed_expert = DEGRADATION_TO_EXPERT[task.degradation_type]
        experts = {
            expert_name: ScriptedExpertAgent(
                expert_name,
                task.scripted_actions if expert_name == routed_expert else ["stop"],
            )
            for expert_name in ExpertName
        }
        return ImageRestorationController(
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


class RealControllerFactory:
    """Create stage D controllers with isolated real restoration and IQA models."""

    def __init__(self, config: RealExampleConfig, tool_registry: ToolRegistry) -> None:
        self.config = config
        self.tool_registry = tool_registry

    def build(
        self,
        task: RealRestorationTask,
        *,
        routed_expert_override: ExpertName | None = None,
        experts_override: Mapping[ExpertName, ExpertAgent] | None = None,
    ) -> ImageRestorationController:
        """Construct one real-model controller with scripted or injected experts."""

        routed_expert = routed_expert_override or DEGRADATION_TO_EXPERT[task.degradation_type]
        experts: Mapping[ExpertName, ExpertAgent] = experts_override or {
            expert_name: ScriptedExpertAgent(
                expert_name,
                task.scripted_actions if expert_name == routed_expert else ["stop"],
            )
            for expert_name in ExpertName
        }
        return ImageRestorationController(
            settings=self.config.workflow,
            tool_registry=self.tool_registry,
            diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type, task.visual_evidence),
            experts=experts,
            worker=SubprocessRestorationWorker(self.config.runtime.restoration, self.tool_registry),
            evaluator=PyiqaSubprocessEvaluator(
                self.config.runtime.evaluator,
                improvement_epsilon=self.config.workflow.improvement_epsilon,
            ),
        )
