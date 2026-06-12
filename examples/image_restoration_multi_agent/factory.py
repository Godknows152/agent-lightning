"""Build isolated deterministic workflow components for each rollout."""

from __future__ import annotations

from agents import ScriptedDiagnosisAgent, ScriptedExpertAgent
from config import ExampleConfig
from controller import ImageRestorationController
from evaluators import ScriptedEvaluator
from schemas import DEGRADATION_TO_EXPERT, ExpertName, RestorationTask
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker


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
