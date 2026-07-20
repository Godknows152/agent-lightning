"""Injected runtime dependencies used by restoration graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from config import WorkflowSettings
from schemas import ExpertName
from tool_registry import ToolRegistry
from workflow_logic import RestorationWorkflowLogic
from workflow_protocols import DiagnosisAgent, Evaluator, ExpertAgent, RestorationWorker


@dataclass(frozen=True)
class RestorationGraphRuntime:
    """Non-serializable collaborators kept outside graph checkpoint state."""

    settings: WorkflowSettings
    tool_registry: ToolRegistry
    diagnosis_agent: DiagnosisAgent
    experts: Mapping[ExpertName, ExpertAgent]
    worker: RestorationWorker
    evaluator: Evaluator
    logic: RestorationWorkflowLogic = field(init=False)

    def __post_init__(self) -> None:
        expected = set(ExpertName)
        if set(self.experts) != expected:
            missing = sorted(item.value for item in expected - set(self.experts))
            extra = sorted(item.value for item in set(self.experts) - expected)
            raise ValueError(f"graph runtime requires exactly four experts; missing={missing}, extra={extra}")
        object.__setattr__(self, "logic", RestorationWorkflowLogic(self.settings, self.tool_registry))
