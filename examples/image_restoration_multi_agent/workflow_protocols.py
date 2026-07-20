"""Dependency protocols shared by controller and LangGraph workflows."""

from __future__ import annotations

from typing import Protocol

from schemas import (
    DiagnosisResult,
    EvaluationResult,
    ExpertDecisionRecord,
    RestorationResult,
    RestorationTrajectoryState,
)


class DiagnosisAgent(Protocol):
    """Protocol required from a degradation diagnosis agent."""

    def diagnose(self, image_path: str) -> DiagnosisResult: ...


class ExpertAgent(Protocol):
    """Protocol required from one restoration expert."""

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord: ...


class RestorationWorker(Protocol):
    """Protocol required from a restoration worker."""

    def restore(self, action: str, input_path: str, output_dir: str, step_index: int) -> RestorationResult: ...


class Evaluator(Protocol):
    """Protocol required from an image quality evaluator."""

    def evaluate(
        self,
        image_path: str,
        *,
        previous_score: float | None,
        original_score: float | None,
        best_score: float | None,
    ) -> EvaluationResult: ...
