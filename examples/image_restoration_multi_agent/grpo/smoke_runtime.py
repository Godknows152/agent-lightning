"""Lightweight environment used by the real-model GRPO smoke test."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from agents import ScriptedDiagnosisAgent
from controller import ExpertAgent, ImageRestorationController
from factory import RealControllerFactory
from schemas import (
    EvaluationResult,
    ExecutionStatus,
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    RealRestorationTask,
    RestorationTrajectoryState,
    ValidationStatus,
)
from tool_registry import STOP_ACTION
from workers import CopyRestorationWorker


def override_smoke_decision(decision: ExpertDecisionRecord, action: str) -> ExpertDecisionRecord:
    """Force a valid smoke action while retaining the real VLM response trace."""

    response_payload = dict(decision.response_payload or {})
    response_payload["smoke_override"] = {
        "action": action,
        "original_parse_status": decision.parse_status.value,
        "original_validation_status": decision.validation_status.value,
        "original_parsed_payload": decision.parsed_payload,
        "original_error": decision.error,
    }
    return decision.model_copy(
        update={
            "action": action,
            "decision_source": ExpertDecisionSource.SMOKE_OVERRIDE,
            "parse_status": ExpertParseStatus.VALID,
            "tool_call_id": f"smoke-override-{decision.step_index}",
            "validation_status": ValidationStatus.VALID,
            "parsed_payload": {
                "name": "restore_image",
                "arguments": {"action": action},
                "smoke_override": True,
            },
            "response_payload": response_payload,
            "error": (f"smoke override selected {action}; " f"original parse status was {decision.parse_status.value}"),
        }
    )


class SmokeActionOverrideExpert:
    """Call the real policy, then force a deterministic multi-turn smoke path."""

    def __init__(self, delegate: ExpertAgent, actions: list[str]) -> None:
        if not actions:
            raise ValueError("smoke override requires at least one action")
        self.delegate = delegate
        self.actions = list(actions)

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        observed = self.delegate.decide(state)
        step_index = len(state.steps)
        action = self.actions[step_index] if step_index < len(self.actions) else STOP_ACTION
        return override_smoke_decision(observed, action)


class ActionScoreSmokeEvaluator:
    """Assign deterministic action-dependent scores without loading IQA models."""

    def __init__(self, *, improvement_epsilon: float) -> None:
        self.improvement_epsilon = improvement_epsilon

    def evaluate(
        self,
        image_path: str,
        *,
        previous_score: float | None,
        original_score: float | None,
        best_score: float | None,
    ) -> EvaluationResult:
        name = Path(image_path).name
        match = re.match(r"step_\d+_(.+)\.png$", name)
        score = 0.0 if match is None else 0.05 + (sum(match.group(1).encode("utf-8")) % 16) / 100.0
        previous = score if previous_score is None else previous_score
        original = score if original_score is None else original_score
        best = score if best_score is None else best_score
        delta_previous = score - previous
        return EvaluationResult(
            status=ExecutionStatus.SUCCESS,
            raw_scores={"smoke_action_score": score},
            normalized_scores={"smoke_action_score": score},
            aggregate_score=score,
            delta_from_previous=delta_previous,
            delta_from_original=score - original,
            delta_from_best=score - best,
            is_new_best=best_score is None or score > best + self.improvement_epsilon,
            feedback=f"Smoke IQA aggregate_score={score:.4f}; delta={delta_previous:.4f}.",
        )


class SmokeControllerFactory(RealControllerFactory):
    """Use real policy calls with copy restoration and deterministic IQA."""

    def build(
        self,
        task: RealRestorationTask,
        *,
        routed_expert_override: ExpertName | None = None,
        experts_override: Mapping[ExpertName, ExpertAgent] | None = None,
    ) -> ImageRestorationController:
        if experts_override is None:
            raise ValueError("GRPO smoke runtime requires injected experts")
        routed_expert = routed_expert_override
        if routed_expert is None:
            raise ValueError("GRPO smoke runtime requires an oracle route")
        return ImageRestorationController(
            settings=self.config.workflow,
            tool_registry=self.tool_registry,
            diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type),
            experts=experts_override,
            worker=CopyRestorationWorker(),
            evaluator=ActionScoreSmokeEvaluator(
                improvement_epsilon=self.config.workflow.improvement_epsilon,
            ),
        )
