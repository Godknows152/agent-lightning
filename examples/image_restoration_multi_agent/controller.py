"""Controller for the deterministic hierarchical restoration workflow."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from config import WorkflowSettings
from exceptions import EvaluationError, InvalidToolCallError, UnknownActionError
from schemas import (
    DiagnosisResult,
    EvaluationResult,
    ExecutionStatus,
    ExpertDecisionRecord,
    ExpertName,
    RestorationResult,
    RestorationStep,
    RestorationTaskBase,
    RestorationTrajectoryState,
    ValidationStatus,
    WorkflowResult,
)
from tool_registry import STOP_ACTION, ToolRegistry

import agentlightning as agl

logger = logging.getLogger(__name__)


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


@contextmanager
def _optional_operation(
    enabled: bool,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Create an Agent Lightning operation only for traced rollouts."""

    if enabled:
        with agl.operation(name=name, **(attributes or {})) as operation:
            yield operation
    else:
        yield None


class ImageRestorationController:
    """Own the single-diagnosis, fixed-expert restoration state machine."""

    def __init__(
        self,
        *,
        settings: WorkflowSettings,
        tool_registry: ToolRegistry,
        diagnosis_agent: DiagnosisAgent,
        experts: Mapping[ExpertName, ExpertAgent],
        worker: RestorationWorker,
        evaluator: Evaluator,
    ) -> None:
        self.settings = settings
        self.tool_registry = tool_registry
        self.diagnosis_agent = diagnosis_agent
        self.experts = experts
        self.worker = worker
        self.evaluator = evaluator

    def run(
        self,
        task: RestorationTaskBase,
        *,
        trajectory_id: str,
        trace: bool = False,
        diagnosis_override: DiagnosisResult | None = None,
    ) -> WorkflowResult:
        """Execute one restoration trajectory and export its state.

        Args:
            task: Input image and scripted expert actions.
            trajectory_id: Stable rollout identifier.
            trace: Whether to emit Agent Lightning operations.
            diagnosis_override: Precomputed diagnosis, used when stage E has
                already made and traced the single allowed VLM request.
        """

        input_path = Path(task.image_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input image does not exist: {input_path}")
        output_dir = Path(task.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        diagnosis = diagnosis_override or self._diagnose(str(input_path), trace=trace)
        expert_name = diagnosis.route_to
        expert = self.experts.get(expert_name)
        if expert is None:
            raise ValueError(f"no expert registered for {expert_name.value}")

        original_evaluation = self._evaluate(
            str(input_path),
            previous_score=None,
            original_score=None,
            best_score=None,
            trace=trace,
            operation_name="iqa_evaluator.original",
        )
        if original_evaluation.status != ExecutionStatus.SUCCESS:
            raise EvaluationError(original_evaluation.error or "original image evaluation failed")

        state = RestorationTrajectoryState(
            trajectory_id=trajectory_id,
            original_image=str(input_path),
            current_image=str(input_path),
            best_image=str(input_path),
            diagnosis=diagnosis,
            expert_name=expert_name,
            original_evaluation=original_evaluation,
            current_evaluation=original_evaluation,
            best_evaluation=original_evaluation,
        )

        for _ in range(self.settings.max_steps):
            decision = self._decide(expert, state, trace=trace)
            try:
                action = self._validate_decision(decision, state)
            except UnknownActionError as error:
                decision.validation_status = ValidationStatus.UNKNOWN_ACTION
                state.invalid_action_count += 1
                self._append_decision_failure(state, decision, str(error))
                self._terminate(state, "invalid_action")
                break
            except InvalidToolCallError as error:
                decision.validation_status = ValidationStatus.INVALID_TOOL_CALL
                state.invalid_action_count += 1
                self._append_decision_failure(state, decision, str(error))
                self._terminate(state, "invalid_tool_call")
                break

            if action == STOP_ACTION:
                state.steps.append(
                    RestorationStep(
                        step_index=decision.step_index,
                        expert_name=state.expert_name,
                        expert_decision=decision,
                        tool_name=None,
                        input_image=state.current_image,
                        output_image=None,
                        step_reward=0.0,
                        success=True,
                        latency_seconds=0.0,
                    )
                )
                self._terminate(state, "expert_stop")
                break

            state.tool_call_count += 1
            restoration = self._restore(action, decision, state, str(output_dir), trace=trace)
            if restoration.status != ExecutionStatus.SUCCESS or restoration.output_path is None:
                state.consecutive_failures += 1
                state.steps.append(
                    RestorationStep(
                        step_index=decision.step_index,
                        expert_name=state.expert_name,
                        expert_decision=decision,
                        tool_name=action,
                        input_image=state.current_image,
                        output_image=None,
                        restoration=restoration,
                        step_reward=-self.settings.failure_penalty,
                        success=False,
                        latency_seconds=restoration.latency_seconds,
                        error=restoration.error,
                    )
                )
                if state.consecutive_failures >= self.settings.max_consecutive_failures:
                    self._terminate(state, "consecutive_worker_failures")
                    break
                continue

            evaluation = self._evaluate(
                restoration.output_path,
                previous_score=state.current_evaluation.aggregate_score,
                original_score=state.original_evaluation.aggregate_score,
                best_score=state.best_evaluation.aggregate_score,
                trace=trace,
                operation_name="iqa_evaluator.score",
            )
            if evaluation.status != ExecutionStatus.SUCCESS:
                state.consecutive_failures += 1
                state.steps.append(
                    RestorationStep(
                        step_index=decision.step_index,
                        expert_name=state.expert_name,
                        expert_decision=decision,
                        tool_name=action,
                        input_image=state.current_image,
                        output_image=restoration.output_path,
                        restoration=restoration,
                        evaluation=evaluation,
                        step_reward=-self.settings.failure_penalty,
                        success=False,
                        latency_seconds=restoration.latency_seconds,
                        error=evaluation.error,
                    )
                )
                if state.consecutive_failures >= self.settings.max_consecutive_failures:
                    self._terminate(state, "consecutive_evaluation_failures")
                    break
                continue

            previous_image = state.current_image
            state.current_image = restoration.output_path
            state.current_evaluation = evaluation
            state.consecutive_failures = 0
            if evaluation.is_new_best:
                state.best_image = restoration.output_path
                state.best_evaluation = evaluation
                state.consecutive_no_improvement = 0
            else:
                state.consecutive_no_improvement += 1

            state.steps.append(
                RestorationStep(
                    step_index=decision.step_index,
                    expert_name=state.expert_name,
                    expert_decision=decision,
                    tool_name=action,
                    input_image=previous_image,
                    output_image=restoration.output_path,
                    restoration=restoration,
                    evaluation=evaluation,
                    step_reward=evaluation.delta_from_previous,
                    success=True,
                    latency_seconds=restoration.latency_seconds,
                )
            )
            if state.consecutive_no_improvement >= self.settings.no_improvement_limit:
                self._terminate(state, "no_improvement_limit")
                break

        if not state.terminated:
            self._terminate(state, "max_steps")

        state.final_reward = self._calculate_final_reward(state)
        trajectory_path = output_dir / "trajectory.json"
        trajectory_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        summary = {
            "trajectory_id": state.trajectory_id,
            "expert_name": state.expert_name.value,
            "best_image": state.best_image,
            "termination_reason": state.termination_reason,
            "tool_call_count": state.tool_call_count,
            "final_reward": state.final_reward,
        }
        if trace:
            agl.emit_object(summary, attributes={"restoration.record_type": "trajectory_summary"})
        return WorkflowResult(state=state, trajectory_path=str(trajectory_path), summary=summary)

    def _diagnose(self, image_path: str, *, trace: bool) -> DiagnosisResult:
        with _optional_operation(
            trace,
            "diagnosis_agent.decision",
            {"agent.name": "diagnosis_agent"},
        ) as operation:
            diagnosis = self.diagnosis_agent.diagnose(image_path)
            if operation is not None:
                payload = diagnosis.model_dump(mode="json")
                operation.set_output(payload)
                agl.emit_object(payload, attributes={"restoration.record_type": "diagnosis"})
            return diagnosis

    def _decide(
        self,
        expert: ExpertAgent,
        state: RestorationTrajectoryState,
        *,
        trace: bool,
    ) -> ExpertDecisionRecord:
        with _optional_operation(
            trace,
            f"{state.expert_name.value}.decision",
            {"agent.name": state.expert_name.value},
        ) as operation:
            decision = expert.decide(state)
            if operation is not None:
                operation.set_input(resource_name=decision.resource_name)
                payload = decision.model_dump(mode="json")
                operation.set_output(payload)
                agl.emit_object(payload, attributes={"restoration.record_type": "expert_decision"})
            return decision

    def _restore(
        self,
        action: str,
        decision: ExpertDecisionRecord,
        state: RestorationTrajectoryState,
        output_dir: str,
        *,
        trace: bool,
    ) -> RestorationResult:
        with _optional_operation(trace, f"restoration_worker.{action}") as operation:
            result = self.worker.restore(
                action,
                state.current_image,
                output_dir,
                decision.step_index,
            )
            if operation is not None:
                payload = result.model_dump(mode="json")
                operation.set_output(payload)
                agl.emit_object(payload, attributes={"restoration.record_type": "worker_result"})
            return result

    def _evaluate(
        self,
        image_path: str,
        *,
        previous_score: float | None,
        original_score: float | None,
        best_score: float | None,
        trace: bool,
        operation_name: str,
    ) -> EvaluationResult:
        with _optional_operation(trace, operation_name) as operation:
            result = self.evaluator.evaluate(
                image_path,
                previous_score=previous_score,
                original_score=original_score,
                best_score=best_score,
            )
            if operation is not None:
                payload = result.model_dump(mode="json")
                operation.set_output(payload)
                agl.emit_object(payload, attributes={"restoration.record_type": "evaluation"})
            return result

    def _validate_decision(self, decision: ExpertDecisionRecord, state: RestorationTrajectoryState) -> str:
        if decision.validation_status == ValidationStatus.UNKNOWN_ACTION:
            raise UnknownActionError(decision.error or "expert selected an unknown action")
        if decision.validation_status != ValidationStatus.VALID or decision.action is None:
            raise InvalidToolCallError(decision.error or f"expert response is {decision.parse_status.value}")
        if decision.expert_name != state.expert_name:
            raise InvalidToolCallError(
                f"expert switched from {state.expert_name.value} to {decision.expert_name.value}"
            )
        if decision.step_index != len(state.steps):
            raise InvalidToolCallError(
                f"decision step_index={decision.step_index} does not match expected {len(state.steps)}"
            )
        self.tool_registry.validate_action(decision.action)
        return decision.action

    def _append_decision_failure(
        self,
        state: RestorationTrajectoryState,
        decision: ExpertDecisionRecord,
        error: str,
    ) -> None:
        state.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=state.expert_name,
                expert_decision=decision,
                tool_name=None,
                input_image=state.current_image,
                output_image=None,
                step_reward=-self.settings.invalid_action_penalty,
                success=False,
                latency_seconds=0.0,
                error=error,
            )
        )

    @staticmethod
    def _terminate(state: RestorationTrajectoryState, reason: str) -> None:
        state.terminated = True
        state.termination_reason = reason

    def _calculate_final_reward(self, state: RestorationTrajectoryState) -> float:
        quality_gain = state.best_evaluation.aggregate_score - state.original_evaluation.aggregate_score
        failures = sum(not step.success for step in state.steps)
        return (
            quality_gain
            - self.settings.tool_call_cost * state.tool_call_count
            - self.settings.invalid_action_penalty * state.invalid_action_count
            - self.settings.failure_penalty * failures
        )
