"""LangGraph nodes for deterministic image-restoration control flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from exceptions import EvaluationError, InvalidToolCallError, UnknownActionError
from schemas import (
    DiagnosisResult,
    EvaluationResult,
    ExecutionStatus,
    ExpertDecisionRecord,
    ExpertName,
    RestorationResult,
    RestorationStep,
    RestorationTask,
    RestorationTrajectoryState,
    ValidationStatus,
    WorkflowResult,
)
from tool_registry import STOP_ACTION

from .runtime import RestorationGraphRuntime
from .state import GRAPH_SCHEMA_VERSION, RestorationGraphState, require_mapping, require_text


def _trajectory(state: RestorationGraphState) -> RestorationTrajectoryState:
    return RestorationTrajectoryState.model_validate(require_mapping(state, "trajectory"))


def _decision(state: RestorationGraphState) -> ExpertDecisionRecord:
    payload = state.get("pending_decision")
    if payload is None:
        raise ValueError("graph state is missing pending_decision")
    return ExpertDecisionRecord.model_validate(payload)


def _restoration(state: RestorationGraphState) -> RestorationResult:
    payload = state.get("pending_restoration")
    if payload is None:
        raise ValueError("graph state is missing pending_restoration")
    return RestorationResult.model_validate(payload)


def _evaluation(state: RestorationGraphState) -> EvaluationResult:
    payload = state.get("pending_evaluation")
    if payload is None:
        raise ValueError("graph state is missing pending_evaluation")
    return EvaluationResult.model_validate(payload)


def _dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


class RestorationGraphNodes:
    """Node implementations bound to one dependency runtime."""

    def __init__(self, runtime: RestorationGraphRuntime) -> None:
        self.runtime = runtime

    def prepare_input(self, state: RestorationGraphState) -> RestorationGraphState:
        """Validate the task, normalize paths, and create the artifact directory."""

        if state.get("schema_version", GRAPH_SCHEMA_VERSION) != GRAPH_SCHEMA_VERSION:
            raise ValueError(f"unsupported graph schema version: {state.get('schema_version')}")
        task = RestorationTask.model_validate(require_mapping(state, "task"))
        input_path = Path(task.image_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input image does not exist: {input_path}")
        output_dir = Path(task.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        task.image_path = str(input_path)
        task.output_dir = str(output_dir)
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "task": task.model_dump(mode="json"),
            "input_image": str(input_path),
            "output_dir": str(output_dir),
        }

    def diagnose(self, state: RestorationGraphState) -> RestorationGraphState:
        """Run the single deterministic diagnosis call."""

        diagnosis = self.runtime.diagnosis_agent.diagnose(require_text(state, "input_image"))
        if diagnosis.route_to not in self.runtime.experts:
            raise ValueError(f"no expert registered for {diagnosis.route_to.value}")
        return {"diagnosis": _dump(diagnosis)}

    def score_original(self, state: RestorationGraphState) -> RestorationGraphState:
        """Create the original-image IQA baseline."""

        evaluation = self.runtime.evaluator.evaluate(
            require_text(state, "input_image"),
            previous_score=None,
            original_score=None,
            best_score=None,
        )
        if evaluation.status != ExecutionStatus.SUCCESS:
            raise EvaluationError(evaluation.error or "original image evaluation failed")
        return {"original_evaluation": _dump(evaluation)}

    def initialize_trajectory(self, state: RestorationGraphState) -> RestorationGraphState:
        """Create the controller-compatible trajectory model."""

        diagnosis = DiagnosisResult.model_validate(require_mapping(state, "diagnosis"))
        evaluation = EvaluationResult.model_validate(require_mapping(state, "original_evaluation"))
        input_image = require_text(state, "input_image")
        trajectory = RestorationTrajectoryState(
            trajectory_id=require_text(state, "trajectory_id"),
            original_image=input_image,
            current_image=input_image,
            best_image=input_image,
            diagnosis=diagnosis,
            expert_name=diagnosis.route_to,
            original_evaluation=evaluation,
            current_evaluation=evaluation,
            best_evaluation=evaluation,
        )
        return {"trajectory": _dump(trajectory)}

    def decide_action(self, expert_name: ExpertName) -> Callable[[RestorationGraphState], RestorationGraphState]:
        """Bind one stable expert identity to a decision node."""

        def node(state: RestorationGraphState) -> RestorationGraphState:
            trajectory = _trajectory(state)
            if trajectory.expert_name != expert_name:
                raise ValueError(
                    f"expert subgraph mismatch: state={trajectory.expert_name.value}, node={expert_name.value}"
                )
            decision = self.runtime.experts[expert_name].decide(trajectory)
            return {
                "pending_decision": _dump(decision),
                "pending_action": None,
                "pending_error": None,
            }

        return node

    def validate_action(self, state: RestorationGraphState) -> RestorationGraphState:
        """Validate the pending decision and terminate before side effects on failure."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        try:
            action = self.runtime.logic.validate_decision(decision, trajectory)
        except UnknownActionError as error:
            decision.validation_status = ValidationStatus.UNKNOWN_ACTION
            trajectory.invalid_action_count += 1
            self.runtime.logic.append_decision_failure(trajectory, decision, str(error))
            self.runtime.logic.terminate(trajectory, "invalid_action")
            return {
                "trajectory": _dump(trajectory),
                "pending_decision": _dump(decision),
                "pending_action": None,
                "pending_error": str(error),
            }
        except InvalidToolCallError as error:
            decision.validation_status = ValidationStatus.INVALID_TOOL_CALL
            trajectory.invalid_action_count += 1
            self.runtime.logic.append_decision_failure(trajectory, decision, str(error))
            self.runtime.logic.terminate(trajectory, "invalid_tool_call")
            return {
                "trajectory": _dump(trajectory),
                "pending_decision": _dump(decision),
                "pending_action": None,
                "pending_error": str(error),
            }
        return {
            "trajectory": _dump(trajectory),
            "pending_decision": _dump(decision),
            "pending_action": action,
            "pending_error": None,
        }

    def record_stop(self, state: RestorationGraphState) -> RestorationGraphState:
        """Record a valid stop decision and terminate the expert subgraph."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        if state.get("pending_action") != STOP_ACTION:
            raise ValueError("record_stop requires pending action stop")
        stop_reward, reward_components = self.runtime.logic.calculate_stop_reward(trajectory)
        trajectory.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=trajectory.expert_name,
                expert_decision=decision,
                tool_name=None,
                input_image=trajectory.current_image,
                output_image=None,
                step_reward=stop_reward,
                reward_components=reward_components,
                success=True,
                latency_seconds=0.0,
            )
        )
        self.runtime.logic.terminate(trajectory, "expert_stop")
        return {
            "trajectory": _dump(trajectory),
            "pending_action": None,
            "pending_restoration": None,
            "pending_evaluation": None,
        }

    def execute_restoration(self, state: RestorationGraphState) -> RestorationGraphState:
        """Execute one validated non-stop restoration action."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        action = state.get("pending_action")
        if action is None or action == STOP_ACTION:
            raise ValueError("execute_restoration requires one non-stop action")
        trajectory.tool_call_count += 1
        result = self.runtime.worker.restore(
            action,
            trajectory.current_image,
            require_text(state, "output_dir"),
            decision.step_index,
        )
        return {
            "trajectory": _dump(trajectory),
            "pending_restoration": _dump(result),
            "pending_evaluation": None,
        }

    def record_worker_failure(self, state: RestorationGraphState) -> RestorationGraphState:
        """Record a failed restoration without replacing the current image."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        restoration = _restoration(state)
        action = state.get("pending_action")
        if action is None:
            raise ValueError("worker failure requires pending_action")
        trajectory.consecutive_failures += 1
        step_reward, reward_components = self.runtime.logic.calculate_failure_reward(trajectory, action)
        trajectory.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=trajectory.expert_name,
                expert_decision=decision,
                tool_name=action,
                input_image=trajectory.current_image,
                output_image=None,
                restoration=restoration,
                step_reward=step_reward,
                reward_components=reward_components,
                success=False,
                latency_seconds=restoration.latency_seconds,
                error=restoration.error,
            )
        )
        if trajectory.consecutive_failures >= self.runtime.settings.max_consecutive_failures:
            self.runtime.logic.terminate(trajectory, "consecutive_worker_failures")
        return {
            "trajectory": _dump(trajectory),
            "pending_action": None,
            "pending_restoration": None,
            "pending_evaluation": None,
        }

    def score_candidate(self, state: RestorationGraphState) -> RestorationGraphState:
        """Evaluate one successful restoration candidate."""

        trajectory = _trajectory(state)
        restoration = _restoration(state)
        if restoration.output_path is None:
            raise ValueError("candidate evaluation requires restoration output_path")
        evaluation = self.runtime.evaluator.evaluate(
            restoration.output_path,
            previous_score=trajectory.current_evaluation.aggregate_score,
            original_score=trajectory.original_evaluation.aggregate_score,
            best_score=trajectory.best_evaluation.aggregate_score,
        )
        return {"pending_evaluation": _dump(evaluation)}

    def record_evaluation_failure(self, state: RestorationGraphState) -> RestorationGraphState:
        """Record a failed IQA result without accepting the candidate image."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        restoration = _restoration(state)
        evaluation = _evaluation(state)
        action = state.get("pending_action")
        if action is None:
            raise ValueError("evaluation failure requires pending_action")
        trajectory.consecutive_failures += 1
        step_reward, reward_components = self.runtime.logic.calculate_failure_reward(trajectory, action)
        trajectory.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=trajectory.expert_name,
                expert_decision=decision,
                tool_name=action,
                input_image=trajectory.current_image,
                output_image=restoration.output_path,
                restoration=restoration,
                evaluation=evaluation,
                step_reward=step_reward,
                reward_components=reward_components,
                success=False,
                latency_seconds=restoration.latency_seconds,
                error=evaluation.error,
            )
        )
        if trajectory.consecutive_failures >= self.runtime.settings.max_consecutive_failures:
            self.runtime.logic.terminate(trajectory, "consecutive_evaluation_failures")
        return {
            "trajectory": _dump(trajectory),
            "pending_action": None,
            "pending_restoration": None,
            "pending_evaluation": None,
        }

    def commit_successful_step(self, state: RestorationGraphState) -> RestorationGraphState:
        """Accept one evaluated candidate and update current and historical-best state."""

        trajectory = _trajectory(state)
        decision = _decision(state)
        restoration = _restoration(state)
        evaluation = _evaluation(state)
        action = state.get("pending_action")
        if action is None or restoration.output_path is None:
            raise ValueError("successful commit requires action and restoration output")

        previous_image = trajectory.current_image
        trajectory.current_image = restoration.output_path
        trajectory.current_evaluation = evaluation
        trajectory.consecutive_failures = 0
        if evaluation.is_new_best:
            trajectory.best_image = restoration.output_path
            trajectory.best_evaluation = evaluation
            trajectory.consecutive_no_improvement = 0
        else:
            trajectory.consecutive_no_improvement += 1

        step_reward, reward_components = self.runtime.logic.calculate_success_reward(
            trajectory,
            action,
            evaluation,
        )
        trajectory.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=trajectory.expert_name,
                expert_decision=decision,
                tool_name=action,
                input_image=previous_image,
                output_image=restoration.output_path,
                restoration=restoration,
                evaluation=evaluation,
                step_reward=step_reward,
                reward_components=reward_components,
                success=True,
                latency_seconds=restoration.latency_seconds,
            )
        )
        if (
            self.runtime.settings.no_improvement_limit is not None
            and trajectory.consecutive_no_improvement >= self.runtime.settings.no_improvement_limit
        ):
            self.runtime.logic.terminate(trajectory, "no_improvement_limit")
        return {
            "trajectory": _dump(trajectory),
            "pending_action": None,
            "pending_restoration": None,
            "pending_evaluation": None,
        }

    def record_max_steps(self, state: RestorationGraphState) -> RestorationGraphState:
        """Terminate when the explicit business step limit is exhausted."""

        trajectory = _trajectory(state)
        self.runtime.logic.terminate(trajectory, "max_steps")
        return {"trajectory": _dump(trajectory)}

    def finalize(self, state: RestorationGraphState) -> RestorationGraphState:
        """Calculate final reward and export the controller-compatible result."""

        trajectory = _trajectory(state)
        if not trajectory.terminated:
            raise ValueError("finalize requires a terminated trajectory")
        trajectory.final_reward = self.runtime.logic.calculate_final_reward(trajectory)
        trajectory_path = Path(require_text(state, "output_dir")) / "trajectory.json"
        trajectory_path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
        summary = {
            "trajectory_id": trajectory.trajectory_id,
            "expert_name": trajectory.expert_name.value,
            "best_image": trajectory.best_image,
            "termination_reason": trajectory.termination_reason,
            "tool_call_count": trajectory.tool_call_count,
            "final_reward": trajectory.final_reward,
        }
        result = WorkflowResult(
            state=trajectory,
            trajectory_path=str(trajectory_path),
            summary=summary,
        )
        return {"trajectory": _dump(trajectory), "result": _dump(result)}
