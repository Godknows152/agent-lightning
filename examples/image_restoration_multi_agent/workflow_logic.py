"""Shared validation, termination, and reward semantics for restoration workflows."""

from __future__ import annotations

from dataclasses import dataclass

from config import WorkflowSettings
from exceptions import InvalidToolCallError, UnknownActionError
from schemas import (
    EvaluationResult,
    ExpertDecisionRecord,
    RestorationStep,
    RestorationTrajectoryState,
    ValidationStatus,
)
from tool_registry import ToolRegistry


@dataclass(frozen=True)
class RestorationWorkflowLogic:
    """Apply controller-equivalent business rules without owning control flow."""

    settings: WorkflowSettings
    tool_registry: ToolRegistry

    def validate_decision(self, decision: ExpertDecisionRecord, state: RestorationTrajectoryState) -> str:
        """Validate one expert decision before any worker side effect."""

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

    def append_decision_failure(
        self,
        state: RestorationTrajectoryState,
        decision: ExpertDecisionRecord,
        error: str,
    ) -> None:
        """Record an invalid decision as one completed failed step."""

        state.steps.append(
            RestorationStep(
                step_index=decision.step_index,
                expert_name=state.expert_name,
                expert_decision=decision,
                tool_name=None,
                input_image=state.current_image,
                output_image=None,
                step_reward=-self.settings.invalid_action_penalty,
                reward_components={"invalid_action_penalty": -self.settings.invalid_action_penalty},
                success=False,
                latency_seconds=0.0,
                error=error,
            )
        )

    @staticmethod
    def terminate(state: RestorationTrajectoryState, reason: str) -> None:
        """Mark one trajectory terminal with an explicit business reason."""

        state.terminated = True
        state.termination_reason = reason

    def calculate_final_reward(self, state: RestorationTrajectoryState) -> float:
        """Calculate the single trajectory-level reward."""

        if state.termination_reason in {"invalid_tool_call", "invalid_action"}:
            return -self.settings.invalid_action_penalty

        if self.settings.reward_mode == "step_iqa_sum_v1":
            trajectory_reward = sum(step.step_reward for step in state.steps)
            if state.termination_reason in {"max_steps", "no_improvement_limit"}:
                trajectory_reward -= self.settings.forced_termination_penalty
            return trajectory_reward

        quality_gain = state.best_evaluation.aggregate_score - state.original_evaluation.aggregate_score
        failures = sum(not step.success for step in state.steps)
        return (
            quality_gain
            + self.settings.tool_call_reward * state.tool_call_count
            - self.settings.tool_call_cost * state.tool_call_count
            - self.settings.invalid_action_penalty * state.invalid_action_count
            - self.settings.failure_penalty * failures
        )

    def calculate_success_reward(
        self,
        state: RestorationTrajectoryState,
        action: str,
        evaluation: EvaluationResult,
    ) -> tuple[float, dict[str, float]]:
        """Calculate the reward for one successful restoration and evaluation."""

        if self.settings.reward_mode != "step_iqa_sum_v1":
            return evaluation.delta_from_previous, {"delta_from_previous": evaluation.delta_from_previous}

        quality_delta = (
            self.settings.reward_alpha * evaluation.delta_from_previous
            + (1.0 - self.settings.reward_alpha) * evaluation.delta_from_original
        )
        scaled_quality = self.settings.reward_scale * quality_delta
        clipped_quality = min(
            self.settings.step_reward_clip,
            max(-self.settings.step_reward_clip, scaled_quality),
        )
        repeated_penalty = (
            self.settings.repeated_action_penalty if any(step.tool_name == action for step in state.steps) else 0.0
        )
        step_reward = clipped_quality + self.settings.tool_call_reward - self.settings.tool_call_cost - repeated_penalty
        return step_reward, {
            "delta_from_previous": evaluation.delta_from_previous,
            "delta_from_original": evaluation.delta_from_original,
            "quality_delta": quality_delta,
            "scaled_clipped_quality": clipped_quality,
            "tool_call_reward": self.settings.tool_call_reward,
            "tool_call_cost": -self.settings.tool_call_cost,
            "repeated_action_penalty": -repeated_penalty,
        }

    def calculate_failure_reward(
        self,
        state: RestorationTrajectoryState,
        action: str,
    ) -> tuple[float, dict[str, float]]:
        """Calculate the reward for a worker or evaluator failure."""

        if self.settings.reward_mode != "step_iqa_sum_v1":
            return -self.settings.failure_penalty, {"failure_penalty": -self.settings.failure_penalty}
        repeated_penalty = (
            self.settings.repeated_action_penalty if any(step.tool_name == action for step in state.steps) else 0.0
        )
        reward = (
            self.settings.tool_call_reward
            - self.settings.failure_penalty
            - self.settings.tool_call_cost
            - repeated_penalty
        )
        return reward, {
            "failure_penalty": -self.settings.failure_penalty,
            "tool_call_reward": self.settings.tool_call_reward,
            "tool_call_cost": -self.settings.tool_call_cost,
            "repeated_action_penalty": -repeated_penalty,
        }

    def calculate_stop_reward(
        self,
        state: RestorationTrajectoryState,
    ) -> tuple[float, dict[str, float]]:
        """Calculate the reward for an expert stop decision."""

        if self.settings.reward_mode != "step_iqa_sum_v1":
            return 0.0, {"stop_reward": 0.0}
        best_gain = state.best_evaluation.aggregate_score - state.original_evaluation.aggregate_score
        valid_stop = (
            state.tool_call_count >= self.settings.stop_min_tool_calls and best_gain >= self.settings.stop_min_best_gain
        )
        stop_reward = self.settings.valid_stop_reward if valid_stop else -self.settings.premature_stop_penalty
        return stop_reward, {
            "best_gain_at_stop": best_gain,
            "valid_stop": 1.0 if valid_stop else 0.0,
            "stop_reward": stop_reward,
        }
