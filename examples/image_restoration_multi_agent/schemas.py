"""Pydantic data contracts for the multi-agent restoration workflow."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared protocol fields."""

    model_config = ConfigDict(extra="forbid")


class DegradationType(str, Enum):
    """Supported primary degradation categories."""

    FOG = "fog"
    SNOW = "snow"
    RAIN = "rain"
    LOW_LIGHT = "low_light"


class ExpertName(str, Enum):
    """Stable expert identities used by Agent Lightning traces."""

    FOG = "fog_expert"
    SNOW = "snow_expert"
    RAIN = "rain_expert"
    LOW_LIGHT = "low_light_expert"


DEGRADATION_TO_EXPERT: dict[DegradationType, ExpertName] = {
    DegradationType.FOG: ExpertName.FOG,
    DegradationType.SNOW: ExpertName.SNOW,
    DegradationType.RAIN: ExpertName.RAIN,
    DegradationType.LOW_LIGHT: ExpertName.LOW_LIGHT,
}


class ValidationStatus(str, Enum):
    """Controller validation result for an expert action."""

    VALID = "valid"
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNKNOWN_ACTION = "unknown_action"


class ExecutionStatus(str, Enum):
    """Worker or evaluator execution status."""

    SUCCESS = "success"
    FAILED = "failed"


class DiagnosisResult(StrictModel):
    """Single deterministic degradation diagnosis and route."""

    primary_type: DegradationType
    visual_evidence: list[str] = Field(default_factory=list)
    route_to: ExpertName

    @model_validator(mode="after")
    def validate_route(self) -> DiagnosisResult:
        expected = DEGRADATION_TO_EXPERT[self.primary_type]
        if self.route_to != expected:
            raise ValueError(f"{self.primary_type.value} must route to {expected.value}")
        return self


class ExpertDecisionRecord(StrictModel):
    """Validated record of one expert tool-call decision."""

    expert_name: ExpertName
    step_index: int = Field(ge=0)
    action: str = Field(min_length=1)
    tool_call_id: str | None = None
    llm_response_id: str | None = None
    validation_status: ValidationStatus = ValidationStatus.VALID
    raw_assistant_output: str | None = None
    generated_token_ids: list[int] | None = None


class RestorationResult(StrictModel):
    """Normalized result returned by a restoration worker."""

    status: ExecutionStatus
    worker: str
    input_path: str
    output_path: str | None = None
    latency_seconds: float = Field(ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> RestorationResult:
        if self.status == ExecutionStatus.SUCCESS and self.output_path is None:
            raise ValueError("successful restoration requires output_path")
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("failed restoration requires error")
        return self


class EvaluationResult(StrictModel):
    """Structured image quality evaluation and feedback."""

    status: ExecutionStatus = ExecutionStatus.SUCCESS
    raw_scores: dict[str, float]
    normalized_scores: dict[str, float]
    aggregate_score: float
    delta_from_previous: float
    delta_from_original: float
    delta_from_best: float
    is_new_best: bool
    feedback: str
    error: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> EvaluationResult:
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("failed evaluation requires error")
        return self


class RestorationStep(StrictModel):
    """One decision and its optional worker/evaluator result."""

    step_index: int = Field(ge=0)
    expert_name: ExpertName
    expert_decision: ExpertDecisionRecord
    tool_name: str | None
    input_image: str
    output_image: str | None
    restoration: RestorationResult | None = None
    evaluation: EvaluationResult | None = None
    step_reward: float = 0.0
    success: bool
    latency_seconds: float = Field(ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_expert_identity(self) -> RestorationStep:
        if self.expert_name != self.expert_decision.expert_name:
            raise ValueError("step expert_name must match expert_decision.expert_name")
        return self


class RestorationTrajectoryState(StrictModel):
    """Serializable state owned by the workflow controller."""

    trajectory_id: str
    original_image: str
    current_image: str
    best_image: str
    diagnosis: DiagnosisResult
    expert_name: ExpertName
    original_evaluation: EvaluationResult
    current_evaluation: EvaluationResult
    best_evaluation: EvaluationResult
    steps: list[RestorationStep] = Field(default_factory=cast(Callable[[], list[RestorationStep]], list))
    tool_call_count: int = Field(default=0, ge=0)
    consecutive_no_improvement: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    invalid_action_count: int = Field(default=0, ge=0)
    terminated: bool = False
    termination_reason: str | None = None
    final_reward: float | None = None

    @model_validator(mode="after")
    def validate_route_and_termination(self) -> RestorationTrajectoryState:
        if self.expert_name != self.diagnosis.route_to:
            raise ValueError("trajectory expert_name must match the diagnosis route")
        if self.terminated and not self.termination_reason:
            raise ValueError("terminated trajectory requires termination_reason")
        return self


class RestorationTask(StrictModel):
    """Input accepted by the deterministic stage A-C rollout."""

    image_path: str
    degradation_type: DegradationType
    scripted_actions: list[str]
    score_sequence: list[float]
    output_dir: str
    fail_actions: list[str] = Field(default_factory=list)
    fail_evaluation_indices: list[int] = Field(default_factory=cast(Callable[[], list[int]], list))
    visual_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sequences(self) -> RestorationTask:
        if not self.scripted_actions:
            raise ValueError("scripted_actions must contain at least one action")
        if not self.score_sequence:
            raise ValueError("score_sequence must contain the original image score")
        return self


class WorkflowResult(StrictModel):
    """Final controller output returned to the smoke-test caller."""

    state: RestorationTrajectoryState
    trajectory_path: str
    summary: dict[str, Any]
