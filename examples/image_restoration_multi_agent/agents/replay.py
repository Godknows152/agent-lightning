"""Replay expert decisions through the same Qwen3 parser used by the VLM."""

from __future__ import annotations

from schemas import (
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    RestorationTrajectoryState,
    ValidationStatus,
)
from tool_registry import RESTORE_FUNCTION_NAME, STOP_ACTION, ToolRegistry

from .vlm_expert import parse_expert_response


class ReplayExpertAgent:
    """Return preconfigured raw Qwen3 tool calls after strict parsing."""

    def __init__(
        self,
        expert_name: ExpertName,
        responses: list[str],
        tool_registry: ToolRegistry,
        *,
        resource_name: str | None = None,
    ) -> None:
        self.expert_name = expert_name
        self.responses = list(responses)
        self.tool_registry = tool_registry
        self.resource_name = resource_name or expert_name.value
        self.call_count = 0

    @classmethod
    def from_actions(
        cls,
        expert_name: ExpertName,
        actions: list[str],
        tool_registry: ToolRegistry,
        *,
        resource_name: str | None = None,
    ) -> ReplayExpertAgent:
        """Encode actions as native Qwen3 calls that pass through strict parsing."""

        return cls(
            expert_name,
            [cls.qwen3_response(action) for action in actions],
            tool_registry,
            resource_name=resource_name,
        )

    @staticmethod
    def qwen3_response(action: str) -> str:
        """Render one native Qwen3 XML tool call."""

        return (
            "<tool_call>\n"
            f"<function={RESTORE_FUNCTION_NAME}>\n"
            "<parameter=action>\n"
            f"{action}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        """Parse the next replay response, falling back to a parsed stop call."""

        step_index = len(state.steps)
        raw_response = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else self.qwen3_response(STOP_ACTION)
        )
        self.call_count += 1
        parse_status, parsed_payload, action, _, error = parse_expert_response(raw_response, self.tool_registry)
        return ExpertDecisionRecord(
            expert_name=self.expert_name,
            step_index=step_index,
            action=action,
            decision_source=ExpertDecisionSource.REPLAY,
            resource_name=self.resource_name,
            parse_status=parse_status,
            api_succeeded=True,
            tool_call_id=f"replay-call-{step_index}" if parse_status == ExpertParseStatus.VALID else None,
            validation_status=(
                ValidationStatus.VALID
                if parse_status == ExpertParseStatus.VALID
                else (
                    ValidationStatus.UNKNOWN_ACTION
                    if parse_status == ExpertParseStatus.UNKNOWN_ACTION
                    else ValidationStatus.INVALID_TOOL_CALL
                )
            ),
            raw_assistant_output=raw_response,
            parsed_payload=parsed_payload,
            error=error,
        )
