"""Replay expert decisions through the same Hermes parser used by the VLM."""

from __future__ import annotations

import json

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
    """Return preconfigured raw Hermes responses after strict parsing."""

    def __init__(self, expert_name: ExpertName, responses: list[str], tool_registry: ToolRegistry) -> None:
        self.expert_name = expert_name
        self.responses = list(responses)
        self.tool_registry = tool_registry
        self.call_count = 0

    @classmethod
    def from_actions(
        cls,
        expert_name: ExpertName,
        actions: list[str],
        tool_registry: ToolRegistry,
    ) -> ReplayExpertAgent:
        """Encode actions as raw Hermes calls that still pass through strict parsing."""

        return cls(expert_name, [cls.hermes_response(action) for action in actions], tool_registry)

    @staticmethod
    def hermes_response(action: str) -> str:
        payload = {
            "name": RESTORE_FUNCTION_NAME,
            "arguments": {"action": action},
        }
        return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        """Parse the next replay response, falling back to a parsed stop call."""

        step_index = len(state.steps)
        raw_response = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else self.hermes_response(STOP_ACTION)
        )
        self.call_count += 1
        parse_status, parsed_payload, action, _, error = parse_expert_response(raw_response, self.tool_registry)
        return ExpertDecisionRecord(
            expert_name=self.expert_name,
            step_index=step_index,
            action=action,
            decision_source=ExpertDecisionSource.REPLAY,
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
