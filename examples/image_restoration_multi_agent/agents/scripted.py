"""Deterministic diagnosis and expert agents used before model integration."""

from __future__ import annotations

import json

from schemas import (
    DEGRADATION_TO_EXPERT,
    DegradationType,
    DiagnosisResult,
    ExpertDecisionRecord,
    ExpertName,
    RestorationTrajectoryState,
)


class ScriptedDiagnosisAgent:
    """Return the task-provided degradation exactly once per controller run."""

    def __init__(self, degradation_type: DegradationType, visual_evidence: list[str] | None = None) -> None:
        self.degradation_type = degradation_type
        self.visual_evidence = visual_evidence or ["scripted diagnosis for stage A-C validation"]
        self.call_count = 0

    def diagnose(self, image_path: str) -> DiagnosisResult:
        """Return the deterministic diagnosis for an existing image path."""

        del image_path
        self.call_count += 1
        return DiagnosisResult(
            primary_type=self.degradation_type,
            visual_evidence=self.visual_evidence,
            route_to=DEGRADATION_TO_EXPERT[self.degradation_type],
        )


class ScriptedExpertAgent:
    """Emit one preconfigured restore_image action per decision step."""

    def __init__(self, expert_name: ExpertName, actions: list[str]) -> None:
        self.expert_name = expert_name
        self.actions = list(actions)
        self.call_count = 0

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        """Return the next scripted action, falling back to stop when exhausted."""

        action = self.actions[self.call_count] if self.call_count < len(self.actions) else "stop"
        step_index = len(state.steps)
        self.call_count += 1
        arguments = json.dumps({"action": action}, separators=(",", ":"))
        raw_output = json.dumps(
            {
                "id": f"scripted-call-{step_index}",
                "type": "function",
                "function": {"name": "restore_image", "arguments": arguments},
            },
            separators=(",", ":"),
        )
        return ExpertDecisionRecord(
            expert_name=self.expert_name,
            step_index=step_index,
            action=action,
            tool_call_id=f"scripted-call-{step_index}",
            llm_response_id=None,
            raw_assistant_output=raw_output,
            generated_token_ids=None,
        )
