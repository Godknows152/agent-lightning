"""Stage F expert parsing, replay, and OpenAI-compatible request tests."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from agents import ReplayExpertAgent, ScriptedDiagnosisAgent, VLMRestorationExpertAgent, parse_expert_response
from config import load_stage_f_example_config
from controller import ImageRestorationController
from evaluators import ScriptedEvaluator
from schemas import (
    DegradationType,
    DiagnosisResult,
    EvaluationResult,
    ExecutionStatus,
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    RealRestorationTask,
    RestorationStep,
    RestorationTrajectoryState,
)
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def __init__(self, content: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self.payload = {
            "id": "chatcmpl-expert-test",
            "model": "glm-4.1v-9b-thinking",
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "short reasoning",
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "token_ids": [3, 4],
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.payload


class _FakeCompletions:
    def __init__(self, content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.call_count = 0
        self.last_request: dict[str, Any] | None = None

    def create(self, **request: Any) -> _FakeResponse:
        self.call_count += 1
        self.last_request = request
        return _FakeResponse(self.content, self.tool_calls)


class _FakeClient:
    def __init__(self, content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.completions = _FakeCompletions(content, tool_calls)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _registry() -> ToolRegistry:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    return ToolRegistry.from_yaml(config.tools_config)


def _parsed_tool_call(action: str = "scunet", function_name: str = "restore_image") -> list[dict[str, Any]]:
    return [
        {
            "id": "call-expert-test",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps({"action": action}, separators=(",", ":")),
            },
        }
    ]


def _evaluation(score: float, feedback: str = "quality feedback") -> EvaluationResult:
    return EvaluationResult(
        status=ExecutionStatus.SUCCESS,
        raw_scores={"mock": score},
        normalized_scores={"mock": score},
        aggregate_score=score,
        delta_from_previous=0.0,
        delta_from_original=0.0,
        delta_from_best=0.0,
        is_new_best=False,
        feedback=feedback,
    )


def _state(image_path: Path, *, with_history: bool = False) -> RestorationTrajectoryState:
    diagnosis = DiagnosisResult(
        primary_type=DegradationType.FOG,
        visual_evidence=[],
        route_to=ExpertName.FOG,
    )
    evaluation = _evaluation(0.4)
    state = RestorationTrajectoryState(
        trajectory_id="expert-test",
        original_image=str(image_path),
        current_image=str(image_path),
        best_image=str(image_path),
        diagnosis=diagnosis,
        expert_name=ExpertName.FOG,
        original_evaluation=evaluation,
        current_evaluation=evaluation,
        best_evaluation=evaluation,
    )
    if with_history:
        decision = ExpertDecisionRecord(
            expert_name=ExpertName.FOG,
            step_index=0,
            action="scunet",
            decision_source=ExpertDecisionSource.REPLAY,
            tool_call_id="history-call",
        )
        state.steps.append(
            RestorationStep(
                step_index=0,
                expert_name=ExpertName.FOG,
                expert_decision=decision,
                tool_name="scunet",
                input_image=str(image_path),
                output_image=str(image_path),
                evaluation=_evaluation(0.5, "quality improved"),
                step_reward=0.1,
                success=True,
                latency_seconds=0.0,
            )
        )
        state.current_evaluation = _evaluation(0.5, "quality improved")
        state.best_evaluation = state.current_evaluation
    return state


@pytest.mark.parametrize(
    ("raw_response", "expected_status"),
    [
        (
            '<tool_call>{"name":"restore_image","arguments":{"action":"scunet"}}</tool_call>',
            ExpertParseStatus.VALID,
        ),
        ('{"name":"restore_image","arguments":{"action":"scunet"}}', ExpertParseStatus.INVALID_TOOL_CALL),
        (
            '<tool_call>{"name":"restore_image","arguments":{}}</tool_call>',
            ExpertParseStatus.MISSING_FIELD,
        ),
        (
            '<tool_call>{"name":"other","arguments":{"action":"scunet"}}</tool_call>',
            ExpertParseStatus.UNKNOWN_FUNCTION,
        ),
        (
            '<tool_call>{"name":"restore_image","arguments":{"action":"not_registered"}}</tool_call>',
            ExpertParseStatus.UNKNOWN_ACTION,
        ),
        (
            '<tool_call>{"name":"restore_image","arguments":{"action":"scunet"}}</tool_call>'
            '<tool_call>{"name":"restore_image","arguments":{"action":"stop"}}</tool_call>',
            ExpertParseStatus.MULTIPLE_TOOL_CALLS,
        ),
        ("", ExpertParseStatus.EMPTY_RESPONSE),
    ],
)
def test_parse_expert_response_statuses(raw_response: str, expected_status: ExpertParseStatus) -> None:
    status, _, action, _, _ = parse_expert_response(raw_response, _registry())

    assert status == expected_status
    assert (action is not None) is (expected_status == ExpertParseStatus.VALID)


def test_vlm_expert_uses_only_latest_image_and_full_text_history(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    client = _FakeClient(None, _parsed_tool_call("scunet"))
    history_image_path = tmp_path / "history.png"
    history_image_path.write_bytes(b"history-image")
    latest_image_path = tmp_path / "latest.png"
    latest_image_path.write_bytes(b"latest-image")
    agent = VLMRestorationExpertAgent(
        config.expert_vlm,
        ExpertName.FOG,
        _registry(),
        max_steps=config.workflow.max_steps,
        client=client,
    )

    state = _state(history_image_path, with_history=True)
    state.current_image = str(latest_image_path)
    decision = agent.decide(state)

    assert client.completions.call_count == 1
    assert decision.parse_status == ExpertParseStatus.VALID
    assert decision.action == "scunet"
    assert decision.prompt_token_ids == [1, 2]
    assert decision.generated_token_ids == [3, 4]
    request = client.completions.last_request
    assert request is not None
    assert request["tool_choice"] == "auto"
    assert request["tools"] == [_registry().build_tool_schema()]
    messages = request["messages"]
    image_parts = [
        part
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(messages) == 2
    assert len(image_parts) == 1
    image_url = image_parts[0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == b"latest-image"
    assert "current_aggregate_score: 0.500000" in str(messages[-1])
    assert '"action":"scunet"' in str(messages[-1])
    assert '"raw_scores":{"mock":0.5}' in str(messages[-1])
    assert '"step_reward":0.1' in str(messages[-1])
    assert '"feedback":"quality improved"' in str(messages[-1])


def test_replay_uses_strict_parser_and_completes_multistep_controller(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = _registry()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"image")
    experts = {
        expert_name: ReplayExpertAgent.from_actions(
            expert_name,
            ["scunet", "s2former", "stop"] if expert_name == ExpertName.FOG else ["stop"],
            registry,
        )
        for expert_name in ExpertName
    }
    controller = ImageRestorationController(
        settings=config.workflow,
        tool_registry=registry,
        diagnosis_agent=ScriptedDiagnosisAgent(DegradationType.FOG),
        experts=experts,
        worker=CopyRestorationWorker(),
        evaluator=ScriptedEvaluator([0.4, 0.6, 0.7], improvement_epsilon=config.workflow.improvement_epsilon),
    )
    task = RealRestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.FOG,
        scripted_actions=["scunet", "s2former", "stop"],
        output_dir=str(tmp_path / "run"),
    )

    result = controller.run(task, trajectory_id="replay-multistep")

    assert result.state.termination_reason == "expert_stop"
    assert result.state.tool_call_count == 2
    assert [step.expert_decision.decision_source for step in result.state.steps] == [
        ExpertDecisionSource.REPLAY,
        ExpertDecisionSource.REPLAY,
        ExpertDecisionSource.REPLAY,
    ]
    assert all(step.expert_decision.parse_status == ExpertParseStatus.VALID for step in result.state.steps)


def test_invalid_replay_stops_before_worker_and_returns_original(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = _registry()
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"image")
    invalid_expert = ReplayExpertAgent(ExpertName.FOG, ["unstructured answer"], registry)
    experts = {
        expert_name: (
            invalid_expert
            if expert_name == ExpertName.FOG
            else ReplayExpertAgent.from_actions(expert_name, ["stop"], registry)
        )
        for expert_name in ExpertName
    }
    controller = ImageRestorationController(
        settings=config.workflow,
        tool_registry=registry,
        diagnosis_agent=ScriptedDiagnosisAgent(DegradationType.FOG),
        experts=experts,
        worker=CopyRestorationWorker(),
        evaluator=ScriptedEvaluator([0.4], improvement_epsilon=config.workflow.improvement_epsilon),
    )
    task = RealRestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.FOG,
        scripted_actions=["stop"],
        output_dir=str(tmp_path / "run"),
    )

    result = controller.run(task, trajectory_id="replay-invalid")

    assert result.state.termination_reason == "invalid_tool_call"
    assert result.state.tool_call_count == 0
    assert result.state.best_image == str(input_path.resolve())
    assert result.state.steps[0].expert_decision.parse_status == ExpertParseStatus.INVALID_TOOL_CALL
    assert result.state.steps[0].output_image is None
