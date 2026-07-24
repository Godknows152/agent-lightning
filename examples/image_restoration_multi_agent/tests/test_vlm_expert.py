"""Stage F expert parsing, replay, and OpenAI-compatible request tests."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from agents import ReplayExpertAgent, ScriptedDiagnosisAgent, VLMRestorationExpertAgent, parse_expert_response
from agents.prompts import (
    build_expert_single_step_sft_system_prompt,
    build_expert_single_step_sft_user_prompt,
    build_expert_system_prompt,
)
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
    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
        *,
        provider_token_ids: bool = False,
    ) -> None:
        choice_token_fields = (
            {"provider_specific_fields": {"token_ids": [3, 4]}} if provider_token_ids else {"token_ids": [3, 4]}
        )
        self.payload = {
            "id": "chatcmpl-expert-test",
            "model": "qwen3.5-9b",
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
                    **choice_token_fields,
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.payload


class _FakeCompletions:
    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        *,
        provider_token_ids: bool = False,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.provider_token_ids = provider_token_ids
        self.call_count = 0
        self.last_request: dict[str, Any] | None = None

    def create(self, **request: Any) -> _FakeResponse:
        self.call_count += 1
        self.last_request = request
        return _FakeResponse(self.content, self.tool_calls, provider_token_ids=self.provider_token_ids)


class _FakeClient:
    def __init__(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        *,
        provider_token_ids: bool = False,
    ) -> None:
        self.completions = _FakeCompletions(content, tool_calls, provider_token_ids=provider_token_ids)
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


def _raw_qwen3_call(action: str = "scunet", function_name: str = "restore_image") -> str:
    return (
        "<tool_call>\n"
        f"<function={function_name}>\n"
        "<parameter=action>\n"
        f"{action}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


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
        (_raw_qwen3_call(), ExpertParseStatus.VALID),
        ('{"name":"restore_image","arguments":{"action":"scunet"}}', ExpertParseStatus.INVALID_TOOL_CALL),
        (
            "<tool_call>\n<function=restore_image>\n</function>\n</tool_call>",
            ExpertParseStatus.MISSING_FIELD,
        ),
        (_raw_qwen3_call(function_name="other"), ExpertParseStatus.UNKNOWN_FUNCTION),
        (_raw_qwen3_call(action="not_registered"), ExpertParseStatus.UNKNOWN_ACTION),
        (_raw_qwen3_call() + _raw_qwen3_call(action="stop"), ExpertParseStatus.MULTIPLE_TOOL_CALLS),
        ("<tool_call>\n<function=restore_image>\n</tool_call>", ExpertParseStatus.INVALID_TOOL_CALL),
        (
            "<tool_call>\n<function=restore_image>\n<parameter=action>scunet\n</function>\n</tool_call>",
            ExpertParseStatus.INVALID_TOOL_CALL,
        ),
        (
            "<tool_call>\n<function=restore_image>\n"
            "<parameter=action>scunet</parameter>\n<parameter=action>stop</parameter>\n"
            "</function>\n</tool_call>",
            ExpertParseStatus.INVALID_TOOL_CALL,
        ),
        ("reasoning before\n" + _raw_qwen3_call(), ExpertParseStatus.INVALID_TOOL_CALL),
        ("", ExpertParseStatus.EMPTY_RESPONSE),
    ],
)
def test_parse_expert_response_statuses(raw_response: str, expected_status: ExpertParseStatus) -> None:
    status, _, action, _, _ = parse_expert_response(raw_response, _registry())

    assert status == expected_status
    assert (action is not None) is (expected_status == ExpertParseStatus.VALID)


def test_parse_expert_response_prefers_openai_tool_calls_and_validates_arguments() -> None:
    status, _, action, tool_call_id, _ = parse_expert_response(
        "raw text that must not be used",
        _registry(),
        _parsed_tool_call("scunet"),
    )

    assert status == ExpertParseStatus.VALID
    assert action == "scunet"
    assert tool_call_id == "call-expert-test"


def test_parse_expert_response_rejects_invalid_openai_arguments_json() -> None:
    tool_calls = _parsed_tool_call()
    tool_calls[0]["function"]["arguments"] = "not-json"

    status, _, action, _, _ = parse_expert_response(None, _registry(), tool_calls)

    assert status == ExpertParseStatus.INVALID_JSON
    assert action is None


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
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "return_token_ids": True,
    }
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
    prompt_text = str(messages[-1])
    assert "Historical tool feedback: Step 0: selected action scunet; IQA aggregate_score=0.5000." in prompt_text
    assert "Do not reuse any restoration action already listed in the history." in prompt_text
    assert "current_aggregate_score" not in prompt_text
    assert "raw_scores" not in prompt_text
    assert "step_reward" not in prompt_text
    assert "quality improved" not in prompt_text
    assert messages[0]["content"] == build_expert_system_prompt(ExpertName.FOG, _registry())


def test_vlm_expert_hides_stop_before_minimum_tool_calls(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = _registry()
    client = _FakeClient(None, _parsed_tool_call("scunet"))
    image_path = tmp_path / "latest.png"
    image_path.write_bytes(b"latest-image")
    agent = VLMRestorationExpertAgent(
        config.expert_vlm,
        ExpertName.FOG,
        registry,
        max_steps=config.workflow.max_steps,
        min_stop_tool_calls=5,
        client=client,
    )
    state = _state(image_path, with_history=True)
    state.tool_call_count = 1

    decision = agent.decide(state)

    assert decision.parse_status == ExpertParseStatus.VALID
    request = client.completions.last_request
    assert request is not None
    action_enum = request["tools"][0]["function"]["parameters"]["properties"]["action"]["enum"]
    assert "stop" not in action_enum
    assert "Do not use action stop before it appears in the supplied tool schema" in request["messages"][0]["content"]


def test_vlm_expert_first_turn_exactly_reuses_single_step_sft_prompts(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = _registry()
    client = _FakeClient(None, _parsed_tool_call("scunet"))
    image_path = tmp_path / "initial.png"
    image_path.write_bytes(b"initial-image")
    agent = VLMRestorationExpertAgent(
        config.expert_vlm,
        ExpertName.FOG,
        registry,
        max_steps=config.workflow.max_steps,
        client=client,
    )

    decision = agent.decide(_state(image_path))

    assert decision.parse_status == ExpertParseStatus.VALID
    request = client.completions.last_request
    assert request is not None
    messages = request["messages"]
    assert messages[0]["content"] == build_expert_single_step_sft_system_prompt(ExpertName.FOG, registry)
    user_text = build_expert_single_step_sft_user_prompt().removeprefix("<image>\n")
    assert messages[1]["content"][1] == {"type": "text", "text": user_text}


def test_vlm_expert_retains_provider_specific_token_ids_for_invalid_output(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"invalid-image")
    client = _FakeClient("malformed output", provider_token_ids=True)
    agent = VLMRestorationExpertAgent(
        config.expert_vlm,
        ExpertName.FOG,
        _registry(),
        max_steps=config.workflow.max_steps,
        client=client,
    )

    decision = agent.decide(_state(image_path))

    assert decision.parse_status == ExpertParseStatus.INVALID_TOOL_CALL
    assert decision.generated_token_ids == [3, 4]


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
    assert result.state.final_reward == -config.workflow.invalid_action_penalty
    assert result.state.tool_call_count == 0
    assert result.state.best_image == str(input_path.resolve())
    assert result.state.steps[0].expert_decision.parse_status == ExpertParseStatus.INVALID_TOOL_CALL
    assert result.state.steps[0].reward_components == {
        "invalid_action_penalty": -config.workflow.invalid_action_penalty
    }
    assert result.state.steps[0].output_image is None
