"""Stage E VLM parsing, tracing, and diagnosis-failure tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents import VLMDegradationDiagnosisAgent, parse_diagnosis_response
from config import load_stage_e_example_config
from diagnosis_metrics import build_diagnosis_metrics
from factory import RealControllerFactory
from lit_agent import VLMImageRestorationAgent
from schemas import DegradationType, DiagnosisParseStatus, ExpertName, VLMDiagnosisAttempt
from tool_registry import ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def __init__(self, content: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self.payload = {
            "id": "chatcmpl-test",
            "model": "glm-4.1v-9b-thinking",
            "prompt_token_ids": [1, 2, 3],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "token_ids": [4, 5],
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
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


def _parsed_tool_call(primary_type: str = "fog", function_name: str = "diagnose_degradation") -> list[dict[str, Any]]:
    return [
        {
            "id": "call-test",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(
                    {"primary_type": primary_type, "visual_evidence": ["visible evidence"]},
                    separators=(",", ":"),
                ),
            },
        }
    ]


@pytest.mark.parametrize(
    ("raw_response", "expected_status"),
    [
        (
            '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"fog","visual_evidence":["low contrast"]}}</tool_call>',
            DiagnosisParseStatus.VALID,
        ),
        (
            '{"name":"diagnose_degradation","arguments":{"primary_type":"snow","visual_evidence":["flakes"]}}',
            DiagnosisParseStatus.INVALID_TOOL_CALL,
        ),
        ("not json", DiagnosisParseStatus.INVALID_TOOL_CALL),
        (
            '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"fog"}}</tool_call>',
            DiagnosisParseStatus.MISSING_FIELD,
        ),
        (
            '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"haze","visual_evidence":[]}}</tool_call>',
            DiagnosisParseStatus.INVALID_CATEGORY,
        ),
        (
            '<tool_call>{"name":"restore_image","arguments":{"primary_type":"fog","visual_evidence":[]}}</tool_call>',
            DiagnosisParseStatus.UNKNOWN_FUNCTION,
        ),
        (
            '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"fog","visual_evidence":[]}}</tool_call>'
            '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"rain","visual_evidence":[]}}</tool_call>',
            DiagnosisParseStatus.MULTIPLE_TOOL_CALLS,
        ),
        ("", DiagnosisParseStatus.EMPTY_RESPONSE),
    ],
)
def test_parse_diagnosis_response_statuses(raw_response: str, expected_status: DiagnosisParseStatus) -> None:
    status, _, diagnosis, _ = parse_diagnosis_response(raw_response)

    assert status == expected_status
    assert (diagnosis is not None) is (expected_status == DiagnosisParseStatus.VALID)


def test_vlm_adapter_calls_once_and_retains_token_ids(tmp_path: Path) -> None:
    config = load_stage_e_example_config(EXAMPLE_DIR / "config" / "stage_e.yaml")
    client = _FakeClient(None, _parsed_tool_call("rain"))
    image_path = tmp_path / "neutral_input.png"
    image_path.write_bytes(b"image-bytes")
    agent = VLMDegradationDiagnosisAgent(config.vlm, client=client)

    attempt = agent.diagnose(str(image_path))

    assert client.completions.call_count == 1
    assert attempt.parse_status == DiagnosisParseStatus.VALID
    assert attempt.prompt_token_ids == [1, 2, 3]
    assert attempt.generated_token_ids == [4, 5]
    assert attempt.diagnosis is not None
    assert attempt.diagnosis.route_to == ExpertName.RAIN
    request = client.completions.last_request
    assert request is not None
    request_text = str(request)
    assert str(image_path) not in request_text
    assert "return_token_ids" in request_text
    assert "diagnose_degradation" in request_text
    assert "route_to" not in str(request["tools"])
    assert request["tool_choice"] == "auto"


def test_diagnosis_metrics_count_invalid_responses() -> None:
    valid_status, payload, diagnosis, _ = parse_diagnosis_response(
        '<tool_call>{"name":"diagnose_degradation","arguments":{"primary_type":"fog","visual_evidence":[]}}</tool_call>'
    )
    config = load_stage_e_example_config(EXAMPLE_DIR / "config" / "stage_e.yaml")
    valid_attempt = VLMDiagnosisAttempt(
        backend=config.vlm.backend,
        parse_status=valid_status,
        api_succeeded=True,
        parsed_payload=payload,
        diagnosis=diagnosis,
        latency_seconds=0.1,
    )
    invalid_attempt = VLMDiagnosisAttempt(
        backend=config.vlm.backend,
        parse_status=DiagnosisParseStatus.REQUEST_FAILED,
        api_succeeded=False,
        latency_seconds=0.2,
        error="failed",
    )

    metrics = build_diagnosis_metrics([(DegradationType.FOG, valid_attempt), (DegradationType.RAIN, invalid_attempt)])

    assert metrics["api_success_rate"] == 0.5
    assert metrics["parse_success_rate"] == 0.5
    assert metrics["accuracy_on_valid"] == 1.0
    assert metrics["invalid_by_true"]["rain"] == 1


@pytest.mark.asyncio
async def test_predicted_strict_invalid_response_stops_before_real_workers(tmp_path: Path) -> None:
    config = load_stage_e_example_config(EXAMPLE_DIR / "config" / "stage_e.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    client = _FakeClient("unstructured answer")
    agent = VLMImageRestorationAgent(
        config,
        RealControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm, client=client),
    )
    input_path = tmp_path / "neutral.png"
    input_path.write_bytes(b"image-placeholder")
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path),
        "degradation_type": "fog",
        "scripted_actions": ["focalnet_dehaze", "stop"],
        "output_dir": str(tmp_path / "run"),
        "routing_mode": "predicted_strict",
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    result = agent.results[rollout.rollout_id]
    operation_names = {
        span.attributes.get("agentlightning.operation.name")
        for span in spans
        if span.attributes and span.attributes.get("agentlightning.operation.name")
    }
    assert rollout.status == "succeeded"
    assert client.completions.call_count == 1
    assert result.termination_reason == "diagnosis_failed"
    assert result.workflow_result is None
    assert result.final_reward == -config.vlm.diagnosis_failure_penalty
    assert "diagnosis_agent.vlm_prediction" in operation_names
    assert "routing.rejected" in operation_names
    assert not any(str(name).startswith("restoration_worker") for name in operation_names)
