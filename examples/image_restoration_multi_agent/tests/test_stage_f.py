"""Agent Lightning integration tests for the stage F decision modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from agents import ScriptedDiagnosisAgent, ScriptedExpertAgent, VLMDegradationDiagnosisAgent
from config import load_stage_f_example_config
from controller import ExpertAgent, ImageRestorationController
from evaluators import ScriptedEvaluator
from factory import RealControllerFactory
from lit_agent import StageFImageRestorationAgent, VLMImageRestorationAgent
from schemas import (
    DiagnosisParseStatus,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    RealRestorationTask,
    RoutingSource,
)
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "id": "chatcmpl-diagnosis",
            "model": "qwen3.5-9b",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "diagnosis-call",
                                "type": "function",
                                "function": {
                                    "name": "diagnose_degradation",
                                    "arguments": json.dumps(
                                        {"primary_type": "fog", "visual_evidence": ["low contrast"]},
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }


class _FakeCompletions:
    def create(self, **request: Any) -> _FakeResponse:
        del request
        return _FakeResponse()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


class _InvalidResponse:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "id": "chatcmpl-invalid-diagnosis",
            "model": "qwen3.5-9b",
            "choices": [
                {
                    "message": {"content": "still thinking", "tool_calls": None},
                    "finish_reason": "length",
                }
            ],
            "usage": {},
        }


class _InvalidCompletions:
    def create(self, **request: Any) -> _InvalidResponse:
        del request
        return _InvalidResponse()


class _InvalidClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _InvalidCompletions()})()


class _CopyControllerFactory(RealControllerFactory):
    """Use stage F controller wiring without loading GPU models in unit tests."""

    def build(
        self,
        task: RealRestorationTask,
        *,
        routed_expert_override: ExpertName | None = None,
        experts_override: Mapping[ExpertName, ExpertAgent] | None = None,
    ) -> ImageRestorationController:
        routed_expert = routed_expert_override or ExpertName.FOG
        experts = experts_override or {
            expert_name: ScriptedExpertAgent(
                expert_name,
                task.scripted_actions if expert_name == routed_expert else ["stop"],
            )
            for expert_name in ExpertName
        }
        return ImageRestorationController(
            settings=self.config.workflow,
            tool_registry=self.tool_registry,
            diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type),
            experts=experts,
            worker=CopyRestorationWorker(),
            evaluator=ScriptedEvaluator(
                [0.4, 0.6],
                improvement_epsilon=self.config.workflow.improvement_epsilon,
            ),
        )


@pytest.mark.asyncio
async def test_stage_e_oracle_observe_completes_when_vlm_output_is_invalid(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    agent = VLMImageRestorationAgent(
        config,
        _CopyControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm, client=_InvalidClient()),
    )
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"image")
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path),
        "degradation_type": "fog",
        "scripted_actions": ["scunet", "stop"],
        "output_dir": str(tmp_path / "stage_e_run"),
        "routing_mode": "oracle_observe",
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")

    result = agent.results[rollout.rollout_id]
    assert rollout.status == "succeeded"
    assert result.diagnosis_attempt.parse_status == DiagnosisParseStatus.INVALID_TOOL_CALL
    assert result.routing_source == RoutingSource.ORACLE_LABEL
    assert result.workflow_result is not None
    assert result.workflow_result.state.termination_reason == "expert_stop"


@pytest.mark.asyncio
async def test_stage_f_replay_records_multistep_trace_and_one_reward(tmp_path: Path) -> None:
    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    factory = _CopyControllerFactory(config, registry)
    agent = StageFImageRestorationAgent(
        config,
        factory,
        VLMDegradationDiagnosisAgent(config.vlm, client=_FakeClient()),
    )
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"image")
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path),
        "degradation_type": "fog",
        "scripted_actions": ["scunet", "stop"],
        "output_dir": str(tmp_path / "run"),
        "routing_mode": "oracle_observe",
        "expert_decision_mode": "replay",
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    result = agent.results[rollout.rollout_id]
    assert result.workflow_result is not None
    decisions = [step.expert_decision for step in result.workflow_result.state.steps]
    operation_names = {
        span.attributes.get("agentlightning.operation.name")
        for span in spans
        if span.attributes and span.attributes.get("agentlightning.operation.name")
    }
    assert rollout.status == "succeeded"
    assert result.termination_reason == "expert_stop"
    assert [decision.decision_source for decision in decisions] == [
        ExpertDecisionSource.REPLAY,
        ExpertDecisionSource.REPLAY,
    ]
    assert all(decision.parse_status == ExpertParseStatus.VALID for decision in decisions)
    assert {
        "diagnosis_agent.vlm_prediction",
        "routing.select",
        "fog_expert.decision",
        "restoration_worker.scunet",
        "iqa_evaluator.score",
    }.issubset(operation_names)
    assert agl.find_final_reward(spans) == result.final_reward
