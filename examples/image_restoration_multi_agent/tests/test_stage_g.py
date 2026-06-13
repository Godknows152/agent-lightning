"""Four-expert routing, resource isolation, and trace tests for stage G."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from agents import ScriptedDiagnosisAgent, VLMDegradationDiagnosisAgent
from config import StageGExampleConfig, load_stage_g_example_config
from controller import ExpertAgent, ImageRestorationController
from evaluators import ScriptedEvaluator
from factory import RealControllerFactory
from lit_agent import StageGImageRestorationAgent
from schemas import DEGRADATION_TO_EXPERT, DegradationType, ExpertName, RealRestorationTask
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


class _DiagnosisResponse:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "id": "chatcmpl-stage-g-diagnosis",
            "model": "glm-4.1v-9b-thinking",
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
                                        {"primary_type": "fog", "visual_evidence": ["test evidence"]},
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


class _DiagnosisCompletions:
    def create(self, **request: Any) -> _DiagnosisResponse:
        del request
        return _DiagnosisResponse()


class _DiagnosisClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _DiagnosisCompletions()})()


class _InvalidExpertResponse:
    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "id": "chatcmpl-stage-g-expert",
            "model": "glm-4.1v-9b-thinking",
            "choices": [
                {
                    "message": {"content": "untrained response", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }


class _RecordingExpertCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> _InvalidExpertResponse:
        self.requests.append(request)
        return _InvalidExpertResponse()


class _RecordingExpertClient:
    def __init__(self) -> None:
        self.completions = _RecordingExpertCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class _CopyControllerFactory(RealControllerFactory):
    """Use stage G routing with deterministic worker and evaluator components."""

    def build(
        self,
        task: RealRestorationTask,
        *,
        routed_expert_override: ExpertName | None = None,
        experts_override: Mapping[ExpertName, ExpertAgent] | None = None,
    ) -> ImageRestorationController:
        if experts_override is None:
            raise ValueError("stage G tests require injected experts")
        return ImageRestorationController(
            settings=self.config.workflow,
            tool_registry=self.tool_registry,
            diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type),
            experts=experts_override,
            worker=CopyRestorationWorker(),
            evaluator=ScriptedEvaluator(
                [0.4, 0.6],
                improvement_epsilon=self.config.workflow.improvement_epsilon,
            ),
        )


def _config_and_registry() -> tuple[StageGExampleConfig, ToolRegistry]:
    config = load_stage_g_example_config(EXAMPLE_DIR / "config" / "stage_g.yaml")
    return config, ToolRegistry.from_yaml(config.tools_config)


def _task(tmp_path: Path, degradation_type: DegradationType, actions: list[str], mode: str) -> dict[str, Any]:
    image_path = tmp_path / f"{degradation_type.value}.png"
    image_path.write_bytes(b"image")
    return {
        "image_path": str(image_path),
        "degradation_type": degradation_type.value,
        "scripted_actions": actions,
        "output_dir": str(tmp_path / "outputs" / degradation_type.value),
        "routing_mode": "oracle_observe",
        "expert_decision_mode": mode,
    }


def test_stage_g_config_defines_four_unique_resources_and_one_tool_registry() -> None:
    config, registry = _config_and_registry()

    assert set(config.expert_resources) == set(ExpertName)
    assert len({resource.resource_name for resource in config.expert_resources.values()}) == 4
    assert {expert.tool_registry for expert in config.experts.values()} == {"all_restoration_tools"}
    assert len(registry.actions) == 17
    assert all(resource.policy_path is None for resource in config.expert_resources.values())


@pytest.mark.asyncio
async def test_all_four_replay_routes_use_parallel_expert_resources_and_isolated_outputs(tmp_path: Path) -> None:
    config, registry = _config_and_registry()
    agent = StageGImageRestorationAgent(
        config,
        _CopyControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm, client=_DiagnosisClient()),
    )
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    trajectory_paths: set[str] = set()

    with runner.run_context(agent=agent, store=store):
        for degradation_type in DegradationType:
            rollout = await runner.step(
                _task(tmp_path, degradation_type, ["scunet", "stop"], "replay"),
                resources={},
                mode="val",
            )
            attempts = await store.query_attempts(rollout.rollout_id)
            spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)
            result = agent.results[rollout.rollout_id]
            assert result.workflow_result is not None
            state = result.workflow_result.state
            expected_expert = DEGRADATION_TO_EXPERT[degradation_type]
            expected_resource = config.expert_resources[expected_expert].resource_name
            expert_spans = [
                span
                for span in spans
                if span.attributes and span.attributes.get("agent.name") in {expert.value for expert in ExpertName}
            ]

            assert rollout.status == "succeeded"
            assert state.expert_name == expected_expert
            assert result.termination_reason == "expert_stop"
            assert {step.expert_name for step in state.steps} == {expected_expert}
            assert {step.expert_decision.resource_name for step in state.steps} == {expected_resource}
            assert {span.attributes.get("agent.name") for span in expert_spans} == {expected_expert.value}
            assert all(
                span.attributes.get("agentlightning.operation.input.resource_name") == expected_resource
                for span in expert_spans
            )
            assert Path(result.result_path).name == "stage_g_result.json"
            assert Path(result.result_path).parent.name == degradation_type.value
            trajectory_paths.add(result.workflow_result.trajectory_path)

    assert len(trajectory_paths) == 4


@pytest.mark.asyncio
async def test_one_expert_failure_does_not_leak_into_next_expert_rollout(tmp_path: Path) -> None:
    config, registry = _config_and_registry()
    agent = StageGImageRestorationAgent(
        config,
        _CopyControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm, client=_DiagnosisClient()),
    )
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()

    with runner.run_context(agent=agent, store=store):
        failed_rollout = await runner.step(
            _task(tmp_path, DegradationType.FOG, ["not_registered"], "replay"),
            resources={},
            mode="val",
        )
        successful_rollout = await runner.step(
            _task(tmp_path, DegradationType.SNOW, ["scunet", "stop"], "replay"),
            resources={},
            mode="val",
        )

    failed = agent.results[failed_rollout.rollout_id]
    successful = agent.results[successful_rollout.rollout_id]
    assert failed.termination_reason == "invalid_action"
    assert successful.termination_reason == "expert_stop"
    assert successful.workflow_result is not None
    assert successful.workflow_result.state.expert_name == ExpertName.SNOW
    assert successful.workflow_result.state.tool_call_count == 1


@pytest.mark.asyncio
async def test_all_four_strict_vlm_interfaces_record_identity_and_reject_invalid_output(tmp_path: Path) -> None:
    config, registry = _config_and_registry()
    expert_client = _RecordingExpertClient()
    agent = StageGImageRestorationAgent(
        config,
        _CopyControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm, client=_DiagnosisClient()),
        expert_client=expert_client,
    )
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()

    with runner.run_context(agent=agent, store=store):
        for degradation_type in DegradationType:
            rollout = await runner.step(
                _task(tmp_path, degradation_type, ["stop"], "vlm_strict"),
                resources={},
                mode="val",
            )
            attempts = await store.query_attempts(rollout.rollout_id)
            spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)
            result = agent.results[rollout.rollout_id]
            assert result.workflow_result is not None
            state = result.workflow_result.state
            expected_expert = DEGRADATION_TO_EXPERT[degradation_type]
            expected_resource = config.expert_resources[expected_expert].resource_name
            decision = state.steps[0].expert_decision
            operation_names = {
                span.attributes.get("agentlightning.operation.name")
                for span in spans
                if span.attributes and span.attributes.get("agentlightning.operation.name")
            }

            assert result.termination_reason == "invalid_tool_call"
            assert decision.expert_name == expected_expert
            assert decision.resource_name == expected_resource
            assert decision.action is None
            assert not any(str(name).startswith("restoration_worker") for name in operation_names)

    assert len(expert_client.completions.requests) == 4
    for expert_name, request in zip(ExpertName, expert_client.completions.requests, strict=True):
        assert expert_name.value in request["messages"][0]["content"]
        assert request["model"] == config.expert_resources[expert_name].served_model_name
        assert request["tools"] == [registry.build_tool_schema()]
