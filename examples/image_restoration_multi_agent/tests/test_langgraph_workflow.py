"""No-model LangGraph parity, routing, checkpoint, and failure tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agents import ReplayExpertAgent, ScriptedDiagnosisAgent
from config import ExampleConfig, WorkflowSettings, load_example_config
from evaluators import ScriptedEvaluator
from exceptions import EvaluationError
from factory import DeterministicControllerFactory
from graph import GRAPH_SCHEMA_VERSION, LangGraphImageRestorationWorkflow, RestorationGraphRuntime
from langgraph_factory import DeterministicLangGraphFactory
from schemas import (
    DEGRADATION_TO_EXPERT,
    DegradationType,
    ExpertName,
    RestorationResult,
    RestorationTask,
    RestorationTrajectoryState,
)
from tool_registry import ToolRegistry
from workers import CopyRestorationWorker

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Scenario:
    name: str
    degradation_type: DegradationType
    actions: list[str]
    scores: list[float]
    settings: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    fail_actions: list[str] = field(default_factory=lambda: list[str]())
    fail_evaluation_indices: list[int] = field(default_factory=lambda: list[int]())


SCENARIOS = [
    _Scenario("historical_best", DegradationType.FOG, ["scunet", "s2former", "stop"], [0.4, 0.7, 0.5]),
    _Scenario("unknown_action", DegradationType.RAIN, ["not_registered"], [0.4]),
    _Scenario(
        "worker_failures",
        DegradationType.SNOW,
        ["scunet", "scunet", "stop"],
        [0.4],
        fail_actions=["scunet"],
    ),
    _Scenario(
        "no_improvement",
        DegradationType.LOW_LIGHT,
        ["scunet"] * 5,
        [0.5, 0.49, 0.48, 0.47, 0.46],
    ),
    _Scenario(
        "no_improvement_disabled",
        DegradationType.LOW_LIGHT,
        ["scunet"] * 4,
        [0.5, 0.49, 0.48, 0.47, 0.46],
        settings={"max_steps": 4, "no_improvement_limit": None},
    ),
    _Scenario(
        "evaluation_failures",
        DegradationType.RAIN,
        ["scunet", "s2former", "stop"],
        [0.5],
        fail_evaluation_indices=[1, 2],
    ),
    _Scenario(
        "step_rewards",
        DegradationType.FOG,
        ["scunet", "s2former", "stop"],
        [0.0, 0.2, 0.25],
        settings={
            "reward_mode": "step_iqa_sum_v1",
            "reward_alpha": 0.5,
            "reward_scale": 5.0,
            "tool_call_cost": 0.05,
            "stop_min_best_gain": 0.05,
            "valid_stop_reward": 0.25,
            "premature_stop_penalty": 1.0,
            "forced_termination_penalty": 0.5,
        },
    ),
    _Scenario(
        "immediate_stop",
        DegradationType.RAIN,
        ["stop"],
        [0.0],
        settings={
            "reward_mode": "step_iqa_sum_v1",
            "stop_min_tool_calls": 1,
            "premature_stop_penalty": 1.0,
        },
    ),
    _Scenario(
        "tool_bonus",
        DegradationType.FOG,
        ["scunet", "stop"],
        [0.0, 0.2],
        settings={
            "reward_mode": "step_iqa_sum_v1",
            "reward_alpha": 0.5,
            "reward_scale": 5.0,
            "tool_call_reward": 0.2,
            "tool_call_cost": 0.0,
            "premature_stop_penalty": 0.0,
            "stop_min_tool_calls": 3,
        },
    ),
    _Scenario(
        "invalid_after_gain",
        DegradationType.FOG,
        ["scunet", "not_registered"],
        [0.0, 0.8],
        settings={
            "reward_mode": "step_iqa_sum_v1",
            "reward_alpha": 0.5,
            "reward_scale": 5.0,
            "invalid_action_penalty": 10.0,
        },
    ),
]


def _input_image(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    path.write_bytes(b"langgraph-deterministic-image")
    return path


def _config(settings: dict[str, Any]) -> ExampleConfig:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    if settings:
        config.workflow = config.workflow.model_copy(update=settings)
    return config


def _task(scenario: _Scenario, image_path: Path, output_dir: Path) -> RestorationTask:
    return RestorationTask(
        image_path=str(image_path),
        degradation_type=scenario.degradation_type,
        scripted_actions=scenario.actions,
        score_sequence=scenario.scores,
        fail_actions=scenario.fail_actions,
        fail_evaluation_indices=scenario.fail_evaluation_indices,
        output_dir=str(output_dir),
    )


def _basename(path: str | None) -> str | None:
    return Path(path).name if path is not None else None


def _canonical(state: RestorationTrajectoryState) -> dict[str, Any]:
    if state.final_reward is None:
        raise ValueError("canonical comparison requires final_reward")
    return {
        "expert_name": state.expert_name.value,
        "termination_reason": state.termination_reason,
        "tool_call_count": state.tool_call_count,
        "invalid_action_count": state.invalid_action_count,
        "consecutive_failures": state.consecutive_failures,
        "consecutive_no_improvement": state.consecutive_no_improvement,
        "current_image": _basename(state.current_image),
        "best_image": _basename(state.best_image),
        "original_score": round(state.original_evaluation.aggregate_score, 8),
        "current_score": round(state.current_evaluation.aggregate_score, 8),
        "best_score": round(state.best_evaluation.aggregate_score, 8),
        "final_reward": round(state.final_reward, 8),
        "steps": [
            {
                "step_index": step.step_index,
                "expert_name": step.expert_name.value,
                "validation_status": step.expert_decision.validation_status.value,
                "tool_name": step.tool_name,
                "input_image": _basename(step.input_image),
                "output_image": _basename(step.output_image),
                "success": step.success,
                "step_reward": round(step.step_reward, 8),
                "reward_components": {key: round(value, 8) for key, value in step.reward_components.items()},
                "error": step.error,
            }
            for step in state.steps
        ],
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_langgraph_matches_controller_business_results(tmp_path: Path, scenario: _Scenario) -> None:
    image_path = _input_image(tmp_path)
    controller_config = _config(scenario.settings)
    graph_config = _config(scenario.settings)
    controller_registry = ToolRegistry.from_yaml(controller_config.tools_config)
    graph_registry = ToolRegistry.from_yaml(graph_config.tools_config)
    controller_task = _task(scenario, image_path, tmp_path / "controller")
    graph_task = _task(scenario, image_path, tmp_path / "langgraph")

    controller_result = (
        DeterministicControllerFactory(controller_config, controller_registry)
        .build(controller_task)
        .run(
            controller_task,
            trajectory_id=f"controller-{scenario.name}",
            trace=False,
        )
    )
    graph = DeterministicLangGraphFactory(graph_config, graph_registry).build(graph_task)
    graph_result = graph.invoke(graph_task, trajectory_id=f"graph-{scenario.name}")

    assert _canonical(graph_result.state) == _canonical(controller_result.state)
    assert Path(graph_result.trajectory_path).is_file()


def test_graph_exposes_four_fixed_expert_subgraphs_and_json_checkpoint(tmp_path: Path) -> None:
    scenario = SCENARIOS[0]
    config = _config({})
    registry = ToolRegistry.from_yaml(config.tools_config)
    task = _task(scenario, _input_image(tmp_path), tmp_path / "graph")
    workflow = DeterministicLangGraphFactory(config, registry).build(task)

    result = workflow.invoke(task, trajectory_id="checkpoint-test")
    checkpoint = workflow.get_checkpoint("checkpoint-test")
    checkpoint_trajectory = checkpoint.get("trajectory")
    checkpoint_result = checkpoint.get("result")

    assert {expert_name.value for expert_name in ExpertName}.issubset(workflow.node_names())
    assert checkpoint.get("schema_version") == GRAPH_SCHEMA_VERSION
    assert checkpoint_trajectory is not None
    assert checkpoint_trajectory["termination_reason"] == result.state.termination_reason
    assert checkpoint_result is not None
    assert checkpoint_result["summary"] == result.summary
    json.dumps(checkpoint)


class _CountingWorker(CopyRestorationWorker):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def restore(self, action: str, input_path: str, output_dir: str, step_index: int) -> RestorationResult:
        self.call_count += 1
        return super().restore(action, input_path, output_dir, step_index)


def _manual_runtime(
    *,
    task: RestorationTask,
    registry: ToolRegistry,
    settings: WorkflowSettings,
    routed_responses: list[str],
    evaluator: ScriptedEvaluator,
    worker: CopyRestorationWorker,
) -> RestorationGraphRuntime:
    routed_expert = DEGRADATION_TO_EXPERT[task.degradation_type]
    experts = {
        expert_name: ReplayExpertAgent(
            expert_name,
            routed_responses if expert_name == routed_expert else [ReplayExpertAgent.qwen3_response("stop")],
            registry,
        )
        for expert_name in ExpertName
    }
    return RestorationGraphRuntime(
        settings=settings,
        tool_registry=registry,
        diagnosis_agent=ScriptedDiagnosisAgent(task.degradation_type),
        experts=experts,
        worker=worker,
        evaluator=evaluator,
    )


def test_invalid_replay_output_terminates_before_worker(tmp_path: Path) -> None:
    config = _config({})
    registry = ToolRegistry.from_yaml(config.tools_config)
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.FOG,
        scripted_actions=["stop"],
        score_sequence=[0.4],
        output_dir=str(tmp_path / "invalid"),
    )
    worker = _CountingWorker()
    runtime = _manual_runtime(
        task=task,
        registry=registry,
        settings=config.workflow,
        routed_responses=["unstructured answer"],
        evaluator=ScriptedEvaluator([0.4], improvement_epsilon=config.workflow.improvement_epsilon),
        worker=worker,
    )

    result = LangGraphImageRestorationWorkflow(runtime).invoke(task, trajectory_id="invalid-replay")

    assert result.state.termination_reason == "invalid_tool_call"
    assert result.state.tool_call_count == 0
    assert result.state.invalid_action_count == 1
    assert worker.call_count == 0


def test_original_evaluation_failure_is_not_silently_routed(tmp_path: Path) -> None:
    config = _config({})
    registry = ToolRegistry.from_yaml(config.tools_config)
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.SNOW,
        scripted_actions=["stop"],
        score_sequence=[0.4],
        output_dir=str(tmp_path / "original-failure"),
    )
    runtime = _manual_runtime(
        task=task,
        registry=registry,
        settings=config.workflow,
        routed_responses=[ReplayExpertAgent.qwen3_response("stop")],
        evaluator=ScriptedEvaluator(
            [0.4],
            improvement_epsilon=config.workflow.improvement_epsilon,
            fail_indices={0},
        ),
        worker=CopyRestorationWorker(),
    )

    with pytest.raises(EvaluationError, match="scripted evaluation failure at index 0"):
        LangGraphImageRestorationWorkflow(runtime).invoke(task, trajectory_id="original-score-failure")
