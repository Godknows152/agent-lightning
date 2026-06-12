"""Deterministic state-machine tests for stage B."""

from __future__ import annotations

from pathlib import Path

from config import load_example_config
from factory import DeterministicControllerFactory
from schemas import DegradationType, RestorationTask, ValidationStatus
from tool_registry import ToolRegistry

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def _factory() -> DeterministicControllerFactory:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    return DeterministicControllerFactory(config, ToolRegistry.from_yaml(config.tools_config))


def _input_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"deterministic-image-placeholder")
    return image_path


def test_complete_fixed_expert_workflow_returns_historical_best(tmp_path: Path) -> None:
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.FOG,
        scripted_actions=["restoration_model_a", "restoration_model_b", "stop"],
        score_sequence=[0.4, 0.7, 0.5],
        output_dir=str(tmp_path / "run"),
    )
    controller = _factory().build(task)

    result = controller.run(task, trajectory_id="trajectory-best", trace=False)

    state = result.state
    assert state.terminated is True
    assert state.termination_reason == "expert_stop"
    assert state.diagnosis.route_to.value == "fog_expert"
    assert {step.expert_name.value for step in state.steps} == {"fog_expert"}
    assert state.tool_call_count == 2
    assert state.current_evaluation.aggregate_score == 0.5
    assert state.best_evaluation.aggregate_score == 0.7
    assert state.best_image.endswith("step_000_restoration_model_a.png")
    assert Path(result.trajectory_path).is_file()


def test_unknown_action_is_rejected_before_worker_execution(tmp_path: Path) -> None:
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.RAIN,
        scripted_actions=["not_registered"],
        score_sequence=[0.4],
        output_dir=str(tmp_path / "run"),
    )

    result = _factory().build(task).run(task, trajectory_id="trajectory-invalid", trace=False)

    state = result.state
    assert state.termination_reason == "invalid_action"
    assert state.tool_call_count == 0
    assert state.invalid_action_count == 1
    assert state.steps[0].expert_decision.validation_status == ValidationStatus.UNKNOWN_ACTION
    assert state.steps[0].output_image is None


def test_consecutive_worker_failures_terminate_without_changing_current_image(tmp_path: Path) -> None:
    input_path = _input_image(tmp_path)
    task = RestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.SNOW,
        scripted_actions=["restoration_model_a", "restoration_model_a", "stop"],
        score_sequence=[0.4],
        fail_actions=["restoration_model_a"],
        output_dir=str(tmp_path / "run"),
    )

    result = _factory().build(task).run(task, trajectory_id="trajectory-failure", trace=False)

    state = result.state
    assert state.termination_reason == "consecutive_worker_failures"
    assert state.tool_call_count == 2
    assert state.current_image == str(input_path.resolve())
    assert state.best_image == str(input_path.resolve())
    assert all(not step.success for step in state.steps)


def test_no_improvement_limit_stops_the_loop(tmp_path: Path) -> None:
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.LOW_LIGHT,
        scripted_actions=["restoration_model_a"] * 5,
        score_sequence=[0.5, 0.49, 0.48, 0.47, 0.46],
        output_dir=str(tmp_path / "run"),
    )

    result = _factory().build(task).run(task, trajectory_id="trajectory-no-improvement", trace=False)

    state = result.state
    assert state.termination_reason == "no_improvement_limit"
    assert state.tool_call_count == 3
    assert state.best_image == state.original_image
    assert len(state.steps) == 3


def test_consecutive_evaluation_failures_do_not_replace_current_image(tmp_path: Path) -> None:
    input_path = _input_image(tmp_path)
    task = RestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.RAIN,
        scripted_actions=["restoration_model_a", "restoration_model_b", "stop"],
        score_sequence=[0.5],
        fail_evaluation_indices=[1, 2],
        output_dir=str(tmp_path / "run"),
    )

    result = _factory().build(task).run(task, trajectory_id="trajectory-evaluation-failure", trace=False)

    state = result.state
    assert state.termination_reason == "consecutive_evaluation_failures"
    assert state.current_image == str(input_path.resolve())
    assert state.best_image == str(input_path.resolve())
    assert state.tool_call_count == 2
    assert all(step.output_image is not None for step in state.steps)
    assert all(not step.success for step in state.steps)
