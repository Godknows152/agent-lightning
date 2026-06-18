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
        scripted_actions=["scunet", "s2former", "stop"],
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
    assert state.best_image.endswith("step_000_scunet.png")
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
        scripted_actions=["scunet", "scunet", "stop"],
        score_sequence=[0.4],
        fail_actions=["scunet"],
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
        scripted_actions=["scunet"] * 5,
        score_sequence=[0.5, 0.49, 0.48, 0.47, 0.46],
        output_dir=str(tmp_path / "run"),
    )

    result = _factory().build(task).run(task, trajectory_id="trajectory-no-improvement", trace=False)

    state = result.state
    assert state.termination_reason == "no_improvement_limit"
    assert state.tool_call_count == 3
    assert state.best_image == state.original_image
    assert len(state.steps) == 3


def test_no_improvement_limit_can_be_disabled(tmp_path: Path) -> None:
    factory = _factory()
    factory.config.workflow = factory.config.workflow.model_copy(
        update={
            "max_steps": 4,
            "no_improvement_limit": None,
        }
    )
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.LOW_LIGHT,
        scripted_actions=["scunet"] * 4,
        score_sequence=[0.5, 0.49, 0.48, 0.47, 0.46],
        output_dir=str(tmp_path / "run"),
    )

    result = factory.build(task).run(task, trajectory_id="trajectory-no-improvement-disabled", trace=False)

    state = result.state
    assert state.termination_reason == "max_steps"
    assert state.tool_call_count == 4
    assert state.consecutive_no_improvement == 4


def test_consecutive_evaluation_failures_do_not_replace_current_image(tmp_path: Path) -> None:
    input_path = _input_image(tmp_path)
    task = RestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.RAIN,
        scripted_actions=["scunet", "s2former", "stop"],
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


def test_step_iqa_rewards_sum_once_at_trajectory_level(tmp_path: Path) -> None:
    factory = _factory()
    factory.config.workflow = factory.config.workflow.model_copy(
        update={
            "reward_mode": "step_iqa_sum_v1",
            "reward_alpha": 0.5,
            "reward_scale": 5.0,
            "tool_call_cost": 0.05,
            "stop_min_best_gain": 0.05,
            "valid_stop_reward": 0.25,
            "premature_stop_penalty": 1.0,
            "forced_termination_penalty": 0.5,
        }
    )
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.FOG,
        scripted_actions=["scunet", "s2former", "stop"],
        score_sequence=[0.0, 0.2, 0.25],
        output_dir=str(tmp_path / "run"),
    )

    state = factory.build(task).run(task, trajectory_id="trajectory-grpo", trace=False).state

    assert [round(step.step_reward, 6) for step in state.steps] == [0.95, 0.7, 0.25]
    assert state.final_reward == 1.9
    assert state.steps[-1].reward_components["valid_stop"] == 1.0


def test_step_iqa_reward_penalizes_immediate_stop(tmp_path: Path) -> None:
    factory = _factory()
    factory.config.workflow = factory.config.workflow.model_copy(
        update={
            "reward_mode": "step_iqa_sum_v1",
            "stop_min_tool_calls": 1,
            "premature_stop_penalty": 1.0,
        }
    )
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.RAIN,
        scripted_actions=["stop"],
        score_sequence=[0.0],
        output_dir=str(tmp_path / "run"),
    )

    state = factory.build(task).run(task, trajectory_id="trajectory-stop", trace=False).state

    assert state.final_reward == -1.0
    assert state.steps[0].reward_components["valid_stop"] == 0.0


def test_invalid_action_overrides_prior_iqa_gains_with_terminal_penalty(tmp_path: Path) -> None:
    factory = _factory()
    factory.config.workflow = factory.config.workflow.model_copy(
        update={
            "reward_mode": "step_iqa_sum_v1",
            "reward_alpha": 0.5,
            "reward_scale": 5.0,
            "invalid_action_penalty": 10.0,
        }
    )
    task = RestorationTask(
        image_path=str(_input_image(tmp_path)),
        degradation_type=DegradationType.FOG,
        scripted_actions=["scunet", "not_registered"],
        score_sequence=[0.0, 0.8],
        output_dir=str(tmp_path / "run"),
    )

    state = factory.build(task).run(task, trajectory_id="trajectory-invalid-after-gain", trace=False).state

    assert state.steps[0].step_reward > 0.0
    assert state.steps[1].step_reward == -10.0
    assert state.termination_reason == "invalid_action"
    assert state.final_reward == -10.0
