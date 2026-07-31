import asyncio
import json
from pathlib import Path

import pytest
from verl.tools import restoration_tool as restoration_tool_module
from verl.tools.restoration_tool import RestorationTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


def _build_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="restore_image",
            description="Apply an image restoration action.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "action": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The restoration action to apply.",
                    )
                },
                required=["action"],
            ),
        ),
    )


def _build_tool(**config) -> RestorationTool:
    return RestorationTool(config=config, tool_schema=_build_tool_schema())


def test_model_action_alias_is_translated_before_runtime_validation_on_cpu() -> None:
    tools_config = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
    tool = _build_tool(tool_registry_path=str(tools_config))

    _, alias_reward, alias_metrics = asyncio.run(
        tool.execute("missing-instance", {"action": "B_scunet"})
    )
    _, canonical_reward, canonical_metrics = asyncio.run(
        tool.execute("missing-instance", {"action": "scunet"})
    )

    assert tool.model_to_runtime_actions["B_scunet"] == "scunet"
    assert alias_reward == pytest.approx(-5.0)
    assert alias_metrics["error"] == "instance_not_found"
    assert canonical_reward == pytest.approx(-5.0)
    assert canonical_metrics == {
        "model_action": "scunet",
        "error": "invalid_action",
        "skip_tool_call_reward": True,
    }


def test_repeat_penalty_discourages_low_gain_repeats_on_cpu():
    tool = _build_tool(
        alpha=0.9,
        reward_scale=5.0,
        repeat_action_penalty=0.2,
        repeat_low_gain_penalty=0.8,
        repeat_low_gain_threshold=0.05,
    )

    fresh_reward = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.02, 1.02, 1.02, 1.02, 1.02],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=[],
    )
    repeated_reward = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.02, 1.02, 1.02, 1.02, 1.02],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=["ridcp"],
    )

    assert fresh_reward["repeat_penalty"] == pytest.approx(0.0)
    assert repeated_reward["repeat_penalty"] == pytest.approx(1.0)
    assert repeated_reward["reward"] < fresh_reward["reward"]
    assert repeated_reward["consecutive_action_count"] == pytest.approx(2.0)
    assert repeated_reward["same_type_action_count"] == pytest.approx(2.0)


@pytest.mark.parametrize("reward_mode", ["step_mixed_v1", "final_iqa_v2", "marginal_efficiency_v1"])
def test_repeat_penalty_does_not_discourage_high_gain_same_type_actions_on_cpu(reward_mode):
    tool = _build_tool(
        reward_mode=reward_mode,
        repeat_action_penalty=0.75,
        repeat_low_gain_penalty=2.0,
        repeat_low_gain_threshold=0.03,
    )

    reward = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.04, 1.04, 1.04, 1.04, 1.04],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="kanet",
        actions_history=["ridcp"],
    )

    assert reward["marginal"] > tool.repeat_low_gain_threshold
    assert reward["repeat_penalty"] == pytest.approx(0.0)
    assert reward["same_type_action_count"] == pytest.approx(2.0)


def test_repeat_penalty_groups_restoration_tool_types_on_cpu():
    tool = _build_tool(
        alpha=0.9,
        reward_scale=5.0,
        repeat_action_penalty=0.2,
        repeat_low_gain_penalty=0.0,
    )
    reward_kwargs = {
        "prev_scores": [1.0, 1.0, 1.0, 1.0, 1.0],
        "curr_scores": [1.02, 1.02, 1.02, 1.02, 1.02],
        "identity_scores": [1.0, 1.0, 1.0, 1.0, 1.0],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
    }

    dehaze_reward = tool._calculate_reward(
        **reward_kwargs,
        action="kanet",
        actions_history=["ridcp", "real_esrgan", "focalnet_dehaze"],
    )
    desnow_reward = tool._calculate_reward(
        **reward_kwargs,
        action="turbo_snow",
        actions_history=["s2former"],
    )
    derain_reward = tool._calculate_reward(
        **reward_kwargs,
        action="idt",
        actions_history=["turbo_rain", "s2former"],
    )
    generic_reward = tool._calculate_reward(
        **reward_kwargs,
        action="scunet",
        actions_history=["real_esrgan", "nafnet_denoise"],
    )

    assert dehaze_reward["repeat_tool_type_key"] == "dehaze"
    assert dehaze_reward["repeat_penalty"] == pytest.approx(0.4)
    assert dehaze_reward["same_type_action_count"] == pytest.approx(3.0)
    assert desnow_reward["repeat_tool_type_key"] == "desnow"
    assert desnow_reward["repeat_penalty"] == pytest.approx(0.2)
    assert derain_reward["repeat_penalty"] == pytest.approx(0.0)
    assert generic_reward["repeat_penalty"] == pytest.approx(0.0)


def test_final_iqa_v2_rewards_only_new_best_iqa_on_cpu():
    tool = _build_tool(
        reward_mode="final_iqa_v2",
        final_iqa_reward_scale=4.0,
        final_iqa_regression_penalty_scale=2.0,
        final_iqa_step_penalty=0.05,
        repeat_action_penalty=0.0,
        repeat_low_gain_penalty=0.0,
    )

    first_best = tool._calculate_reward(
        prev_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        curr_scores=[1.10, 1.10, 1.10, 1.10, 1.10],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="scunet",
        actions_history=[],
        best_identity_delta=0.0,
    )
    regression = tool._calculate_reward(
        prev_scores=[1.10, 1.10, 1.10, 1.10, 1.10],
        curr_scores=[1.08, 1.08, 1.08, 1.08, 1.08],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="ridcp",
        actions_history=["scunet"],
        best_identity_delta=first_best["best_identity_delta"],
    )
    new_best = tool._calculate_reward(
        prev_scores=[1.08, 1.08, 1.08, 1.08, 1.08],
        curr_scores=[1.15, 1.15, 1.15, 1.15, 1.15],
        identity_scores=[1.0, 1.0, 1.0, 1.0, 1.0],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        action="real_esrgan",
        actions_history=["scunet", "ridcp"],
        best_identity_delta=first_best["best_identity_delta"],
    )

    assert first_best["best_improvement"] == pytest.approx(0.10)
    assert first_best["reward"] == pytest.approx(0.35)
    assert regression["best_improvement"] == pytest.approx(0.0)
    assert regression["regression_penalty"] == pytest.approx(0.04)
    assert regression["reward"] == pytest.approx(-0.09)
    assert new_best["best_improvement"] == pytest.approx(0.05)
    assert new_best["reward"] == pytest.approx(0.15, abs=1e-6)


def test_marginal_efficiency_reward_telescopes_and_ignores_original_image_term_on_cpu():
    tool = _build_tool(
        reward_mode="marginal_efficiency_v1",
        reward_scale=4.0,
        tool_call_cost=0.12,
        repeat_action_penalty=0.0,
        repeat_low_gain_penalty=0.0,
    )
    reward_kwargs = {
        "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "action": "scunet",
        "actions_history": [],
    }

    first = tool._calculate_reward(
        prev_scores=[1.0] * 5,
        curr_scores=[1.05] * 5,
        identity_scores=[1.0] * 5,
        **reward_kwargs,
    )
    same_marginal_different_identity = tool._calculate_reward(
        prev_scores=[2.0] * 5,
        curr_scores=[2.05] * 5,
        identity_scores=[1.0] * 5,
        **reward_kwargs,
    )
    second = tool._calculate_reward(
        prev_scores=[1.05] * 5,
        curr_scores=[1.03] * 5,
        identity_scores=[1.0] * 5,
        weights=reward_kwargs["weights"],
        action="ridcp",
        actions_history=["scunet"],
    )

    assert first["marginal"] == pytest.approx(0.05)
    assert first["base_reward"] == pytest.approx(0.20)
    assert first["tool_call_cost"] == pytest.approx(0.12)
    assert first["reward"] == pytest.approx(0.08, abs=1e-6)
    assert same_marginal_different_identity["identity"] == pytest.approx(1.05)
    assert same_marginal_different_identity["reward"] == pytest.approx(first["reward"], abs=1e-6)
    assert first["reward"] + second["reward"] == pytest.approx(4.0 * 0.03 - 2 * 0.12, abs=1e-6)


def test_marginal_efficiency_stop_is_always_reward_neutral_on_cpu():
    tool = _build_tool(
        reward_mode="marginal_efficiency_v1",
        stop_min_step=5,
        stop_early_penalty=-1.2,
    )
    tool._instance_dict["test-instance"] = {
        "step": 0,
        "scores_history": [[1.0] * 5],
        "identity_scores": [1.0] * 5,
        "weights": [0.2] * 5,
        "rewards_history": [],
        "best_identity_delta": 0.0,
    }

    response, reward, metrics = asyncio.run(tool.execute("test-instance", {"action": "stop"}))

    assert reward == pytest.approx(0.0)
    assert metrics["reward_mode"] == "marginal_efficiency_v1"
    assert metrics["step"] == 0
    assert "Restoration stopped after 0 step(s)" in response.text
    assert tool._instance_dict["test-instance"]["rewards_history"][-1] == pytest.approx(0.0)
    assert tool._instance_dict["test-instance"]["trajectory_calls"][-1]["reward"] == pytest.approx(0.0)


def test_stop_reward_is_neutral_after_minimum_step_on_cpu():
    tool = _build_tool(
        stop_min_step=3,
        stop_iqa_delta_threshold=0.25,
        stop_early_penalty=-1.0,
        stop_recent_reward_window=2,
        stop_recent_reward_threshold=0.25,
    )

    early_stop = tool._calculate_stop_reward(step=1, identity_delta=0.4, recent_rewards=[0.1])
    plateau_stop = tool._calculate_stop_reward(step=4, identity_delta=0.4, recent_rewards=[0.1, 0.0])
    premature_stop = tool._calculate_stop_reward(step=4, identity_delta=0.1, recent_rewards=[1.0, 0.8])

    assert early_stop["reward"] == pytest.approx(-1.0)
    assert plateau_stop["reward"] == pytest.approx(0.0)
    assert plateau_stop["plateau"] is True
    assert plateau_stop["good_enough"] is True
    assert premature_stop["reward"] == pytest.approx(0.0)


def test_stop_before_five_steps_executes_with_early_penalty_on_cpu():
    tool = _build_tool(
        stop_min_step=5,
        stop_early_penalty=-1.2,
        stop_iqa_delta_threshold=0.25,
    )
    tool._instance_dict["test-instance"] = {
        "step": 4,
        "scores_history": [[1.0] * 5],
        "identity_scores": [1.0] * 5,
        "weights": [0.2] * 5,
        "rewards_history": [0.1] * 4,
        "best_identity_delta": 0.0,
    }

    response, reward, metrics = asyncio.run(tool.execute("test-instance", {"action": "stop"}))

    assert reward == pytest.approx(-1.2)
    assert metrics["action"] == "stop"
    assert metrics["step"] == 4
    assert "error" not in metrics
    assert "Restoration stopped after 4 step(s)" in response.text
    assert tool._instance_dict["test-instance"]["step"] == 4
    assert tool._instance_dict["test-instance"]["rewards_history"][-1] == pytest.approx(-1.2)
    assert tool._instance_dict["test-instance"]["trajectory_calls"][-1]["action"] == "stop"
    assert tool._instance_dict["test-instance"]["trajectory_calls"][-1]["reward"] == pytest.approx(-1.2)


def test_release_logs_one_complete_trajectory_json_record_on_cpu(monkeypatch):
    tool = _build_tool(reward_mode="step_mixed_v1")
    tool._instance_dict["trajectory-1"] = {
        "original_image": "/images/input.png",
        "current_image": "/images/final.png",
        "actions_history": ["ridcp", "kanet"],
        "scores_history": [[0.0] * 5, [0.1] * 5, [0.14] * 5],
        "rewards_history": [0.4, 0.16, 0.0],
        "identity_scores": [0.0] * 5,
        "weights": [0.2] * 5,
        "degradation_type": "fog",
        "step": 2,
        "best_identity_delta": 0.14,
        "trajectory_started_at": 0.0,
        "trajectory_calls": [
            {
                "call_index": 1,
                "action": "ridcp",
                "restoration_step": 1,
                "reward": 0.4,
                "marginal_iqa_gain": 0.1,
                "aggregate_score": 0.1,
                "repeat_penalty": 0.0,
            },
            {
                "call_index": 2,
                "action": "kanet",
                "restoration_step": 2,
                "reward": 0.16,
                "marginal_iqa_gain": 0.04,
                "aggregate_score": 0.14,
                "repeat_penalty": 0.0,
            },
            {
                "call_index": 3,
                "action": "stop",
                "restoration_step": 2,
                "reward": 0.0,
                "identity_delta": 0.14,
            },
        ],
    }
    log_messages = []
    monkeypatch.setattr(restoration_tool_module.trajectory_logger, "info", log_messages.append)

    asyncio.run(tool.release("trajectory-1"))

    assert "trajectory-1" not in tool._instance_dict
    assert len(log_messages) == 1
    summary = json.loads(log_messages[0])
    assert summary["event"] == "restoration_trajectory"
    assert summary["trajectory_id"] == "trajectory-1"
    assert summary["termination_reason"] == "stop"
    assert summary["action_path"] == ["ridcp", "kanet", "stop"]
    assert summary["restoration_step_count"] == 2
    assert summary["tool_call_count"] == 3
    assert summary["initial_aggregate_score"] == pytest.approx(0.0)
    assert summary["final_aggregate_score"] == pytest.approx(0.14)
    assert summary["restoration_reward_sum"] == pytest.approx(0.56)
    assert summary["stop_reward"] == pytest.approx(0.0)
    assert summary["total_tool_reward"] == pytest.approx(0.56)
    assert summary["failed_tool_call_count"] == 0
    assert [call["action"] for call in summary["tool_calls"]] == ["ridcp", "kanet", "stop"]


def test_trajectory_summary_marks_rollout_end_without_stop_on_cpu():
    tool = _build_tool()
    summary = tool._build_trajectory_summary(
        "trajectory-without-stop",
        {
            "original_image": "/images/input.png",
            "current_image": "/images/final.png",
            "actions_history": ["ridcp"],
            "scores_history": [[0.0] * 5, [0.1] * 5],
            "rewards_history": [0.4],
            "identity_scores": [0.0] * 5,
            "weights": [0.2] * 5,
            "step": 1,
            "best_identity_delta": 0.1,
            "trajectory_calls": [
                {
                    "call_index": 1,
                    "action": "ridcp",
                    "status": "success",
                    "reward": 0.4,
                }
            ],
        },
    )

    assert summary["termination_reason"] == "rollout_end_without_stop"
    assert summary["action_path"] == ["ridcp"]
    assert summary["total_tool_reward"] == pytest.approx(0.4)


def test_feedback_calls_out_repeated_low_gain_pattern_on_cpu():
    tool = _build_tool(stop_min_step=3, repeat_low_gain_threshold=0.05)

    feedback = tool._generate_feedback(
        action="ridcp",
        step=4,
        reward=0.12,
        actions_history=["ridcp", "ridcp", "ridcp", "ridcp"],
        marginal=0.01,
        identity_delta=0.35,
        same_type_action_count=4,
        repeat_tool_type_key="dehaze",
    )

    assert "Trajectory uses of dehaze tools: 4" in feedback
    assert "Recent gains are small" in feedback


def test_final_iqa_v2_feedback_focuses_on_trajectory_best_on_cpu():
    tool = _build_tool(reward_mode="final_iqa_v2", stop_min_step=3)

    feedback = tool._generate_feedback(
        action="real_esrgan",
        step=3,
        reward=0.15,
        actions_history=["scunet", "ridcp", "real_esrgan"],
        marginal=0.07,
        identity_delta=0.15,
        same_type_action_count=1,
        best_identity_delta=0.15,
        best_improvement=0.05,
        regression_penalty=0.0,
        step_penalty=0.05,
    )

    assert "Trajectory-best improvement over original image: 0.1500" in feedback
    assert "New best-IQA gain from this action: 0.0500" in feedback
    assert "This action set a new trajectory-best IQA" in feedback
    assert "Recent gains are small" not in feedback


def test_final_iqa_v2_feedback_calls_out_regression_on_cpu():
    tool = _build_tool(reward_mode="final_iqa_v2", stop_min_step=3)

    feedback = tool._generate_feedback(
        action="ridcp",
        step=4,
        reward=-0.09,
        actions_history=["scunet", "ridcp"],
        marginal=-0.02,
        identity_delta=0.08,
        same_type_action_count=1,
        best_identity_delta=0.10,
        best_improvement=0.0,
        regression_penalty=0.04,
        step_penalty=0.05,
    )

    assert "This action fell below the trajectory-best IQA" in feedback
    assert "did not improve the trajectory-best IQA" in feedback
