"""Tests for the stage H GRPO integration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from config import load_stage_g_example_config
from grpo.agent import parse_grpo_task
from grpo.smoke_runtime import override_smoke_decision
from grpo.train_grpo import _configure_swanlab_environment, _load_run_config, _verl_config
from schemas import (
    DegradationType,
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    ValidationStatus,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def test_parse_grpo_task_accepts_agent_lightning_metadata() -> None:
    task = parse_grpo_task(
        {
            "sample_id": "fog-0001",
            "image_path": "/tmp/fog.png",
            "degradation_type": "fog",
            "output_root": "/tmp/grpo",
            "visual_evidence": ["low contrast"],
            "index": 0,
            "data_id": "runtime-id",
        }
    )

    assert task.degradation_type == DegradationType.FOG
    assert task.sample_id == "fog-0001"


def test_parse_grpo_task_rejects_unknown_business_fields() -> None:
    with pytest.raises(ValueError, match="unexpected GRPO task fields"):
        parse_grpo_task(
            {
                "sample_id": "fog-0001",
                "image_path": "/tmp/fog.png",
                "degradation_type": "fog",
                "output_root": "/tmp/grpo",
                "unknown": True,
            }
        )


def test_grpo_config_uses_trajectory_transition_swanlab_and_stochastic_smoke() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    smoke = _verl_config(run, smoke=True)
    formal = _verl_config(run, smoke=False)

    assert smoke["agentlightning"]["trace_aggregator"]["level"] == "trajectory_transition"
    assert smoke["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "seq-mean-token-mean"
    assert smoke["actor_rollout_ref"]["rollout"]["temperature"] > 0
    assert smoke["actor_rollout_ref"]["rollout"]["n"] == 2
    assert smoke["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 2
    assert formal["actor_rollout_ref"]["rollout"]["n"] == 4
    assert formal["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 4
    assert formal["trainer"]["n_gpus_per_node"] == 4
    assert run["training"]["n_runners"] == 256
    assert run["training"]["n_runners"] == run["training"]["train_batch_size"] * run["training"]["rollout_n"]
    assert formal["data"]["max_response_length"] == 640
    assert formal["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"] is False
    assert formal["actor_rollout_ref"]["model"]["enable_activation_offload"] is False
    assert formal["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 8
    assert formal["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert formal["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1
    assert formal["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 16384
    assert formal["actor_rollout_ref"]["actor"]["clip_ratio_high"] == pytest.approx(0.3)
    assert formal["actor_rollout_ref"]["actor"]["grad_clip"] == pytest.approx(1.0)
    assert formal["actor_rollout_ref"]["actor"]["use_kl_loss"] is True
    assert formal["algorithm"]["use_kl_in_reward"] is False
    assert formal["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert formal["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert "swanlab" in formal["trainer"]["logger"]
    assert formal["trainer"]["project_name"] == "image-restoration-multi-agent"
    assert formal["trainer"]["experiment_name"] == "fog-expert-grpo"
    assert smoke["trainer"]["experiment_name"] == "fog-expert-grpo-smoke"
    assert formal["trainer"]["max_actor_ckpt_to_keep"] == 2


def test_stage_h_uses_persistent_split_gpu_tool_runtime() -> None:
    config = load_stage_g_example_config(EXAMPLE_DIR / "config/stage_h.yaml")

    assert config.workflow.max_steps == 6
    assert config.runtime.evaluator.device == "cuda:0"
    assert config.runtime.restoration.device == "cuda:1"
    assert config.runtime.evaluator.service_url == "http://127.0.0.1:8767"
    assert config.runtime.restoration.service_url == "http://127.0.0.1:8767"

    launch_script = (EXAMPLE_DIR / "grpo/run_expert_grpo.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES:-0,1,2,3" in launch_script
    assert "IMAGE_RESTORATION_WORKERS_PER_DEVICE:-1" in launch_script
    assert "IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE:-2" in launch_script
    assert "IMAGE_RESTORATION_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3" in launch_script
    assert "IMAGE_RESTORATION_IQA_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3" in launch_script
    assert "RESTORATION_WORKERS=$((RESTORATION_WORKERS_PER_DEVICE * RESTORATION_DEVICE_COUNT))" in launch_script
    assert "IQA_WORKERS=$((IQA_WORKERS_PER_DEVICE * IQA_DEVICE_COUNT))" in launch_script
    assert '--restoration-workers "$RESTORATION_WORKERS"' in launch_script
    assert '--iqa-workers "$IQA_WORKERS"' in launch_script
    assert '--restoration-devices "$RESTORATION_DEVICES"' in launch_script
    assert '--iqa-devices "$IQA_DEVICES"' in launch_script


def test_all_expert_grpo_configs_expose_the_same_training_parameters() -> None:
    config_paths = sorted((EXAMPLE_DIR / "grpo/configs").glob("*.yaml"))
    runs = [_load_run_config(path) for path in config_paths]
    expected_keys = set(runs[0]["training"])

    assert len(config_paths) == 4
    assert {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_epochs",
        "ppo_max_token_len_per_gpu",
        "clip_ratio",
        "clip_ratio_low",
        "clip_ratio_high",
        "grad_clip",
        "rollout_log_prob_micro_batch_size_per_gpu",
        "ref_log_prob_micro_batch_size_per_gpu",
        "use_kl_loss",
        "use_kl_in_reward",
        "fsdp_param_offload",
        "resume_mode",
    } <= expected_keys
    assert all(set(run["training"]) == expected_keys for run in runs)
    assert all(run["training"]["enable_gradient_checkpointing"] is False for run in runs)
    assert all(run["training"]["enable_activation_offload"] is False for run in runs)
    assert all(run["training"]["n_gpus_per_node"] == 4 for run in runs)
    assert all(run["training"]["tensor_model_parallel_size"] == 4 for run in runs)
    assert all(
        run["training"]["n_runners"] == run["training"]["train_batch_size"] * run["training"]["rollout_n"]
        for run in runs
    )
    assert all(
        set(run["swanlab"])
        == {
            "enabled",
            "project_name",
            "experiment_name",
            "mode",
            "smoke_mode",
            "log_dir",
        }
        for run in runs
    )
    assert all(run["swanlab"]["project_name"] == "image-restoration-multi-agent" for run in runs)
    assert all(run["swanlab"]["mode"] == "online" for run in runs)
    assert all(run["swanlab"]["smoke_mode"] == "offline" for run in runs)
    assert len({run["swanlab"]["experiment_name"] for run in runs}) == 4
    assert all(Path(run["swanlab"]["log_dir"]).is_absolute() for run in runs)


def test_swanlab_environment_and_disable_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    monkeypatch.delenv("SWANLAB_LOG_DIR", raising=False)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)

    _configure_swanlab_environment(run, smoke=False)
    assert Path(os.environ["SWANLAB_LOG_DIR"]) == Path(run["swanlab"]["log_dir"])
    assert os.environ["SWANLAB_MODE"] == "online"

    _configure_swanlab_environment(run, smoke=True)
    assert os.environ["SWANLAB_MODE"] == "offline"

    run["swanlab"]["enabled"] = False
    assert _verl_config(run, smoke=False)["trainer"]["logger"] == ["console"]


def test_smoke_override_retains_real_response_and_forces_valid_action() -> None:
    observed = ExpertDecisionRecord(
        expert_name=ExpertName.FOG,
        step_index=0,
        action=None,
        decision_source=ExpertDecisionSource.VLM,
        parse_status=ExpertParseStatus.INVALID_TOOL_CALL,
        validation_status=ValidationStatus.INVALID_TOOL_CALL,
        raw_assistant_output="analysis without a Hermes call",
        response_payload={"id": "response-1"},
        error="missing tool call",
    )

    overridden = override_smoke_decision(observed, "scunet")

    assert overridden.action == "scunet"
    assert overridden.parse_status == ExpertParseStatus.VALID
    assert overridden.decision_source == ExpertDecisionSource.SMOKE_OVERRIDE
    assert overridden.raw_assistant_output == observed.raw_assistant_output
    assert overridden.response_payload is not None
    assert overridden.response_payload["smoke_override"]["original_parse_status"] == "invalid_tool_call"
