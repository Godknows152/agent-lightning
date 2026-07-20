"""Tests for the stage H GRPO integration."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from config import load_stage_g_example_config
from grpo.agent import ArtifactRetentionConfig, cleanup_rollout_artifacts, parse_grpo_task
from grpo.smoke_runtime import override_smoke_decision
from grpo.train_grpo import _apply_expert_vlm_overrides, _configure_swanlab_environment, _load_run_config, _verl_config
from schemas import (
    DegradationType,
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    ValidationStatus,
)

from agentlightning.verl.daemon import _extract_image_restoration_metadata
from agentlightning.verl.trainer import (
    compute_effective_ppo_update_batch_size,
    compute_pre_advantage_padding_divisor,
    compute_trajectory_transition_grpo_advantage,
    compute_transition_grpo_advantage,
    normalize_local_rollout_server_addresses,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXAMPLE_DIR.parents[1]


def test_trajectory_fallback_keeps_single_visual_slot() -> None:
    daemon_source = (REPO_DIR / "agentlightning/verl/daemon.py").read_text(encoding="utf-8")

    assert "def strip_vision_token_spans" in daemon_source
    assert "strip_vision_token_spans(" in daemon_source
    assert "trace_prompt_ids" in daemon_source
    assert (
        'trajectory_image_source = str(self.trace_aggregator.get("trajectory_image_source", "first"))' in daemon_source
    )
    assert "current_merged_trace_idx[-1]" in daemon_source
    assert 'if trajectory_image_source == "latest"' in daemon_source
    assert (
        "image_placeholder_count = count_vision_placeholders(prompt_ids + response_ids, self.tokenizer)"
        in daemon_source
    )


def test_image_restoration_trace_metadata_extraction() -> None:
    class SpanWithAttributes:
        attributes = {
            "image_restoration.termination_reason": "expert_stop",
            "image_restoration.turn_count": 4,
            "image_restoration.final_reward": 1.25,
        }

    metadata = _extract_image_restoration_metadata([SpanWithAttributes()])

    assert metadata["image_restoration.termination_reason"] == "expert_stop"
    assert metadata["image_restoration.turn_count"] == 4
    assert metadata["image_restoration.final_reward"] == pytest.approx(1.25)


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


def test_grpo_config_uses_verl_style_trajectory_swanlab_and_stochastic_smoke() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    smoke = _verl_config(run, smoke=True)
    formal = _verl_config(run, smoke=False)

    assert smoke["agentlightning"]["trace_aggregator"]["level"] == "trajectory"
    assert smoke["agentlightning"]["trace_aggregator"]["force_one_sample_per_rollout"] is True
    assert smoke["agentlightning"]["trace_aggregator"]["trajectory_image_source"] == "latest"
    assert smoke["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "seq-mean-token-mean"
    assert smoke["actor_rollout_ref"]["rollout"]["temperature"] > 0
    assert smoke["actor_rollout_ref"]["rollout"]["n"] == 2
    assert smoke["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 1
    assert formal["actor_rollout_ref"]["rollout"]["n"] == 4
    assert formal["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 4
    assert formal["trainer"]["n_gpus_per_node"] == 4
    assert run["training"]["train_batch_size"] == 32
    assert run["training"]["n_runners"] == 128
    assert run["training"]["n_runners"] == run["training"]["train_batch_size"] * run["training"]["rollout_n"]
    assert run["training"]["temperature"] == pytest.approx(1.0)
    assert run["expert_vlm"] == {"max_tokens": 128, "temperature": 1.0, "top_p": 0.9}
    assert run["training"]["artifact_cleanup_enabled"] is True
    assert run["training"]["artifact_keep_sample_dirs"] == 256
    assert run["training"]["artifact_min_age_seconds"] == pytest.approx(7200.0)
    assert run["training"]["artifact_cleanup_interval_rollouts"] == 48
    assert formal["data"]["max_response_length"] == 640
    assert formal["actor_rollout_ref"]["rollout"]["temperature"] == pytest.approx(1.0)
    assert formal["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"] == 4096
    assert formal["agentlightning"]["trace_aggregator"]["trajectory_max_response_length"] == 6144
    assert formal["agentlightning"]["trace_aggregator"]["trajectory_image_source"] == "latest"
    assert formal["agentlightning"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert formal["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"] is True
    assert formal["actor_rollout_ref"]["model"]["enable_activation_offload"] is False
    assert formal["agentlightning"]["rollout_resource_control"] == {
        "enabled": True,
        "base_url": "http://127.0.0.1:8767",
        "timeout_seconds": 1800.0,
    }
    assert formal["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 8
    assert formal["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert formal["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1
    assert formal["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 4096
    assert formal["actor_rollout_ref"]["actor"]["clip_ratio_high"] == pytest.approx(0.3)
    assert formal["actor_rollout_ref"]["actor"]["grad_clip"] == pytest.approx(1.0)
    assert formal["actor_rollout_ref"]["actor"]["use_kl_loss"] is True
    assert formal["algorithm"]["use_kl_in_reward"] is False
    assert formal["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert formal["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert "swanlab" in formal["trainer"]["logger"]
    assert formal["trainer"]["project_name"] == "image-restoration-multi-agent"
    assert formal["trainer"]["experiment_name"] == "qwen3.5-fog-expert-grpo"
    assert smoke["trainer"]["experiment_name"] == "qwen3.5-fog-expert-grpo-smoke"
    assert formal["trainer"]["max_actor_ckpt_to_keep"] == 2


def test_rollout_artifact_cleanup_keeps_recent_and_current_sample_dirs(tmp_path: Path) -> None:
    root = tmp_path / "rollouts" / "fog"
    root.mkdir(parents=True)
    now = time.time()
    old_dirs = [root / f"fog-{index:06d}" for index in range(4)]
    recent_dir = root / "fog-999998"
    current_dir = root / "fog-999999"
    for index, sample_dir in enumerate([*old_dirs, recent_dir, current_dir]):
        (sample_dir / "rollout" / "attempt").mkdir(parents=True)
        (sample_dir / "rollout" / "attempt" / "trajectory.json").write_text("{}", encoding="utf-8")
        mtime = now - 1000 + index
        os.utime(sample_dir, (mtime, mtime))
    os.utime(recent_dir, (now, now))
    os.utime(current_dir, (now - 1000, now - 1000))

    removed = cleanup_rollout_artifacts(
        root,
        current_sample_dir=current_dir,
        retention=ArtifactRetentionConfig(
            enabled=True,
            keep_sample_dirs=2,
            min_age_seconds=10,
            cleanup_interval_rollouts=1,
        ),
    )

    assert removed == 3
    assert old_dirs[0].exists() is False
    assert old_dirs[1].exists() is False
    assert old_dirs[2].exists() is False
    assert old_dirs[3].exists()
    assert recent_dir.exists()
    assert current_dir.exists()


def test_grpo_chat_template_uses_qwen35_native_tool_nothink() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    formal = _verl_config(run, smoke=False)
    template = formal["actor_rollout_ref"]["model"]["custom_chat_template"]

    expected_generation_prompt = "add_generation_prompt"
    assert expected_generation_prompt in template
    assert "<think>\\n\\n</think>\\n\\n" in template
    assert "enable_thinking is defined and enable_thinking is false" in template
    assert "<|vision_start|><|image_pad|><|vision_end|>" in template
    assert "<function=example_function_name>" in template
    assert "tool_call.arguments|items" in template
    assert formal["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["chat_template"] == str(
        EXAMPLE_DIR / "grpo/templates/qwen35_native_tool_nothink.jinja"
    )
    assert formal["actor_rollout_ref"]["rollout"]["multi_turn"] == {"format": "qwen3_coder"}
    assert formal["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] == "qwen3_coder"
    assert "chat_template_kwargs" not in formal["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]


def test_diagnosis_and_expert_serving_use_separate_tool_protocols() -> None:
    diagnosis_script = (EXAMPLE_DIR / "serve_qwen35.sh").read_text(encoding="utf-8")
    expert_script = (EXAMPLE_DIR / "serve_qwen35_expert.sh").read_text(encoding="utf-8")

    assert 'PORT="${QWEN35_PORT:-8000}"' in diagnosis_script
    assert "--tool-call-parser hermes" in diagnosis_script
    assert "qwen35_hermes_nothink.jinja" in diagnosis_script
    assert 'PORT="${QWEN35_EXPERT_PORT:-8001}"' in expert_script
    assert "--tool-call-parser qwen3_coder" in expert_script
    assert "qwen35_native_tool_nothink.jinja" in expert_script


def test_stage_h_uses_persistent_split_gpu_tool_runtime() -> None:
    config = load_stage_g_example_config(EXAMPLE_DIR / "config/stage_h.yaml")

    assert config.workflow.max_steps == 6
    assert config.workflow.invalid_action_penalty == 10.0
    assert config.workflow.tool_call_reward == pytest.approx(0.2)
    assert config.workflow.premature_stop_penalty == pytest.approx(0.0)
    assert config.vlm.base_url == "http://127.0.0.1:8000/v1"
    assert config.expert_vlm.base_url == "http://127.0.0.1:8001/v1"
    assert config.expert_vlm.max_tokens == 512
    assert config.expert_vlm.temperature == pytest.approx(1.0)
    assert config.runtime.evaluator.device == "cuda:0"
    assert config.runtime.restoration.device == "cuda:1"
    assert config.runtime.evaluator.service_url == "http://127.0.0.1:8767"
    assert config.runtime.restoration.service_url == "http://127.0.0.1:8767"

    common_script = (EXAMPLE_DIR / "grpo/run_expert_grpo.sh").read_text(encoding="utf-8")
    four_gpu_script = (EXAMPLE_DIR / "grpo/run_expert_grpo_4gpu.sh").read_text(encoding="utf-8")
    two_gpu_script = (EXAMPLE_DIR / "grpo/run_expert_grpo_2gpu.sh").read_text(encoding="utf-8")
    assert "RESTORATION_WORKERS=$((RESTORATION_WORKERS_PER_DEVICE * RESTORATION_DEVICE_COUNT))" in common_script
    assert "IQA_WORKERS=$((IQA_WORKERS_PER_DEVICE * IQA_DEVICE_COUNT))" in common_script
    assert '--restoration-workers "$RESTORATION_WORKERS"' in common_script
    assert '--iqa-workers "$IQA_WORKERS"' in common_script
    assert "CUDA_VISIBLE_DEVICES:-0,1,2,3" in four_gpu_script
    assert 'GRPO_CONFIG_DIR="examples/image_restoration_multi_agent/grpo/configs"' in four_gpu_script
    assert 'IMAGE_RESTORATION_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"' in four_gpu_script
    assert 'IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE="2"' in four_gpu_script
    assert "for expert in fog snow rain low_light; do" in four_gpu_script
    assert "remaining experts will not be started" in four_gpu_script
    assert "CUDA_VISIBLE_DEVICES:-0,1" in two_gpu_script
    assert 'GRPO_CONFIG_DIR="examples/image_restoration_multi_agent/grpo/configs_2gpu"' in two_gpu_script
    assert 'IMAGE_RESTORATION_DEVICES="cuda:0,cuda:1"' in two_gpu_script
    assert 'IMAGE_RESTORATION_IQA_WORKERS_PER_DEVICE="2"' in two_gpu_script
    assert "for expert in fog snow rain low_light; do" in two_gpu_script
    assert "remaining experts will not be started" in two_gpu_script


def test_grpo_run_config_overrides_expert_vlm_sampling() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    config = _apply_expert_vlm_overrides(load_stage_g_example_config(run["stage_config"]), run)

    assert config.expert_vlm.max_tokens == 128
    assert config.expert_vlm.temperature == pytest.approx(1.0)
    assert config.expert_vlm.top_p == pytest.approx(0.9)


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
        "trace_aggregator_level",
        "force_one_sample_per_rollout",
        "trajectory_image_source",
        "artifact_cleanup_enabled",
        "artifact_keep_sample_dirs",
        "artifact_min_age_seconds",
        "artifact_cleanup_interval_rollouts",
    } <= expected_keys
    assert all(set(run["training"]) == expected_keys for run in runs)
    assert all(run["training"]["enable_gradient_checkpointing"] is True for run in runs)
    assert all(run["training"]["enable_activation_offload"] is False for run in runs)
    assert all(run["training"]["enable_thinking"] is False for run in runs)
    assert all(run["training"]["temperature"] == pytest.approx(1.0) for run in runs)
    assert all(run["training"]["artifact_cleanup_enabled"] is True for run in runs)
    assert all(run["training"]["artifact_keep_sample_dirs"] == 256 for run in runs)
    assert all(run["training"]["artifact_min_age_seconds"] == pytest.approx(7200.0) for run in runs)
    assert all(run["training"]["artifact_cleanup_interval_rollouts"] == 48 for run in runs)
    assert all(run["expert_vlm"]["max_tokens"] == 128 for run in runs)
    assert all(run["expert_vlm"]["temperature"] == pytest.approx(run["training"]["temperature"]) for run in runs)
    assert all(run["expert_vlm"]["top_p"] == pytest.approx(run["training"]["top_p"]) for run in runs)
    assert all(run["training"]["train_batch_size"] == 32 for run in runs)
    assert all(run["training"]["rollout_n"] == 4 for run in runs)
    assert all(run["training"]["n_runners"] == 128 for run in runs)
    assert all(run["training"]["n_gpus_per_node"] == 4 for run in runs)
    assert all(run["training"]["tensor_model_parallel_size"] == 4 for run in runs)
    assert all(run["training"]["trace_aggregator_level"] == "trajectory" for run in runs)
    assert all(run["training"]["force_one_sample_per_rollout"] is True for run in runs)
    assert all(run["training"]["trajectory_image_source"] == "latest" for run in runs)
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


def test_all_two_gpu_expert_configs_use_two_gpu_topology() -> None:
    config_paths = sorted((EXAMPLE_DIR / "grpo/configs_2gpu").glob("*.yaml"))
    runs = [_load_run_config(path) for path in config_paths]

    assert len(config_paths) == 4
    assert all(run["training"]["train_batch_size"] == 8 for run in runs)
    assert all(run["training"]["enable_gradient_checkpointing"] is True for run in runs)
    assert all(run["training"]["enable_thinking"] is False for run in runs)
    assert all(run["training"]["temperature"] == pytest.approx(0.5) for run in runs)
    assert all(run["training"]["artifact_cleanup_enabled"] is True for run in runs)
    assert all(run["training"]["artifact_keep_sample_dirs"] == 256 for run in runs)
    assert all(run["training"]["artifact_min_age_seconds"] == pytest.approx(7200.0) for run in runs)
    assert all(run["training"]["artifact_cleanup_interval_rollouts"] == 48 for run in runs)
    assert all(run["expert_vlm"]["max_tokens"] == 128 for run in runs)
    assert all(run["expert_vlm"]["temperature"] == pytest.approx(run["training"]["temperature"]) for run in runs)
    assert all(run["expert_vlm"]["top_p"] == pytest.approx(run["training"]["top_p"]) for run in runs)
    assert all(run["training"]["rollout_n"] == 3 for run in runs)
    assert all(run["training"]["n_runners"] == 24 for run in runs)
    assert all(run["training"]["n_gpus_per_node"] == 2 for run in runs)
    assert all(run["training"]["tensor_model_parallel_size"] == 2 for run in runs)
    assert all(run["training"]["trace_aggregator_level"] == "trajectory" for run in runs)
    assert all(run["training"]["force_one_sample_per_rollout"] is True for run in runs)
    assert all(run["training"]["trajectory_image_source"] == "latest" for run in runs)
    assert all("_2gpu" in run["output_dir"] for run in runs)
    assert all(str(run["swanlab"]["experiment_name"]).endswith("-2gpu") for run in runs)
    assert all(
        run["training"]["n_runners"] == run["training"]["train_batch_size"] * run["training"]["rollout_n"]
        for run in runs
    )


def test_transition_advantage_uses_each_turn_reward_group() -> None:
    token_level_rewards = torch.zeros((4, 3), dtype=torch.float32)
    token_level_rewards[:, -1] = torch.tensor([1.0, 3.0, 10.0, 14.0])
    response_mask = torch.ones_like(token_level_rewards)
    advantages, returns = compute_transition_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        data_ids=np.array(["sample-a", "sample-a", "sample-a", "sample-a"], dtype=object),
        turn_indices=np.array([0, 0, 1, 1]),
        norm_by_std=False,
    )

    row_advantages = advantages[:, 0]
    assert row_advantages.tolist() == pytest.approx([-1.0, 1.0, -2.0, 2.0])
    assert torch.equal(returns, advantages)


def test_trajectory_transition_advantage_uses_rollout_reward_and_turn_weighting() -> None:
    token_level_rewards = torch.zeros((5, 3), dtype=torch.float32)
    token_level_rewards[:, -1] = torch.tensor([1.0, 1.0, 4.0, 4.0, 4.0])
    response_mask = torch.ones_like(token_level_rewards)
    advantages, returns = compute_trajectory_transition_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        data_ids=np.array(["sample-a"] * 5, dtype=object),
        rollout_ids=np.array(["rollout-low", "rollout-low", "rollout-high", "rollout-high", "rollout-high"]),
        norm_by_std=False,
    )

    row_advantages = advantages[:, 0]
    assert row_advantages.tolist() == pytest.approx([-0.75, -0.75, 0.5, 0.5, 0.5])
    assert torch.equal(returns, advantages)


def test_effective_ppo_update_batch_size_matches_rollout_expansion() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs_2gpu/fog.yaml")
    formal = _verl_config(run, smoke=False)

    assert (
        compute_effective_ppo_update_batch_size(
            ppo_mini_batch_size=formal["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"],
            rollout_n=formal["actor_rollout_ref"]["rollout"]["n"],
        )
        == 24
    )


def test_pre_advantage_padding_divisor_covers_log_prob_micro_batches() -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs_2gpu/fog.yaml")
    formal = _verl_config(run, smoke=False)

    assert (
        compute_pre_advantage_padding_divisor(
            actor_world_size=formal["trainer"]["n_gpus_per_node"],
            rollout_log_prob_micro_batch_size_per_gpu=formal["actor_rollout_ref"]["rollout"][
                "log_prob_micro_batch_size_per_gpu"
            ],
            ref_world_size=formal["trainer"]["n_gpus_per_node"],
            ref_log_prob_micro_batch_size_per_gpu=formal["actor_rollout_ref"]["ref"][
                "log_prob_micro_batch_size_per_gpu"
            ],
        )
        == 4
    )
    assert (
        compute_pre_advantage_padding_divisor(
            actor_world_size=4,
            rollout_log_prob_micro_batch_size_per_gpu=2,
            ref_world_size=4,
            ref_log_prob_micro_batch_size_per_gpu=2,
        )
        == 8
    )


def test_controller_emits_step_rewards_for_transition_training() -> None:
    controller_source = (EXAMPLE_DIR / "controller.py").read_text(encoding="utf-8")

    assert "def _emit_step_reward" in controller_source
    assert '"restoration.reward_scope": "transition"' in controller_source
    assert "agl.emit_reward(" in controller_source


def test_swanlab_environment_and_disable_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    monkeypatch.delenv("SWANLAB_LOG_DIR", raising=False)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    monkeypatch.delenv("GRPO_OFFLINE", raising=False)
    monkeypatch.delenv("GRPO_SWANLAB_MODE", raising=False)

    _configure_swanlab_environment(run, smoke=False)
    assert Path(os.environ["SWANLAB_LOG_DIR"]) == Path(run["swanlab"]["log_dir"])
    assert os.environ["SWANLAB_MODE"] == "cloud"

    monkeypatch.setenv("GRPO_SWANLAB_MODE", "offline")
    _configure_swanlab_environment(run, smoke=False)
    assert os.environ["SWANLAB_MODE"] == "offline"
    monkeypatch.delenv("GRPO_SWANLAB_MODE", raising=False)

    monkeypatch.setenv("GRPO_OFFLINE", "1")
    _configure_swanlab_environment(run, smoke=False)
    assert os.environ["SWANLAB_MODE"] == "cloud"

    _configure_swanlab_environment(run, smoke=True)
    assert os.environ["SWANLAB_MODE"] == "offline"

    run["swanlab"]["enabled"] = False
    assert _verl_config(run, smoke=False)["trainer"]["logger"] == ["console"]


def test_local_rollout_server_addresses_can_be_forced_to_loopback() -> None:
    assert normalize_local_rollout_server_addresses(["10.246.1.30:41423"], force=True) == ["10.246.1.30:41423"]
    assert normalize_local_rollout_server_addresses(["10.246.1.30:41423"], force=False) == ["10.246.1.30:41423"]
    assert normalize_local_rollout_server_addresses(["0.0.0.0:41423"], force=True) == ["127.0.0.1:41423"]
    assert normalize_local_rollout_server_addresses(["localhost:41423"], force=True) == ["127.0.0.1:41423"]
    assert normalize_local_rollout_server_addresses(["[::1]:41423"], force=True) == ["127.0.0.1:41423"]


def test_grpo_tool_lifecycle_uses_runtime_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _load_run_config(EXAMPLE_DIR / "grpo/configs/fog.yaml")
    monkeypatch.setenv("IMAGE_RESTORATION_SERVICE_URL", "http://127.0.0.1:9876")

    formal = _verl_config(run, smoke=False)

    assert formal["agentlightning"]["rollout_resource_control"]["base_url"] == "http://127.0.0.1:9876"


def test_agent_lightning_verl_imports_with_transformers5_qwen35_compat() -> None:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq

    import agentlightning as agl
    import agentlightning.verl  # noqa: F401

    run = _load_run_config(EXAMPLE_DIR / "grpo/configs_2gpu/fog.yaml")
    algorithm = agl.VERL(_verl_config(run, smoke=True))

    assert type(algorithm).__name__ == "VERL"
    assert AutoModelForVision2Seq is AutoModelForImageTextToText


def test_smoke_override_retains_real_response_and_forces_valid_action() -> None:
    observed = ExpertDecisionRecord(
        expert_name=ExpertName.FOG,
        step_index=0,
        action=None,
        decision_source=ExpertDecisionSource.VLM,
        parse_status=ExpertParseStatus.INVALID_TOOL_CALL,
        validation_status=ValidationStatus.INVALID_TOOL_CALL,
        raw_assistant_output="analysis without a Qwen3 tool call",
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
