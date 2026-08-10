from __future__ import annotations

from pathlib import Path

import yaml


OLD_VERL_ROOT = Path(__file__).resolve().parents[1] / "old_verl_grpo"
EXPERTS = ("fog", "rain", "snow", "lowlight")
VERSIONS = ("v1", "v2", "v3", "v4", "v4.1.1")
PROJECT_NAMES = {
    "fog": "FogRL",
    "rain": "RainRL",
    "snow": "SnowRL",
    "lowlight": "LowLightRL",
}
DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_four_gpu_common_config_uses_all_cards_with_two_tp2_rollout_replicas() -> None:
    config = _read_yaml(
        OLD_VERL_ROOT / "config" / "restoration_common_config_4gpu.yaml"
    )

    assert config["trainer"]["n_gpus_per_node"] == 4
    tensor_parallel_size = config["actor_rollout_ref"]["rollout"][
        "tensor_model_parallel_size"
    ]
    assert tensor_parallel_size == 2
    assert config["trainer"]["n_gpus_per_node"] // tensor_parallel_size == 2
    assert config["actor_rollout_ref"]["rollout"]["multi_turn"][
        "tool_config_path"
    ].endswith("_4gpu.yaml")


def test_four_gpu_tool_configs_create_one_restoration_and_iqa_worker_per_card() -> None:
    tool_config_root = OLD_VERL_ROOT / "config" / "tool_config"
    tool_configs = {
        tool_config_root / "restoration_tool_config_current_iqa_4gpu.yaml": True,
        tool_config_root / "restoration_tool_config_marginal_efficiency_4gpu.yaml": True,
        tool_config_root / "v1" / "restoration_tool_config_4gpu.yaml": True,
        tool_config_root / "v2" / "restoration_tool_config_4gpu.yaml": False,
        tool_config_root / "v3" / "restoration_tool_config_4gpu.yaml": True,
    }

    for path, keep_models_loaded in tool_configs.items():
        runtime = _read_yaml(path)["tools"][0]["config"]
        assert runtime["worker_devices"] == DEVICES, path
        assert runtime["model_devices"] == DEVICES, path
        assert runtime["iqa_devices"] == DEVICES, path
        assert runtime["output_dir"].endswith(
            "agent_lightning_old_verl_restoration_4gpu"
        ), path
        assert runtime["tool_result_cache_dir"] == (
            "/home/LXJ/tmp/agent_lightning_old_verl_cache/tool_results"
        ), path
        assert runtime["identity_iqa_cache_dir"] == (
            "/home/LXJ/tmp/agent_lightning_old_verl_cache/identity_iqa"
        ), path
        assert runtime["keep_models_loaded_between_sampling_steps"] is keep_models_loaded, path


def test_all_tool_configs_use_persistent_shared_caches() -> None:
    tool_config_root = OLD_VERL_ROOT / "config" / "tool_config"
    for path in tool_config_root.rglob("*.yaml"):
        runtime = _read_yaml(path)["tools"][0]["config"]
        assert runtime["enable_tool_result_cache"] is True, path
        assert runtime["tool_result_cache_dir"] == (
            "/home/LXJ/tmp/agent_lightning_old_verl_cache/tool_results"
        ), path
        assert runtime["tool_result_cache_ttl_hours"] == 0, path
        assert runtime["enable_identity_iqa_cache"] is True, path
        assert runtime["identity_iqa_cache_dir"] == (
            "/home/LXJ/tmp/agent_lightning_old_verl_cache/identity_iqa"
        ), path
        assert runtime["identity_iqa_cache_ttl_hours"] == 0, path


def test_strict_phase_scheduling_is_enabled_only_for_two_gpu_training() -> None:
    two_gpu = _read_yaml(OLD_VERL_ROOT / "config" / "restoration_common_config_2gpu.yaml")
    three_gpu = _read_yaml(OLD_VERL_ROOT / "config" / "restoration_common_config_3gpu.yaml")
    four_gpu = _read_yaml(OLD_VERL_ROOT / "config" / "restoration_common_config_4gpu.yaml")

    assert two_gpu["actor_rollout_ref"]["rollout"]["multi_turn"]["phase_separated_tool_execution"] is True
    assert three_gpu["actor_rollout_ref"]["rollout"]["multi_turn"]["phase_separated_tool_execution"] is False
    assert four_gpu["actor_rollout_ref"]["rollout"]["multi_turn"]["phase_separated_tool_execution"] is False


def test_all_expert_versions_have_isolated_four_gpu_configs_and_launchers() -> None:
    for expert in EXPERTS:
        runtime_expert = "low_light" if expert == "lowlight" else expert
        for version in VERSIONS:
            config_path = (
                OLD_VERL_ROOT
                / "config"
                / expert
                / version
                / f"{expert}_config_4gpu.yaml"
            )
            config = _read_yaml(config_path)
            output_version = "v4.1.2" if version == "v4.1.1" else version
            defaults = config["defaults"]
            assert "restoration_common_config_4gpu" in defaults, config_path
            data_root = "data"
            if not (expert == "fog" and version == "v4.1.1"):
                data_root = "data/4gpu"
            assert config["data"]["train_files"] == [
                f"examples/image_restoration_multi_agent/old_verl_grpo/{data_root}/{runtime_expert}_train.parquet"
            ], config_path
            assert config["data"]["val_files"] == [
                f"examples/image_restoration_multi_agent/old_verl_grpo/{data_root}/{runtime_expert}_val.parquet"
            ], config_path
            tool_config = config["actor_rollout_ref"]["rollout"]["multi_turn"][
                "tool_config_path"
            ]
            assert tool_config.endswith("_4gpu.yaml"), config_path
            trainer = config["trainer"]
            assert trainer["project_name"] == PROJECT_NAMES[expert], config_path
            assert trainer["experiment_name"] == f"{expert}_{output_version}", config_path
            assert trainer["default_local_dir"].endswith(
                f"outputs/{expert}/{output_version}/4gpu"
            ), config_path
            env_vars = trainer["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
            assert env_vars["SWANLAB_LOG_DIR"].endswith(
                f"outputs/{expert}/{output_version}/4gpu/swanlab"
            ), config_path
            assert env_vars["VERL_LOG_DIR"].endswith(f"log/{expert}/{output_version}/4gpu"), (
                config_path
            )

            script_version = "v4_1_1" if version == "v4.1.1" else version
            launcher = OLD_VERL_ROOT / "scripts" / expert / f"{expert}_{script_version}_4gpu.sh"
            content = launcher.read_text(encoding="utf-8")
            assert "run_expert_old_verl_grpo_4gpu.sh" in content, launcher
            assert "${EXPERT}_config_4gpu" in content, launcher
            assert "/${EXPERT}/${VERSION}/4gpu}" in content, launcher
            assert (
                'OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VERSION}}"'
                in content
            ), launcher


def test_runtime_pool_binds_each_replica_to_one_restoration_and_iqa_device() -> None:
    runtime_path = (
        OLD_VERL_ROOT.parent / "verl_backend" / "verl" / "tools" / "restoration_tool.py"
    )
    content = runtime_path.read_text(encoding="utf-8")

    assert "for index, device in enumerate(worker_devices):" in content
    assert "model_devices=[device]" in content
    assert (
        "iqa_device = resolved_iqa_devices[index % len(resolved_iqa_devices)]"
        in content
    )
    assert "get_iqa_scorer(\n                    device=iqa_device," in content


def test_four_gpu_helper_launchers_do_not_fall_back_to_two_gpu_files() -> None:
    helpers = (
        OLD_VERL_ROOT / "run_expert_old_verl_grpo_4gpu.sh",
        OLD_VERL_ROOT / "run_four_experts_serial_old_verl_grpo_4gpu.sh",
        OLD_VERL_ROOT / "run_action_rarity_v3_old_verl_grpo_4gpu.sh",
    )

    for helper in helpers:
        content = helper.read_text(encoding="utf-8")
        assert "run_expert_old_verl_grpo_2gpu.sh" not in content, helper
        assert "_config_2gpu" not in content, helper
        assert "OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1}" not in content, helper

    common_wrapper = helpers[0].read_text(encoding="utf-8")
    assert (
        'export CUDA_VISIBLE_DEVICES="${OLD_VERL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"'
        in common_wrapper
    )
    assert (
        'TRAIN_PARQUET="${OLD_VERL_DIR}/data/4gpu/${EXPERT}_train.parquet"'
        in common_wrapper
    )
    assert "trainer.n_gpus_per_node != 4" in common_wrapper
    assert "rollout.tensor_model_parallel_size != 2" in common_wrapper
