from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "old_verl_grpo"
    / "scripts"
    / "resolve_training_run_name.py"
)
OLD_VERL_ROOT = Path(__file__).resolve().parents[1] / "old_verl_grpo"
PROJECT_NAMES = {
    "fog": "FogRL",
    "rain": "RainRL",
    "snow": "SnowRL",
    "lowlight": "LowLightRL",
}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_training_run_name", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v4_launchers_default_to_versioned_swanlab_names() -> None:
    for expert in ("fog", "rain", "snow", "lowlight"):
        launcher = OLD_VERL_ROOT / "scripts" / expert / f"{expert}_v4.sh"
        content = launcher.read_text(encoding="utf-8")
        assert 'VERSION="v4"' in content
        assert (
            'export OLD_VERL_EXPERIMENT_NAME="${OLD_VERL_EXPERIMENT_NAME:-${EXPERT}_${VERSION}}"'
            in content
        )
        if expert == "lowlight":
            assert 'RUNTIME_EXPERT="low_light"' in content
            assert '"${RUNTIME_EXPERT}"' in content


def test_all_versioned_launchers_isolate_training_and_tool_logs() -> None:
    expected_export = 'export OLD_VERL_LOG_DIR="${OLD_VERL_LOG_DIR:-${OLD_VERL_DIR}/log/${EXPERT}/${VERSION}/2gpu}"'
    expected_output_export = 'export OLD_VERL_OUTPUT_DIR="${OLD_VERL_OUTPUT_DIR:-${OLD_VERL_DIR}/outputs/${EXPERT}/${VERSION}/2gpu}"'
    for expert in ("fog", "rain", "snow", "lowlight"):
        for version in ("v1", "v2", "v3", "v4", "v4.1.1"):
            script_version = "v4_1_1" if version == "v4.1.1" else version
            launcher = OLD_VERL_ROOT / "scripts" / expert / f"{expert}_{script_version}.sh"
            content = launcher.read_text(encoding="utf-8")
            assert expected_export in content, launcher
            if version not in {"v4", "v4.1.1"}:
                assert expected_output_export in content, launcher
            assert 'LOG_DIR="${OLD_VERL_LOG_DIR}"' in content, launcher
            assert (
                'MAIN_LOG="${LOG_DIR}/${EXPERT}_${VERSION}_${TIMESTAMP}.log"' in content
            ), launcher

            config = (
                OLD_VERL_ROOT
                / "config"
                / expert
                / version
                / f"{expert}_config_2gpu.yaml"
            )
            config_content = config.read_text(encoding="utf-8")
            expected_log_dir = f"/old_verl_grpo/log/{expert}/{version}/2gpu"
            assert expected_log_dir in config_content, config
            expected_output_dir = f"old_verl_grpo/outputs/{expert}/{version}/2gpu"
            assert expected_output_dir in config_content, config


def test_common_wrapper_routes_both_tool_logs_to_resolved_version_directory() -> None:
    wrapper = (OLD_VERL_ROOT / "run_expert_old_verl_grpo_2gpu.sh").read_text(
        encoding="utf-8"
    )
    assert 'LOG_DIR="${OLD_VERL_LOG_DIR:-${LOG_ROOT}/${EXPERT}/2gpu}"' in wrapper
    assert '--output-root "${OLD_VERL_DIR}/outputs/2gpu"' in wrapper
    assert (
        "\"trainer.ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOG_DIR='${LOG_DIR}'\""
        in wrapper
    )
    assert 'export VERL_LOG_DIR="${LOG_DIR}"' in wrapper
    assert 'TOOL_INFO_LOG="${LOG_DIR}/restoration_tool_info.log"' in wrapper
    assert 'TOOL_DEBUG_LOG="${LOG_DIR}/restoration_tools.log"' in wrapper


def test_two_gpu_configs_keep_outputs_logs_and_swanlab_under_two_gpu_subdirectories() -> (
    None
):
    for expert in ("fog", "rain", "snow", "lowlight"):
        for version in ("v1", "v2", "v3", "v4", "v4.1.1"):
            config = (
                OLD_VERL_ROOT
                / "config"
                / expert
                / version
                / f"{expert}_config_2gpu.yaml"
            )
            content = config.read_text(encoding="utf-8")
            assert f"project_name: {PROJECT_NAMES[expert]}" in content, config
            assert f'experiment_name: "{expert}_{version}"' in content, config
            assert f"old_verl_grpo/outputs/{expert}/{version}/2gpu" in content, config
            assert (
                f"old_verl_grpo/outputs/{expert}/{version}/2gpu/swanlab" in content
            ), config
            assert f"old_verl_grpo/log/{expert}/{version}/2gpu" in content, config
            assert "/2gpu/2gpu/" not in content, config


def test_fresh_run_uses_expert_and_current_month_day(tmp_path: Path) -> None:
    module = _load_module()

    naming = module.resolve_run_naming(
        expert="rain",
        output_root=tmp_path,
        now=datetime(2026, 7, 20, 9, 30),
    )

    assert naming.experiment_name == "rain_0720"
    assert naming.output_dir == tmp_path / "rain_0720"
    assert naming.swanlab_log_dir == tmp_path / "rain_0720" / "swanlab"
    assert naming.resume_from_path is None


def test_explicit_checkpoint_adds_continuation_suffix(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "previous_run" / "global_step_40"
    checkpoint.mkdir(parents=True)

    naming = module.resolve_run_naming(
        expert="snow",
        output_root=tmp_path,
        now=datetime(2026, 7, 20, 9, 30),
        resume_mode="resume_path",
        resume_from_path=checkpoint,
    )

    assert naming.experiment_name == "snow_0720_续"
    assert naming.output_dir == tmp_path / "snow_0720_续"
    assert naming.swanlab_log_dir == tmp_path / "snow_0720_续" / "swanlab"
    assert naming.resume_from_path == checkpoint.resolve()


def test_auto_resume_uses_latest_checkpoint_and_continuation_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fresh_output = tmp_path / "low_light_0720"
    (fresh_output / "global_step_5").mkdir(parents=True)
    latest_checkpoint = fresh_output / "global_step_15"
    latest_checkpoint.mkdir()
    (fresh_output / "latest_checkpointed_iteration.txt").write_text(
        "15\n", encoding="utf-8"
    )

    naming = module.resolve_run_naming(
        expert="low_light",
        output_root=tmp_path,
        now=datetime(2026, 7, 20, 9, 30),
    )

    assert naming.experiment_name == "low_light_0720_续"
    assert naming.output_dir == tmp_path / "low_light_0720_续"
    assert naming.resume_from_path == latest_checkpoint.resolve()


def test_restart_of_continuation_prefers_its_checkpoint(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "fog_0720" / "global_step_10").mkdir(parents=True)
    continuation_checkpoint = tmp_path / "fog_0720_续" / "global_step_20"
    continuation_checkpoint.mkdir(parents=True)

    naming = module.resolve_run_naming(
        expert="fog",
        output_root=tmp_path,
        now=datetime(2026, 7, 20, 9, 30),
    )

    assert naming.experiment_name == "fog_0720_续"
    assert naming.output_dir == tmp_path / "fog_0720_续"
    assert naming.resume_from_path == continuation_checkpoint.resolve()


def test_checkpoint_output_directory_is_independent_from_swanlab_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    configured_output = tmp_path / "rain_from_sft_lora_0718"
    checkpoint = configured_output / "global_step_60"
    checkpoint.mkdir(parents=True)

    naming = module.resolve_run_naming(
        expert="rain",
        output_root=tmp_path,
        output_dir=configured_output,
        now=datetime(2026, 7, 21, 9, 30),
    )

    assert naming.experiment_name == "rain_0721_续"
    assert naming.output_dir == configured_output.resolve()
    assert naming.swanlab_log_dir == configured_output.resolve() / "swanlab"
    assert naming.resume_from_path == checkpoint.resolve()


def test_explicit_experiment_name_gets_continuation_suffix(tmp_path: Path) -> None:
    module = _load_module()
    configured_output = tmp_path / "fog" / "v3"
    checkpoint = configured_output / "global_step_20"
    checkpoint.mkdir(parents=True)

    naming = module.resolve_run_naming(
        expert="fog",
        output_root=tmp_path,
        output_dir=configured_output,
        experiment_name="fog_v3",
    )

    assert naming.experiment_name == "fog_v3_续"
    assert naming.output_dir == configured_output.resolve()
    assert naming.resume_from_path == checkpoint.resolve()


def test_fresh_explicit_experiment_name_keeps_its_original_name(tmp_path: Path) -> None:
    module = _load_module()
    configured_output = tmp_path / "rain" / "v1"

    naming = module.resolve_run_naming(
        expert="rain",
        output_root=tmp_path,
        output_dir=configured_output,
        experiment_name="rain_v1",
    )

    assert naming.experiment_name == "rain_v1"
    assert naming.output_dir == configured_output.resolve()
    assert naming.resume_from_path is None


def test_existing_continuation_suffix_is_not_duplicated(tmp_path: Path) -> None:
    module = _load_module()
    configured_output = tmp_path / "snow" / "v2"
    checkpoint = configured_output / "global_step_8"
    checkpoint.mkdir(parents=True)

    naming = module.resolve_run_naming(
        expert="snow",
        output_root=tmp_path,
        output_dir=configured_output,
        experiment_name="snow_v2_续",
    )

    assert naming.experiment_name == "snow_v2_续"
    assert naming.output_dir == configured_output.resolve()
    assert naming.resume_from_path == checkpoint.resolve()
