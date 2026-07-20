from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "old_verl_grpo" / "scripts" / "resolve_training_run_name.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resolve_training_run_name", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_auto_resume_uses_latest_checkpoint_and_continuation_name(tmp_path: Path) -> None:
    module = _load_module()
    fresh_output = tmp_path / "low_light_0720"
    (fresh_output / "global_step_5").mkdir(parents=True)
    latest_checkpoint = fresh_output / "global_step_15"
    latest_checkpoint.mkdir()
    (fresh_output / "latest_checkpointed_iteration.txt").write_text("15\n", encoding="utf-8")

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


def test_checkpoint_output_directory_is_independent_from_swanlab_name(tmp_path: Path) -> None:
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
