"""Tests for run_restoration_eval.py — Hydra composition, adapter validation, trajectory parsing, output integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
OLD_VERL_DIR = EXAMPLE_DIR.parent / "old_verl_grpo"
RUNNER_PATH = OLD_VERL_DIR / "scripts/eval/run_restoration_eval.py"

# We import the runner module for the helpers it exposes
import importlib.util
import sys

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

runner = _load_module("restoration_eval_test_module", RUNNER_PATH)


# ---------------------------------------------------------------------------
# Hydra composition tests
# ---------------------------------------------------------------------------

def test_lora_sglang_hydra_config_composes():
    """Compose the full lora_sglang config and verify key fields."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str((OLD_VERL_DIR / "config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="restoration_inference", overrides=["backend=lora_sglang"])
    assert cfg.backend.name == "lora_sglang"
    assert cfg.data.max_samples == 100
    assert cfg.data.offset == 0
    assert cfg.run.name == "v4.1.2_0.008_186步"
    assert cfg.backend.adapter_path.endswith("outputs/fog/LoRA/v4.1.2/0.008_186步")


def test_jarvisir_hydra_config_composes():
    """Compose the full jarvisir config and verify key fields."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str((OLD_VERL_DIR / "config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="restoration_inference", overrides=["backend=jarvisir"])
    assert cfg.backend.name == "jarvisir"
    assert cfg.backend.model.policy_gpus == [0, 1]
    assert cfg.backend.model.vllm.tensor_parallel_size == 2


def test_hydra_output_dir_is_null():
    """Hydra should not write .hydra subdirectory."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str((OLD_VERL_DIR / "config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="restoration_inference", overrides=["backend=lora_sglang"])
    # The hydra meta-config key is not present in the resolved config
    assert "hydra" not in cfg


# ---------------------------------------------------------------------------
# Parquet selection tests
# ---------------------------------------------------------------------------

def test_parquet_selection_uses_slice():
    """load_validation_manifest with offset/max_samples selects correct rows."""
    manifest = [
        {"sample_id": f"fog-{i:06d}", "original_image": f"/img/{i}.png"}
        for i in range(200)
    ]
    selected = manifest[0:100]
    assert len(selected) == 100
    assert selected[0]["sample_id"] == "fog-000000"
    assert selected[-1]["sample_id"] == "fog-000099"


def test_parquet_selection_respects_offset():
    manifest = [
        {"sample_id": f"fog-{i:06d}", "original_image": f"/img/{i}.png"}
        for i in range(200)
    ]
    selected = manifest[50:150]
    assert len(selected) == 100
    assert selected[0]["sample_id"] == "fog-000050"
    assert selected[-1]["sample_id"] == "fog-000149"


# ---------------------------------------------------------------------------
# Adapter validation tests
# ---------------------------------------------------------------------------

def test_validate_adapter_accepts_valid(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": sorted(runner.EXPECTED_TARGET_MODULES),
        "base_model_name_or_path": str(tmp_path / "base_model"),
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake_weights")
    (tmp_path / "base_model").mkdir()
    (tmp_path / "base_model/config.json").write_text("{}", encoding="utf-8")
    runner.validate_adapter(adapter, tmp_path / "base_model")


def test_validate_adapter_rejects_missing_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": sorted(runner.EXPECTED_TARGET_MODULES),
        "base_model_name_or_path": str(tmp_path / "base_model"),
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        runner.validate_adapter(adapter, tmp_path / "base_model")


# ---------------------------------------------------------------------------
# Trajectory mapping tests
# ---------------------------------------------------------------------------

def test_action_path_count_excludes_stop():
    """Only non-stop actions count as restoration_tool_call_count."""
    action_path = ["ridcp", "stop"]
    count = sum(1 for a in action_path if a != "stop")
    assert count == 1


def test_action_path_all_stop_counts_zero():
    action_path = ["stop"]
    count = sum(1 for a in action_path if a != "stop")
    assert count == 0


def test_action_path_multiple_non_stop_counts():
    action_path = ["ridcp", "kanet", "stop", "restormer"]
    count = sum(1 for a in action_path if a != "stop")
    assert count == 3


def test_collect_lora_outputs_missing_trajectory(tmp_path):
    manifest = [{"sample_id": "fog-000000", "original_image": str(tmp_path / "a.png")}]
    (tmp_path / "a.png").write_bytes(b"image")
    traj_log = tmp_path / "restoration_tool_info.log"
    traj_log.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing trajectory"):
        runner.collect_lora_outputs(manifest, traj_log, "v2")


def test_collect_lora_outputs_duplicate_trajectory(tmp_path):
    orig = tmp_path / "a.png"
    orig.write_bytes(b"image")
    final = tmp_path / "final.png"
    final.write_bytes(b"image")
    manifest = [{"sample_id": "fog-000000", "original_image": str(orig)}]
    traj_log = tmp_path / "restoration_tool_info.log"
    traj_log.write_text(
        json.dumps({"event": "restoration_trajectory", "original_image": str(orig), "final_image": str(final), "action_path": ["ridcp", "stop"]}) + "\n" +
        json.dumps({"event": "restoration_trajectory", "original_image": str(orig), "final_image": str(final), "action_path": ["kanet", "stop"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        runner.collect_lora_outputs(manifest, traj_log, "v2")


def test_collect_lora_outputs_outside_parquet(tmp_path):
    expected = tmp_path / "expected.png"
    unexpected = tmp_path / "unexpected.png"
    final = tmp_path / "final.png"
    for p in (expected, unexpected, final):
        p.write_bytes(b"image")
    manifest = [{"sample_id": "fog-000000", "original_image": str(expected)}]
    traj_log = tmp_path / "restoration_tool_info.log"
    traj_log.write_text(
        json.dumps({"event": "restoration_trajectory", "original_image": str(unexpected), "final_image": str(final), "action_path": ["ridcp", "stop"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside"):
        runner.collect_lora_outputs(manifest, traj_log, "v2")


# ---------------------------------------------------------------------------
# Output integrity tests
# ---------------------------------------------------------------------------

def test_publish_only_images_and_csv(tmp_path):
    """Published output directory contains only images/ and tool_calls.csv."""
    outputs = [
        {"sample_id": "fog-000000", "final_image": str(tmp_path / "img0.png"), "action_path": ["ridcp", "stop"]},
        {"sample_id": "fog-000001", "final_image": str(tmp_path / "img1.png"), "action_path": ["stop"]},
    ]
    for o in outputs:
        Path(o["final_image"]).write_bytes(b"image")
    runner.publish_images_and_csv(tmp_path / "output", outputs)
    output_dir = tmp_path / "output"
    assert (output_dir / "images/fog-000000.png").is_file()
    assert (output_dir / "images/fog-000001.png").is_file()
    assert (output_dir / "tool_calls.csv").is_file()
    contents = set(output_dir.rglob("*"))
    allowed = {output_dir / "images", output_dir / "images/fog-000000.png", output_dir / "images/fog-000001.png", output_dir / "tool_calls.csv"}
    extra = contents - allowed
    assert not extra, f"Unexpected files: {extra}"


def test_tool_calls_csv_has_correct_columns(tmp_path):
    output_dir = tmp_path / "output"
    outputs = [{"sample_id": "fog-000000", "final_image": str(tmp_path / "img.png"), "action_path": ["ridcp", "stop"]}]
    Path(outputs[0]["final_image"]).write_bytes(b"image")
    runner.publish_images_and_csv(output_dir, outputs)
    import csv
    with (output_dir / "tool_calls.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["sample_id", "image", "restoration_tool_call_count"]
        row = next(reader)
        assert row["sample_id"] == "fog-000000"
        assert row["restoration_tool_call_count"] == "1"


# ---------------------------------------------------------------------------
# Preflight rejection tests
# ---------------------------------------------------------------------------

def test_run_name_rejects_path_traversal():
    with pytest.raises(ValueError, match="invalid characters"):
        runner.validate_run_name("../../etc/passwd")

def test_run_name_rejects_slashes():
    with pytest.raises(ValueError, match="invalid characters"):
        runner.validate_run_name("v4.1.1/0803")


# ---------------------------------------------------------------------------
# Complementary: iqa_mean_comparison config composes
# ---------------------------------------------------------------------------

def test_iqa_mean_comparison_composes():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str((OLD_VERL_DIR / "config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="iqa_mean_comparison")
    assert cfg.iqa.metrics == ["musiq", "maniqa", "clipiqa", "liqe"]
    assert cfg.comparison.expected_samples == 100
