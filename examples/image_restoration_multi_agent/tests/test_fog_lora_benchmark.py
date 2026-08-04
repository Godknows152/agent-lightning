from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = EXAMPLE_DIR / "old_verl_grpo/scripts/eval/benchmark_fog_loras.py"
CALIBRATION_PATH = EXAMPLE_DIR / "calibrate_iqa_reward.py"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_script("fog_lora_benchmark_test_module", BENCHMARK_PATH)
calibration = load_script("iqa_calibration_test_module", CALIBRATION_PATH)
benchmark_config = benchmark.compose_benchmark_config()
settings = benchmark.settings_from_config(benchmark_config)


def test_configurable_calibration_weights_are_normalized_and_capped():
    metrics = ("musiq", "maniqa", "clipiqa", "liqe")
    delta_matrix = np.asarray(
        [
            [0.8, 0.5, 0.2, 0.7],
            [0.4, 0.3, 0.1, 0.5],
            [-0.2, 0.1, -0.1, 0.0],
            [1.1, 0.6, 0.4, 0.9],
        ],
        dtype=np.float64,
    )

    components = calibration.derive_metric_weight_components(metrics, [1.0, 0.5, 0.2, 0.8], delta_matrix)

    assert calibration.parse_metric_names("MUSIQ, maniqa,clipiqa,liqe") == metrics
    assert components["final_weight"].sum() == pytest.approx(1.0)
    assert components["final_weight"].max() <= 0.35 + 1e-12
    assert np.isfinite(components["delta_correlation"]).all()


def test_configurable_calibration_rejects_duplicate_metrics():
    with pytest.raises(ValueError, match="Duplicate IQA metrics"):
        calibration.parse_metric_names("musiq,maniqa,musiq")


def test_generated_tool_config_is_isolated_and_preserves_runtime_contract(tmp_path: Path):
    destination = tmp_path / "attempt/tool_config.yaml"
    benchmark.create_tool_config(
        settings.tool_config,
        destination,
        tmp_path / "attempt",
        settings.smoke_topology,
    )

    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    config = payload["tools"][0]["config"]
    schema = payload["tools"][0]["tool_schema"]["function"]
    assert config["device"] == "cuda:0"
    assert config["worker_devices"] == ["cuda:0"]
    assert config["preload"] is True
    assert config["auto_unload"] is False
    assert config["keep_models_loaded_between_sampling_steps"] is True
    assert Path(config["output_dir"]) == (tmp_path / "attempt/tool_outputs").resolve()
    assert schema["name"] == "restore_image"
    assert len(schema["parameters"]["properties"]["action"]["enum"]) == 17


def test_formal_hydra_config_composes_validation_only_runtime(tmp_path: Path):
    attempt_dir = tmp_path / "attempt"
    tool_config = benchmark.create_tool_config(
        settings.tool_config,
        attempt_dir / "tool_config.yaml",
        attempt_dir,
        settings.formal_topology,
    )
    adapter = settings.model_adapters[settings.model_names[0]]
    overrides = benchmark.build_hydra_overrides(
        adapter_path=adapter,
        dataset_path=settings.dataset,
        attempt_dir=attempt_dir,
        tool_config_path=tool_config,
        topology=settings.formal_topology,
        max_samples=settings.max_samples,
        sampling=settings.sampling,
        dataset_selection=settings.dataset_selection,
    )

    config = benchmark.compose_config(settings.config_dir, settings.config_name, overrides)
    benchmark.validate_composed_config(
        config,
        settings.formal_topology,
        adapter,
        settings.dataset,
        settings.max_samples,
        settings.sampling,
        settings.dataset_selection,
    )
    formal_tool_config = yaml.safe_load(tool_config.read_text(encoding="utf-8"))["tools"][0]["config"]

    assert config.data.val_max_samples == 100
    assert config.data.shuffle is False
    assert config.data.validation_shuffle is False
    assert config.actor_rollout_ref.rollout.temperature == pytest.approx(0.7)
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert config.trainer.val_only is True
    assert config.trainer.resume_mode == "disable"
    assert config.trainer.n_gpus_per_node == 2
    assert config.ray_kwargs.ray_init.num_gpus == 2
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES == "0,1,2"
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP_SYNC_CPU_OFFLOAD == "1"
    assert settings.formal_topology.tool_device == "cuda:2"
    assert settings.scoring_gpu == 2
    assert formal_tool_config["device"] == "cuda:2"
    assert formal_tool_config["iqa_device"] == "cuda:2"
    assert len(str(config.ray_kwargs.ray_init._temp_dir).encode()) < 30
    assert settings.benchmark_dir == benchmark.OLD_VERL_DIR / "outputs/eval/v4.1.1_vs_v2"
    assert settings.work_dir == benchmark.OLD_VERL_DIR / "log/eval/state"


def test_hydra_benchmark_config_uses_100_samples_and_absolute_lora_paths():
    assert settings.max_samples == 100
    assert settings.model_adapters["v2"] == (
        benchmark.OLD_VERL_DIR / "outputs/fog/LoRA/v2"
    ).resolve()
    assert settings.model_adapters["v4.1.1_0803"] == (
        benchmark.OLD_VERL_DIR / "outputs/fog/LoRA/v4.1.1/0803"
    ).resolve()
    assert all(path.is_absolute() and path.is_dir() for path in settings.model_adapters.values())
    assert settings.sampling.temperature == pytest.approx(0.7)


def test_benchmark_environment_enables_synchronous_fsdp_cpu_offload(tmp_path: Path):
    environment = benchmark.build_runtime_environment(tmp_path / "attempt", settings.smoke_topology)

    assert environment["VERL_FSDP_SYNC_CPU_OFFLOAD"] == "1"


def test_progress_counter_counts_only_complete_trajectory_events(tmp_path: Path):
    trajectory_log = tmp_path / "restoration_tool_info.log"
    trajectory_log.write_text(
        '\n'.join(
            [
                '{"event":"restoration_trajectory","sample_id":"a"}',
                '{"event":"tool_call","sample_id":"a"}',
                '{"event": "restoration_trajectory", "sample_id": "b"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert benchmark.count_completed_trajectories(trajectory_log) == 2


def test_trajectory_mapping_preserves_all_samples_and_falls_back_to_original(tmp_path: Path):
    original_a = tmp_path / "a.png"
    original_b = tmp_path / "b.png"
    final_a = tmp_path / "a-final.png"
    for path in (original_a, original_b, final_a):
        path.write_bytes(b"image")
    manifest = [
        {"sample_id": "fog-000000", "original_image": str(original_a)},
        {"sample_id": "fog-000001", "original_image": str(original_b)},
    ]
    trajectory_log = tmp_path / "restoration_tool_info.log"
    trajectory_log.write_text(
        json.dumps(
            {
                "event": "restoration_trajectory",
                "trajectory_id": "trajectory-a",
                "original_image": str(original_a),
                "final_image": str(final_a),
                "action_path": ["ridcp", "stop"],
                "termination_reason": "stop",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = benchmark.collect_final_outputs(manifest, trajectory_log, "v2")

    assert len(outputs) == 2
    assert outputs[0]["final_image"] == str(final_a.resolve())
    assert outputs[0]["used_original_fallback"] is False
    assert outputs[1]["final_image"] == str(original_b.resolve())
    assert outputs[1]["fallback_reason"] == "missing_trajectory"


def test_trajectory_mapping_rejects_samples_outside_configured_parquet_head(tmp_path: Path):
    expected_image = tmp_path / "expected.png"
    unexpected_image = tmp_path / "unexpected.png"
    final_image = tmp_path / "final.png"
    for path in (expected_image, unexpected_image, final_image):
        path.write_bytes(b"image")
    trajectory_log = tmp_path / "restoration_tool_info.log"
    trajectory_log.write_text(
        json.dumps(
            {
                "event": "restoration_trajectory",
                "original_image": str(unexpected_image),
                "final_image": str(final_image),
                "action_path": ["ridcp", "stop"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the configured parquet head"):
        benchmark.collect_final_outputs(
            [{"sample_id": "fog-000000", "original_image": str(expected_image)}],
            trajectory_log,
            "v2",
        )


def test_resume_requires_complete_outputs_and_existing_final_images(tmp_path: Path):
    model_dir = tmp_path / "model"
    attempt_dir = model_dir / "attempts/one"
    final_image = attempt_dir / "final.png"
    final_image.parent.mkdir(parents=True)
    final_image.write_bytes(b"image")
    outputs_path = attempt_dir / "final_outputs.jsonl"
    rows = [
        {
            "sample_id": "fog-000000",
            "model": "v2",
            "final_image": str(final_image),
        }
    ]
    benchmark.write_jsonl(outputs_path, rows)
    benchmark.write_json(model_dir / "latest_success.json", {"outputs_path": str(outputs_path)})
    manifest = [{"sample_id": "fog-000000"}]

    assert benchmark.completed_model_outputs(model_dir, manifest) == rows
    final_image.unlink()
    assert benchmark.completed_model_outputs(model_dir, manifest) is None


def test_complete_outputs_are_not_reused_without_deterministic_head_selection(tmp_path: Path):
    metadata = {
        "dataset": {
            "path": "/dataset/fog_val.parquet",
            "sha256": "dataset-hash",
            "evaluation_sample_count": 1,
            "selection": "parquet_rows_0_through_0",
            "shuffle": False,
            "validation_shuffle": False,
        },
        "adapters": {
            model: {"path": f"/adapters/{model}", "weights_sha256": f"hash-{model}"}
            for model in settings.model_names
        },
    }
    legacy_metadata = {
        **metadata,
        "dataset": {
            "path": "/dataset/fog_val.parquet",
            "sha256": "dataset-hash",
            "evaluation_sample_count": 1,
        },
        "status": "complete",
    }
    benchmark.write_json(tmp_path / "inference_parameters.json", legacy_metadata)

    assert not benchmark.benchmark_outputs_reusable(tmp_path, 1, settings.model_names, metadata)


def test_simple_outputs_contain_only_requested_trajectory_scores_and_means(tmp_path: Path):
    model_outputs = {}
    score_rows = []
    for model_name, action_path, duration in (
        (settings.model_names[0], ["ridcp", "stop"], 2.0),
        (settings.model_names[1], ["kanet", "stop"], 4.0),
    ):
        image = tmp_path / f"{model_name}.png"
        image.write_bytes(b"image")
        rows = []
        for index in range(2):
            sample_id = f"fog-{index:06d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "model": model_name,
                    "final_image": str(image),
                    "repaired_image": str(image),
                    "action_path": action_path,
                    "tool_calls": [{"action": action_path[0], "status": "success"}],
                    "duration_seconds": duration,
                }
            )
            score_rows.append(
                {
                    "image_id": f"{sample_id}::{model_name}",
                    "scores": {metric: float(index + 1) for metric in settings.metrics},
                }
            )
        model_outputs[model_name] = rows

    benchmark.write_simple_outputs(tmp_path, model_outputs, score_rows, settings.model_names, settings.metrics)

    trajectories = benchmark.read_jsonl(tmp_path / "trajectories.jsonl")
    with (tmp_path / "iqa_scores.csv").open(encoding="utf-8", newline="") as file:
        iqa_scores = list(csv.DictReader(file))
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as file:
        summary = {row["model"]: row for row in csv.DictReader(file)}
    assert len(trajectories) == 4
    assert len(iqa_scores) == 4
    assert trajectories[0]["iqa_scores"]["musiq"] == pytest.approx(1.0)
    assert trajectories[0]["restoration_tool_call_count"] == 1
    assert float(summary[settings.model_names[0]]["musiq_mean"]) == pytest.approx(1.5)
    assert float(summary[settings.model_names[1]]["average_restoration_tool_calls"]) == pytest.approx(1.0)
    assert float(summary[settings.model_names[1]]["average_duration_seconds"]) == pytest.approx(4.0)
    assert benchmark.benchmark_outputs_complete(tmp_path, 2, settings.model_names)
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "iqa_scores.jsonl").exists()
    assert not (tmp_path / "results").exists()
