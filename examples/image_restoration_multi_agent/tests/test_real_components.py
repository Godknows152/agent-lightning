"""Unit tests for stage D subprocess adapters without loading GPU models."""

from __future__ import annotations

import sys
from pathlib import Path

from config import EvaluatorSettings, IQAMetricConfig, SubprocessSettings
from evaluators.pyiqa_evaluator import PyiqaSubprocessEvaluator
from PIL import Image
from schemas import ExecutionStatus
from tool_registry import ToolDefinition, ToolRegistry, ToolRegistryConfig, ToolRuntime
from workers import SubprocessRestorationWorker


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(80, 100, 120)).save(path)


def _registry() -> ToolRegistry:
    return ToolRegistry(
        ToolRegistryConfig(
            registry_name="real-test",
            tools=[
                ToolDefinition(
                    name="test_tool",
                    description="Test subprocess tool",
                    runtime=ToolRuntime(adapter="verl_toolkit", model="test_model"),
                )
            ],
        )
    )


def test_subprocess_worker_publishes_only_valid_images(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import argparse, json, shutil\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--adapter')\n"
        "p.add_argument('--model'); p.add_argument('--external-tools-root'); p.add_argument('--device')\n"
        "a=p.parse_args(); shutil.copy2(a.input,a.output)\n"
        "print('RESULT_JSON='+json.dumps({'status':'success','model':a.model}))\n",
        encoding="utf-8",
    )
    settings = SubprocessSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
    )

    result = SubprocessRestorationWorker(settings, _registry()).restore(
        "test_tool", str(input_path), str(tmp_path / "output"), 0
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.output_path is not None
    assert Path(result.output_path).is_file()
    assert not list((tmp_path / "output").glob("*.partial.png"))


def test_subprocess_worker_rejects_and_cleans_corrupt_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "worker.py"
    entrypoint.write_text(
        "import argparse, json\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--adapter')\n"
        "p.add_argument('--model'); p.add_argument('--external-tools-root'); p.add_argument('--device')\n"
        "a=p.parse_args(); open(a.output,'wb').write(b'broken')\n"
        "print('RESULT_JSON='+json.dumps({'status':'success'}))\n",
        encoding="utf-8",
    )
    settings = SubprocessSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
    )

    result = SubprocessRestorationWorker(settings, _registry()).restore(
        "test_tool", str(input_path), str(tmp_path / "output"), 0
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.output_path is None
    assert not list((tmp_path / "output").glob("*.partial.png"))


def test_pyiqa_evaluator_normalizes_and_aggregates_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    entrypoint = tmp_path / "iqa.py"
    entrypoint.write_text(
        "import json\n"
        "print('RESULT_JSON='+json.dumps({'status':'success','raw_scores':"
        "{'topiq_nr':0.8,'musiq':60.0,'niqe':2.0}}))\n",
        encoding="utf-8",
    )
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        entrypoint=str(entrypoint),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        device="cpu",
        timeout_seconds=5,
        metrics=[
            IQAMetricConfig(name="topiq_nr", weight=0.45, minimum=0, maximum=1),
            IQAMetricConfig(name="musiq", weight=0.35, minimum=0, maximum=100),
            IQAMetricConfig(name="niqe", weight=0.20, minimum=0, maximum=10, higher_is_better=False),
        ],
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.6, original_score=0.5, best_score=0.6
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.normalized_scores == {"topiq_nr": 0.8, "musiq": 0.6, "niqe": 0.8}
    assert abs(result.aggregate_score - 0.73) < 1e-9
    assert result.is_new_best is True


def test_pyiqa_evaluator_failure_has_zero_gain(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    _write_image(input_path)
    settings = EvaluatorSettings(
        python_executable=sys.executable,
        entrypoint=str(tmp_path / "missing.py"),
        external_tools_root=str(tmp_path),
        iqa_repo=str(tmp_path),
        metrics=[IQAMetricConfig(name="topiq_nr", weight=1, minimum=0, maximum=1)],
    )

    result = PyiqaSubprocessEvaluator(settings, improvement_epsilon=1e-6).evaluate(
        str(input_path), previous_score=0.6, original_score=0.5, best_score=0.7
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.aggregate_score == 0.6
    assert result.delta_from_previous == 0.0
    assert result.is_new_best is False
