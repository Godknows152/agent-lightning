#!/usr/bin/env python3
"""Evaluate one image with IQA-PyTorch inside the isolated `verl` environment."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Protocol, cast

import packaging
import packaging.version
import torch

RESULT_PREFIX = "RESULT_JSON="
READY_PREFIX = "READY_JSON="
packaging.version = packaging.version


class IQAMetric(Protocol):
    """Callable metric interface used from IQA-PyTorch."""

    def __call__(self, image_path: str) -> torch.Tensor: ...


class PyiqaModule(Protocol):
    """Runtime subset of the dynamically imported pyiqa package."""

    __version__: str

    def create_metric(self, metric_name: str, *, device: str) -> IQAMetric: ...


def _emit(prefix: str, payload: dict[str, object]) -> None:
    print(f"{prefix}{json.dumps(payload, separators=(',', ':'))}", flush=True)


def _evaluate_metrics(
    metrics: dict[str, IQAMetric],
    input_path: Path,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    raw_scores: dict[str, float] = {}
    metric_seconds: dict[str, float] = {}
    for metric_name, metric in metrics.items():
        started = time.perf_counter()
        with torch.inference_mode():
            score = metric(str(input_path))
        torch.cuda.synchronize(device)
        raw_scores[metric_name] = float(score.reshape(-1)[0].item())
        metric_seconds[metric_name] = time.perf_counter() - started
    return raw_scores, metric_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    pyiqa = cast(PyiqaModule, importlib.import_module("pyiqa"))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    metrics: dict[str, IQAMetric] = {}

    def wake_metrics() -> None:
        if metrics:
            return
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(diagnostics):
            metrics.update(
                {metric_name: pyiqa.create_metric(metric_name, device=str(device)) for metric_name in metric_names}
            )
        torch.cuda.synchronize(device)

    def sleep_metrics() -> None:
        metrics.clear()
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    wake_metrics()
    torch.cuda.synchronize(device)
    if args.serve_jsonl:
        _emit(
            READY_PREFIX,
            {
                "status": "ready",
                "metrics": metric_names,
                "device": str(device),
                "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            },
        )
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id: object = None
            try:
                request = json.loads(line)
                request_id = request.get("request_id")
                command = request.get("command", "infer")
                if command == "sleep":
                    sleep_metrics()
                    _emit(
                        RESULT_PREFIX,
                        {
                            "request_id": request_id,
                            "status": "success",
                            "state": "sleeping",
                            "device": str(device),
                            "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                        },
                    )
                    continue
                if command == "wake":
                    wake_metrics()
                    _emit(
                        RESULT_PREFIX,
                        {
                            "request_id": request_id,
                            "status": "success",
                            "state": "ready",
                            "device": str(device),
                            "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                        },
                    )
                    continue
                if command != "infer":
                    raise ValueError(f"unknown worker command: {command}")
                if not metrics:
                    raise RuntimeError("IQA models are sleeping")
                input_path = Path(request["input"]).expanduser().resolve()
                raw_scores, metric_seconds = _evaluate_metrics(metrics, input_path, device)
                _emit(
                    RESULT_PREFIX,
                    {
                        "request_id": request_id,
                        "status": "success",
                        "raw_scores": raw_scores,
                        "metric_seconds": metric_seconds,
                        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                        "pyiqa_version": pyiqa.__version__,
                        "torch_version": torch.__version__,
                    },
                )
            except Exception as error:
                _emit(
                    RESULT_PREFIX,
                    {
                        "request_id": request_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
        return
    if args.input is None:
        raise ValueError("one-shot mode requires --input")

    input_path = args.input.expanduser().resolve()
    raw_scores, metric_seconds = _evaluate_metrics(metrics, input_path, device)
    result = {
        "status": "success",
        "raw_scores": raw_scores,
        "metric_seconds": metric_seconds,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "pyiqa_version": pyiqa.__version__,
        "torch_version": torch.__version__,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
