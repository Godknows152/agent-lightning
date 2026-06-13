#!/usr/bin/env python3
"""Evaluate one image with IQA-PyTorch inside the isolated `verl` environment."""

from __future__ import annotations

import argparse
import contextlib
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
packaging.version = packaging.version


class IQAMetric(Protocol):
    """Callable metric interface used from IQA-PyTorch."""

    def __call__(self, image_path: str) -> torch.Tensor: ...


class PyiqaModule(Protocol):
    """Runtime subset of the dynamically imported pyiqa package."""

    __version__: str

    def create_metric(self, metric_name: str, *, device: str) -> IQAMetric: ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    sys.path.insert(0, str(repo))
    pyiqa = cast(PyiqaModule, importlib.import_module("pyiqa"))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    raw_scores: dict[str, float] = {}
    metric_seconds: dict[str, float] = {}
    diagnostics = io.StringIO()
    for metric_name in [item.strip() for item in args.metrics.split(",") if item.strip()]:
        started = time.perf_counter()
        with contextlib.redirect_stdout(diagnostics):
            metric = pyiqa.create_metric(metric_name, device=str(device))
            with torch.inference_mode():
                score = metric(str(input_path))
        torch.cuda.synchronize(device)
        raw_scores[metric_name] = float(score.reshape(-1)[0].item())
        metric_seconds[metric_name] = time.perf_counter() - started
        del metric
        torch.cuda.empty_cache()

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
