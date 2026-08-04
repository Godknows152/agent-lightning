#!/usr/bin/env python3
"""Score JarvisIR images with the benchmark IQA metrics and compare with 0803."""

from __future__ import annotations

import csv
import importlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import packaging
import packaging.version
import torch
from omegaconf import DictConfig, OmegaConf

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
IQA_REPO = PROJECT_ROOT / "External_Tools/iqa_repos/IQA-PyTorch"
EXPECTED_METRICS = ("musiq", "maniqa", "clipiqa", "liqe")


def absolute_path(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{description} must be an absolute path: {value}")
    return path.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover_images(images_dir: Path, pattern: str, expected_samples: int) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"JarvisIR image directory is missing: {images_dir}")
    expression = re.compile(pattern)
    images: dict[str, Path] = {}
    unrecognized = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        match = expression.fullmatch(path.name)
        if match is None:
            unrecognized.append(path.name)
            continue
        sample_id = match.group(1)
        if sample_id in images:
            raise ValueError(f"Duplicate JarvisIR sample ID {sample_id}: {images[sample_id]} and {path}")
        images[sample_id] = path.resolve()
    if unrecognized:
        raise ValueError(f"Unrecognized files in JarvisIR image directory: {unrecognized[:5]}")
    if len(images) != expected_samples:
        raise ValueError(f"Expected {expected_samples} JarvisIR images, found {len(images)}")
    return images


def load_reference_rows(
    reference_csv: Path,
    reference_model: str,
    metrics: tuple[str, ...],
    expected_samples: int,
) -> dict[str, dict[str, str]]:
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Reference IQA score CSV is missing: {reference_csv}")
    rows = [row for row in read_csv(reference_csv) if row.get("model") == reference_model]
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} {reference_model} rows, found {len(rows)}")
    required = {"model", "sample_id", "image", *metrics}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Reference IQA CSV is missing columns: {sorted(missing)}")
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate {reference_model} sample IDs in {reference_csv}")
    for row in rows:
        for metric in metrics:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {metric} reference score for {row['sample_id']}: {value}")
    return by_id


def validate_alignment(images: dict[str, Path], reference_rows: dict[str, dict[str, str]]) -> list[str]:
    image_ids = set(images)
    reference_ids = set(reference_rows)
    if image_ids != reference_ids:
        raise ValueError(
            "JarvisIR/reference sample IDs differ: "
            f"missing JarvisIR={sorted(reference_ids - image_ids)}, "
            f"missing reference={sorted(image_ids - reference_ids)}"
        )
    return sorted(image_ids)


def score_images(
    sample_ids: list[str],
    images: dict[str, Path],
    metrics: tuple[str, ...],
    device: str,
    seed: int,
    progress_interval: int,
) -> list[dict[str, Any]]:
    packaging.version = packaging.version
    sys.path.insert(0, str(IQA_REPO))
    pyiqa = importlib.import_module("pyiqa")
    torch.manual_seed(seed)
    metric_models = {name: pyiqa.create_metric(name, device=device).eval() for name in metrics}
    rows = []
    started_at = time.perf_counter()
    for index, sample_id in enumerate(sample_ids, start=1):
        scores = {}
        for name, metric_model in metric_models.items():
            with torch.inference_mode():
                value = float(metric_model(str(images[sample_id])).reshape(-1)[0].item())
            torch.cuda.synchronize()
            if not math.isfinite(value):
                raise ValueError(f"{name} returned a non-finite score for {images[sample_id]}: {value}")
            scores[name] = value
        rows.append({"model": "JarvisIR", "sample_id": sample_id, "image": str(images[sample_id]), **scores})
        if index % progress_interval == 0 or index == len(sample_ids):
            elapsed = time.perf_counter() - started_at
            print(
                f"IQA scoring progress: {index}/{len(sample_ids)}; "
                f"elapsed={elapsed:.1f}s; rate={index / elapsed * 60.0:.2f} images/min",
                flush=True,
            )
    return rows


def build_comparison_outputs(
    jarvis_rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, str]],
    model_name: str,
    reference_model: str,
    metrics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows = []
    for row in jarvis_rows:
        reference = reference_rows[row["sample_id"]]
        comparison = {
            "sample_id": row["sample_id"],
            "jarvisir_image": row["image"],
            "reference_image": reference["image"],
        }
        for metric in metrics:
            jarvis_value = float(row[metric])
            reference_value = float(reference[metric])
            comparison[f"{model_name}_{metric}"] = jarvis_value
            comparison[f"{reference_model}_{metric}"] = reference_value
            comparison[f"delta_{metric}"] = jarvis_value - reference_value
        comparison_rows.append(comparison)

    model_summary_rows = []
    for name, rows in ((model_name, jarvis_rows), (reference_model, list(reference_rows.values()))):
        model_summary_rows.append(
            {
                "model": name,
                "sample_count": len(rows),
                **{f"{metric}_mean": float(np.mean([float(row[metric]) for row in rows])) for metric in metrics},
            }
        )

    metric_summary_rows = []
    for metric in metrics:
        deltas = np.asarray([float(row[f"delta_{metric}"]) for row in comparison_rows], dtype=np.float64)
        metric_summary_rows.append(
            {
                "metric": metric,
                f"{model_name}_mean": model_summary_rows[0][f"{metric}_mean"],
                f"{reference_model}_mean": model_summary_rows[1][f"{metric}_mean"],
                "mean_delta": float(deltas.mean()),
                "median_delta": float(np.median(deltas)),
                f"{model_name}_wins": int((deltas > 0).sum()),
                f"{reference_model}_wins": int((deltas < 0).sum()),
                "ties": int((deltas == 0).sum()),
            }
        )
    return comparison_rows, model_summary_rows, metric_summary_rows


@hydra.main(version_base=None, config_path="../../config/eval", config_name="jarvisir_iqa_comparison")
def main(config: DictConfig) -> None:
    images_dir = absolute_path(str(config.images_dir), "JarvisIR image directory")
    reference_csv = absolute_path(str(config.reference_scores_csv), "Reference score CSV")
    output_dir = absolute_path(str(config.output_dir), "Output directory")
    metrics = tuple(str(name) for name in config.metrics)
    if metrics != EXPECTED_METRICS:
        raise ValueError(f"Metrics must be {EXPECTED_METRICS}, found {metrics}")
    expected_samples = int(config.expected_samples)
    model_name = str(config.model_name)
    reference_model = str(config.reference_model)

    images = discover_images(images_dir, str(config.sample_id_pattern), expected_samples)
    reference_rows = load_reference_rows(reference_csv, reference_model, metrics, expected_samples)
    sample_ids = validate_alignment(images, reference_rows)
    print(f"Validated {len(sample_ids)} aligned JarvisIR/{reference_model} samples.", flush=True)
    jarvis_rows = score_images(
        sample_ids,
        images,
        metrics,
        str(config.device),
        int(config.seed),
        int(config.progress_interval),
    )
    for row in jarvis_rows:
        row["model"] = model_name
    comparison_rows, model_summary_rows, metric_summary_rows = build_comparison_outputs(
        jarvis_rows, reference_rows, model_name, reference_model, metrics
    )

    write_csv(output_dir / "iqa_scores.csv", jarvis_rows, ["model", "sample_id", "image", *metrics])
    comparison_fields = ["sample_id", "jarvisir_image", "reference_image"]
    for metric in metrics:
        comparison_fields.extend([f"{model_name}_{metric}", f"{reference_model}_{metric}", f"delta_{metric}"])
    write_csv(output_dir / f"comparison_vs_{reference_model}.csv", comparison_rows, comparison_fields)
    write_csv(
        output_dir / "summary.csv",
        model_summary_rows,
        ["model", "sample_count", *(f"{metric}_mean" for metric in metrics)],
    )
    write_csv(
        output_dir / f"comparison_summary_vs_{reference_model}.csv",
        metric_summary_rows,
        [
            "metric",
            f"{model_name}_mean",
            f"{reference_model}_mean",
            "mean_delta",
            "median_delta",
            f"{model_name}_wins",
            f"{reference_model}_wins",
            "ties",
        ],
    )
    (output_dir / "iqa_parameters.json").write_text(
        json.dumps(OmegaConf.to_container(config, resolve=True), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"IQA comparison complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
