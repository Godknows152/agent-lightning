#!/usr/bin/env python3
"""IQA mean comparison for multiple restoration output sets.

Usage:
  # Run comparison
  python score_restoration_eval.py comparison.name=my_compare \
    comparison.output_csv=/path/to/output.csv \
    'sets=[{name:model1,output_dir:/path/to/model1},{name:model2,output_dir:/path/to/model2}]'

  # Score-worker subprocess (internal)
  python score_restoration_eval.py command=score-worker \
    iqa.manifest_path=/path/to/manifest.json
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[5]
IQA_REPO = PROJECT_ROOT / "External_Tools/iqa_repos/IQA-PyTorch"
EXPECTED_METRICS = ("musiq", "maniqa", "clipiqa", "liqe")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_set(name: str, output_dir: Path, expected_samples: int) -> dict[str, dict[str, Any]]:
    """Load one output set and return {sample_id: {sample_id, image_path, tool_count}}."""
    images_dir = output_dir / "images"
    csv_path = output_dir / "tool_calls.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing tool_calls.csv in {output_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images/ directory in {output_dir}")
    rows = read_csv(csv_path)
    if len(rows) != expected_samples:
        raise ValueError(
            f"{name}: expected {expected_samples} samples, found {len(rows)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row["sample_id"])
        if sid in result:
            raise ValueError(f"{name}: duplicate sample_id {sid}")
        img_path = (output_dir / row["image"]).resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"{name}: image missing for {sid}: {img_path}")
        tc = str(row.get("restoration_tool_call_count", ""))
        if not tc.isdigit() or int(tc) < 0:
            raise ValueError(f"{name}: invalid tool count for {sid}: {tc!r}")
        result[sid] = {
            "sample_id": sid,
            "image_path": str(img_path),
            "tool_count": int(tc),
        }
    return result


def _score_worker(manifest: list[dict[str, Any]], metrics: list[str], device: str,
                  seed: int, progress_interval: int) -> list[dict[str, Any]]:
    """Score images in a subprocess with IQA models loaded on the given device."""
    sys.path.insert(0, str(IQA_REPO))
    pyiqa = importlib.import_module("pyiqa")
    torch.manual_seed(seed)
    metric_models = {}
    for name in metrics:
        metric_models[name] = pyiqa.create_metric(name, device=device).eval()
    rows = []
    started_at = time.perf_counter()
    for idx, entry in enumerate(manifest, start=1):
        sample_id = entry["sample_id"]
        image_path = entry["image_path"]
        scores: dict[str, float] = {}
        for name, model in metric_models.items():
            with torch.inference_mode():
                value = float(model(str(image_path)).reshape(-1)[0].item())
            torch.cuda.synchronize()
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} returned non-finite score for {sample_id}: {value}"
                )
            scores[name] = value
        rows.append({"sample_id": sample_id, **scores})
        if idx % progress_interval == 0 or idx == len(manifest):
            elapsed = time.perf_counter() - started_at
            print(
                f"IQA progress: {idx}/{len(manifest)}; "
                f"elapsed={elapsed:.1f}s; rate={idx/elapsed*60.0:.2f} img/min",
                flush=True,
            )
    return rows


@hydra.main(version_base=None, config_path="../../config/eval", config_name="iqa_mean_comparison")
def main(config: DictConfig) -> int:
    cfg = OmegaConf.to_container(config, resolve=True)
    command = str(cfg.get("command", "run"))

    # ---- score-worker subprocess ----
    if command == "score-worker":
        manifest = json.loads(Path(str(cfg["iqa"]["manifest_path"])).read_text(encoding="utf-8"))
        metrics = [str(m) for m in cfg["iqa"]["metrics"]]
        device = str(cfg["iqa"]["device"])
        seed = int(cfg["iqa"]["seed"])
        progress_interval = int(cfg["iqa"]["progress_interval"])
        results = _score_worker(manifest, metrics, device, seed, progress_interval)
        print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False))
        return 0

    # ---- main orchestrator ----
    comparison_name = str(cfg["comparison"]["name"])
    output_csv = Path(str(cfg["comparison"]["output_csv"])).resolve()
    require_identical = bool(cfg["comparison"]["require_identical_sample_ids"])
    expected_samples = int(cfg["comparison"]["expected_samples"])
    sets_raw = cfg["sets"]
    metrics = [str(m) for m in cfg["iqa"]["metrics"]]
    seed = int(cfg["iqa"]["seed"])
    progress_interval = int(cfg["iqa"]["progress_interval"])
    state_root = Path(str(cfg["paths"]["state_root"])).resolve()

    if metrics != list(EXPECTED_METRICS):
        raise ValueError(f"Metrics must be {EXPECTED_METRICS}, found {metrics}")

    if not sets_raw:
        raise ValueError("At least one output set is required")

    # Load all sets
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for s in sets_raw:
        name = str(s["name"])
        output_dir = Path(str(s["output_dir"])).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Set directory not found: {output_dir}")
        loaded[name] = _load_set(name, output_dir, expected_samples)

    # Validate alignment
    if require_identical:
        id_sets = [set(ld.keys()) for ld in loaded.values()]
        master = id_sets[0]
        for i, ids in enumerate(id_sets[1:], start=1):
            if ids != master:
                raise ValueError(
                    f"Sample ID mismatch: set {list(loaded)[i]} has {len(ids)} IDs, "
                    f"expected {len(master)}. Missing: {sorted(master - ids)}, "
                    f"extra: {sorted(ids - master)}"
                )
    sample_ids = sorted(next(iter(loaded.values())))

    # Build scoring manifest
    manifest = []
    for sid in sample_ids:
        for model_name, entries in loaded.items():
            entry = entries[sid]
            manifest.append({
                "sample_id": f"{sid}::{model_name}",
                "image_path": entry["image_path"],
            })

    # Launch score-worker subprocess
    import subprocess
    import os as _os

    state_dir = state_root / comparison_name
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "score_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    result_path = state_dir / "score_result.json"

    python = str(cfg["iqa"]["python"])
    physical_gpu = int(cfg["iqa"]["physical_gpu"])
    worker_env = _os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    worker_env["PYTHONUNBUFFERED"] = "1"

    worker_cmd = [
        python,
        str(SCRIPT_PATH),
        f"command=score-worker",
        f"iqa.manifest_path={manifest_path}",
        f"iqa.metrics={metrics}",
        f"iqa.device=cuda:0",
        f"iqa.seed={seed}",
        f"iqa.progress_interval={progress_interval}",
    ]
    print(f"Launching IQA score worker on physical GPU {physical_gpu}", flush=True)
    proc = subprocess.run(
        worker_cmd, cwd=PROJECT_ROOT, env=worker_env,
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"IQA score worker failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    worker_output = json.loads(proc.stdout)
    if worker_output.get("status") != "ok":
        raise RuntimeError(f"IQA score worker error: {worker_output}")
    result_path.write_text(proc.stdout, encoding="utf-8")

    # Parse results back into per-model dicts
    all_scores: dict[str, list[dict[str, Any]]] = {name: [] for name in loaded}
    for entry in worker_output["results"]:
        full_id = entry["sample_id"]
        sid, model_name = full_id.rsplit("::", 1)
        if model_name not in all_scores:
            continue
        all_scores[model_name].append({"sample_id": sid, **{m: entry[m] for m in metrics}})

    # Compute means
    summary_rows = []
    for model_name in sorted(loaded):
        scores = all_scores[model_name]
        if len(scores) != expected_samples:
            raise ValueError(
                f"{model_name}: expected {expected_samples} scores, got {len(scores)}"
            )
        means = {}
        for m in metrics:
            vals = [float(s[m]) for s in scores]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"{model_name}: non-finite {m} values")
            means[f"{m}_mean"] = float(np.mean(vals))
        summary_rows.append({
            "model": model_name,
            "sample_count": expected_samples,
            **means,
        })

    # Write output CSV
    fieldnames = ["model", "sample_count", *(f"{m}_mean" for m in metrics)]
    write_csv(output_csv, summary_rows, fieldnames)
    print(f"Comparison complete: {output_csv}", flush=True)
    print(json.dumps(summary_rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
