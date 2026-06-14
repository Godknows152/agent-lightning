#!/usr/bin/env python3
"""Calibrate the four-metric IQA reward from sampled training images.

Run from the Agent_Lightning repository root:

    /home/LXJ/anaconda3/envs/verl/bin/python \
      examples/image_restoration_multi_agent/calibrate_iqa_reward.py \
      --gpus 0,1 --samples-per-class 8

The coordinator launches one persistent restoration process per action, scores
all original/restored images on two GPUs, and writes resumable calibration
artifacts below the example's ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
EXAMPLE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "LlamaFactory/image_restoration_diagnosis/data/images/train"
DEFAULT_TOOLS_CONFIG = EXAMPLE_DIR / "config/tools.yaml"
DEFAULT_EXTERNAL_TOOLS = PROJECT_ROOT / "External_Tools"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "artifacts/iqa_calibration_v1"
DEFAULT_PYTHON = Path("/home/LXJ/anaconda3/envs/verl/bin/python")
IQA_REPO = DEFAULT_EXTERNAL_TOOLS / "iqa_repos/IQA-PyTorch"

LABELS = ("fog", "snow", "rain", "low_light")
CLASS_DIRECTORIES = {"fog": "fog", "snow": "snow", "rain": "rain", "low_light": "low_light"}
METRICS = ("maniqa", "niqe", "clipiqa", "topiq_nr")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TARGETED_ACTIONS = {
    "fog": {"ridcp", "kanet", "focalnet_dehaze", "mb_taylorformer_dehaze"},
    "snow": {"turbo_snow", "snowmaster", "focalnet_desnow"},
    "rain": {"turbo_rain", "s2former", "idt"},
    "low_light": {"retinexformer_fivek", "hvicidnet", "lightdiff"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["coordinator", "restore", "score"], default="coordinator")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--tools-config", type=Path, default=DEFAULT_TOOLS_CONFIG)
    parser.add_argument("--external-tools-root", type=Path, default=DEFAULT_EXTERNAL_TOOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action")
    parser.add_argument("--gpu")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def restoration_progress_path(args: argparse.Namespace) -> Path:
    suffix = f".rank{args.rank}" if args.rank is not None and args.world_size is not None else ""
    return args.output_dir / "progress/restoration" / f"{args.action}{suffix}.jsonl"


def load_restoration_completed(args: argparse.Namespace) -> set[str]:
    progress_dir = args.output_dir / "progress/restoration"
    paths = [progress_dir / f"{args.action}.jsonl", *sorted(progress_dir.glob(f"{args.action}.rank*.jsonl"))]
    return {row["sample_id"] for path in paths for row in read_jsonl(path) if row.get("status") == "success"}


def load_tools(path: Path) -> list[dict[str, Any]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(payload["tools"])


def resize_image(source: Path, destination: Path, max_pixels: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / pixels)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        image.save(destination)


def create_sample_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = args.output_dir / "sample_manifest.jsonl"
    if args.resume and manifest_path.exists():
        rows = read_jsonl(manifest_path)
        expected = args.samples_per_class * len(LABELS)
        if len(rows) != expected:
            raise ValueError(f"Existing manifest has {len(rows)} rows; expected {expected}.")
        return rows

    rng = random.Random(args.seed)
    rows = []
    for label in LABELS:
        class_dir = args.train_root / CLASS_DIRECTORIES[label]
        candidates = sorted(
            item.resolve() for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(candidates) < args.samples_per_class:
            raise ValueError(f"{class_dir} has only {len(candidates)} usable images.")
        for source in rng.sample(candidates, args.samples_per_class):
            sample_id = f"{label}-{source.stem}"
            staged = args.output_dir / "images/original" / label / f"{source.stem}.png"
            resize_image(source, staged, args.max_pixels)
            rows.append(
                {
                    "sample_id": sample_id,
                    "degradation_type": label,
                    "source_path": str(source),
                    "original_path": str(staged.resolve()),
                }
            )
    rng.shuffle(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def import_restoration_entrypoint() -> ModuleType:
    path = EXAMPLE_DIR / "tool_runtime/restoration_entrypoint.py"
    spec = importlib.util.spec_from_file_location("iqa_calibration_restoration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_candidate_batch(
    tool: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace, completed: set[str]
) -> None:
    import torch

    runtime = tool["runtime"]
    module = import_restoration_entrypoint()
    repo = (args.external_tools_root / runtime["repo"]).resolve()
    checkpoint = (args.external_tools_root / runtime["checkpoint"]).resolve()
    model_name = runtime["model"]
    device = torch.device("cuda:0")
    if model_name == "nafnet_denoise":
        model = module._load_nafnet(repo, checkpoint)
    elif model_name in {"focal_dehaze", "focal_desnow"}:
        model = module._load_focalnet(repo, checkpoint, model_name)
    elif model_name == "mb_taylorformer_dehaze":
        model = module._load_mb_taylorformer(repo, checkpoint)
    else:
        raise ValueError(f"Unsupported candidate model: {model_name}")
    model = model.eval().to(device)

    pending_count = len(rows) - len(completed)
    processed = 0
    for row in rows:
        if row["sample_id"] in completed:
            continue
        output_path = args.output_dir / "images/restored" / tool["name"] / f"{row['sample_id']}.png"
        started = time.perf_counter()
        image = module._load_image(Path(row["original_path"]), device)
        original_height, original_width = image.shape[-2:]
        if model_name.startswith("focal_"):
            image, original_height, original_width = module._pad_to_multiple(image, 4)
        elif model_name == "mb_taylorformer_dehaze":
            image, original_height, original_width = module._pad_to_multiple(image, 8)
        with torch.inference_mode():
            restored = model(image)
            if model_name.startswith("focal_"):
                restored = restored[2]
        module._save_image(restored[:, :, :original_height, :original_width], output_path)
        append_jsonl(
            restoration_progress_path(args),
            {
                "sample_id": row["sample_id"],
                "action": tool["name"],
                "output_path": str(output_path.resolve()),
                "seconds": time.perf_counter() - started,
                "status": "success",
            },
        )
        processed += 1
        if processed % 100 == 0 or processed == pending_count:
            print(f"[{tool['name']}] {processed}/{pending_count}", flush=True)
        del image, restored
    del model
    gc.collect()
    torch.cuda.empty_cache()


def prepare_toolkit_imports(external_tools_root: Path, model_name: str) -> None:
    bundle = external_tools_root / "verl_bundle"
    agent_tools_dir = bundle / "agent_tools"
    source_directories = {
        "retinexformer_fivek": "Retinexformer",
        "lightdiff": "LightenDiffusion",
        "idt": "IDT",
        "ridcp": "RIDCP",
        "turbo_rain": "img2img_turbo",
        "turbo_snow": "img2img_turbo",
    }
    sys.path.insert(0, str(bundle))
    selected_source = source_directories.get(model_name)
    if selected_source:
        source_dir = agent_tools_dir / selected_source
        sys.path.insert(0, str(source_dir))
        if (source_dir / "src").is_dir():
            sys.path.insert(0, str(source_dir / "src"))


def run_toolkit_batch(
    tool: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace, completed: set[str]
) -> None:
    import torch

    runtime = tool["runtime"]
    model_name = runtime["model"]
    prepare_toolkit_imports(args.external_tools_root, model_name)
    toolkit_module = importlib.import_module("agent_tools.restoration_toolkit")
    toolkit = toolkit_module.RestorationToolkit(
        models=[model_name],
        device="cuda:0",
        load_iqa=False,
        preload=False,
        auto_unload=False,
    )
    if toolkit.load_single_model(model_name) is None:
        raise RuntimeError(f"Failed to load restoration model: {model_name}")

    pending_count = len(rows) - len(completed)
    processed = 0
    for row in rows:
        if row["sample_id"] in completed:
            continue
        output_path = args.output_dir / "images/restored" / tool["name"] / f"{row['sample_id']}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"{model_name}-", dir=output_path.parent) as temporary_dir:
            generated = Path(
                toolkit.process_image_with_models([model_name], row["original_path"], temporary_dir)
            ).resolve()
            if not generated.is_file():
                raise RuntimeError(f"{model_name} did not produce an image for {row['sample_id']}")
            shutil.copy2(generated, output_path)
        append_jsonl(
            restoration_progress_path(args),
            {
                "sample_id": row["sample_id"],
                "action": tool["name"],
                "output_path": str(output_path.resolve()),
                "seconds": time.perf_counter() - started,
                "status": "success",
            },
        )
        processed += 1
        if processed % 100 == 0 or processed == pending_count:
            print(f"[{tool['name']}] {processed}/{pending_count}", flush=True)
    del toolkit
    gc.collect()
    torch.cuda.empty_cache()


def restoration_worker(args: argparse.Namespace) -> int:
    if args.action is None:
        raise ValueError("--action is required in restore mode.")
    if (args.rank is None) != (args.world_size is None):
        raise ValueError("--rank and --world-size must be provided together in restore mode.")
    if args.world_size is not None and (args.world_size < 1 or not 0 <= args.rank < args.world_size):
        raise ValueError("Restore rank must satisfy 0 <= rank < world-size.")
    local_packages = args.external_tools_root / "python_packages"
    if local_packages.is_dir():
        sys.path.insert(0, str(local_packages))
    tools = {tool["name"]: tool for tool in load_tools(args.tools_config)}
    tool = tools[args.action]
    rows = read_jsonl(args.output_dir / "sample_manifest.jsonl")
    completed = load_restoration_completed(args) if args.resume else set()
    if len(completed) == len(rows):
        print(f"[{args.action}] already complete; skipping model load", flush=True)
        return 0
    if args.rank is not None and args.world_size is not None:
        rows = rows[args.rank :: args.world_size]
    completed = completed.intersection(row["sample_id"] for row in rows)
    if len(completed) == len(rows):
        print(f"[{args.action} rank {args.rank}] shard already complete; skipping model load", flush=True)
        return 0
    if tool["runtime"]["adapter"] == "candidate":
        run_candidate_batch(tool, rows, args, completed)
    else:
        run_toolkit_batch(tool, rows, args, completed)
    return 0


def collect_image_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.output_dir / "sample_manifest.jsonl")
    actions = [tool["name"] for tool in load_tools(args.tools_config)]
    records = []
    for row in rows:
        records.append(
            {
                "image_id": f"{row['sample_id']}::original",
                "sample_id": row["sample_id"],
                "degradation_type": row["degradation_type"],
                "action": "original",
                "image_path": row["original_path"],
            }
        )
        for action in actions:
            path = args.output_dir / "images/restored" / action / f"{row['sample_id']}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing restoration output: {path}")
            records.append(
                {
                    "image_id": f"{row['sample_id']}::{action}",
                    "sample_id": row["sample_id"],
                    "degradation_type": row["degradation_type"],
                    "action": action,
                    "image_path": str(path.resolve()),
                }
            )
    return records


def score_worker(args: argparse.Namespace) -> int:
    import packaging
    import packaging.version
    import torch

    if args.rank is None or args.world_size is None:
        raise ValueError("--rank and --world-size are required in score mode.")
    packaging.version = packaging.version
    sys.path.insert(0, str(IQA_REPO))
    pyiqa = importlib.import_module("pyiqa")
    records = collect_image_records(args)[args.rank :: args.world_size]
    progress_path = args.output_dir / "progress/iqa" / f"rank{args.rank}.jsonl"
    completed = (
        {row["image_id"] for path in sorted(progress_path.parent.glob("rank*.jsonl")) for row in read_jsonl(path)}
        if args.resume
        else set()
    )
    metrics = {name: pyiqa.create_metric(name, device="cuda:0") for name in METRICS}
    for index, record in enumerate(records, start=1):
        if record["image_id"] in completed:
            continue
        raw_scores = {}
        oriented_scores = {}
        metric_seconds = {}
        for name, metric in metrics.items():
            started = time.perf_counter()
            with torch.inference_mode():
                value = float(metric(record["image_path"]).reshape(-1)[0].item())
            torch.cuda.synchronize()
            raw_scores[name] = value
            oriented_scores[name] = -value if name == "niqe" else value
            metric_seconds[name] = time.perf_counter() - started
        append_jsonl(
            progress_path,
            {
                **record,
                "raw_scores": raw_scores,
                "oriented_scores": oriented_scores,
                "metric_seconds": metric_seconds,
            },
        )
        if index % 100 == 0 or index == len(records):
            print(f"[iqa rank {args.rank}] {index}/{len(records)}", flush=True)
    return 0


def cap_and_normalize(weights: np.ndarray, cap: float) -> np.ndarray:
    result = weights.astype(np.float64)
    for _ in range(10):
        over = result > cap
        if not over.any():
            break
        excess = float((result[over] - cap).sum())
        result[over] = cap
        under = ~over
        if not under.any():
            break
        available = result[under]
        if float(available.sum()) <= 1e-12:
            result[under] += excess / int(under.sum())
        else:
            result[under] += excess * available / float(available.sum())
    return result / result.sum()


def summarize(args: argparse.Namespace) -> None:
    score_rows = []
    for path in sorted((args.output_dir / "progress/iqa").glob("rank*.jsonl")):
        score_rows.extend(read_jsonl(path))
    expected = len(collect_image_records(args))
    if len(score_rows) != expected:
        raise RuntimeError(f"Expected {expected} IQA rows, found {len(score_rows)}.")
    score_rows.sort(key=lambda row: row["image_id"])

    values = {metric: np.array([row["oriented_scores"][metric] for row in score_rows]) for metric in METRICS}
    stats = {
        metric: {
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std(ddof=1)),
            "minimum": float(metric_values.min()),
            "maximum": float(metric_values.max()),
            "count": int(metric_values.size),
            "direction": "higher_is_better",
            "raw_transform": "-NIQE" if metric == "niqe" else "identity",
        }
        for metric, metric_values in values.items()
    }
    z_rows = []
    by_sample: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in score_rows:
        z_scores = {
            metric: (row["oriented_scores"][metric] - stats[metric]["mean"]) / stats[metric]["std"]
            for metric in METRICS
        }
        enriched = {**row, "z_scores": z_scores}
        z_rows.append(enriched)
        by_sample[row["sample_id"]][row["action"]] = z_scores

    grouped_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in z_rows:
        group = grouped_values[(row["degradation_type"], row["action"])]
        for metric in METRICS:
            group[metric].append(row["oriented_scores"][metric])
    distribution_rows = []
    for (label, action), group in sorted(grouped_values.items()):
        for metric in METRICS:
            metric_values = np.asarray(group[metric], dtype=np.float64)
            distribution_rows.append(
                {
                    "degradation_type": label,
                    "action": action,
                    "metric": metric,
                    "count": int(metric_values.size),
                    "mean": float(metric_values.mean()),
                    "std": float(metric_values.std(ddof=1)) if metric_values.size > 1 else 0.0,
                    "minimum": float(metric_values.min()),
                    "p05": float(np.quantile(metric_values, 0.05)),
                    "p25": float(np.quantile(metric_values, 0.25)),
                    "median": float(np.median(metric_values)),
                    "p75": float(np.quantile(metric_values, 0.75)),
                    "p95": float(np.quantile(metric_values, 0.95)),
                    "maximum": float(metric_values.max()),
                }
            )

    targeted_deltas: dict[str, list[float]] = defaultdict(list)
    other_deltas: dict[str, list[float]] = defaultdict(list)
    all_delta_vectors = []
    per_class: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        class_targeted: dict[str, list[float]] = defaultdict(list)
        class_other: dict[str, list[float]] = defaultdict(list)
        for sample_id, action_scores in by_sample.items():
            if not sample_id.startswith(f"{label}-"):
                continue
            original = action_scores["original"]
            for action, action_z in action_scores.items():
                if action == "original":
                    continue
                all_delta_vectors.append([action_z[metric] - original[metric] for metric in METRICS])
                destination = class_targeted if action in TARGETED_ACTIONS[label] else class_other
                global_destination = targeted_deltas if action in TARGETED_ACTIONS[label] else other_deltas
                for metric in METRICS:
                    delta = action_z[metric] - original[metric]
                    destination[metric].append(delta)
                    global_destination[metric].append(delta)
        per_class[label] = {
            metric: {
                "targeted_mean_delta_z": float(np.mean(class_targeted[metric])),
                "other_mean_delta_z": float(np.mean(class_other[metric])),
                "discriminability": float(np.mean(class_targeted[metric]) - np.mean(class_other[metric])),
            }
            for metric in METRICS
        }

    diagnostics = {}
    signals = []
    for metric in METRICS:
        targeted = np.asarray(targeted_deltas[metric], dtype=np.float64)
        other = np.asarray(other_deltas[metric], dtype=np.float64)
        targeted_mean = float(targeted.mean())
        other_mean = float(other.mean())
        discriminability = targeted_mean - other_mean
        positive_rate = float((targeted > 0).mean())
        signal = max(0.0, discriminability) + 0.25 * max(0.0, targeted_mean) * positive_rate
        diagnostics[metric] = {
            "targeted_mean_delta_z": targeted_mean,
            "other_mean_delta_z": other_mean,
            "discriminability": discriminability,
            "targeted_positive_rate": positive_rate,
            "raw_weight_signal": signal,
        }
        signals.append(signal)

    signal_array = np.asarray(signals, dtype=np.float64)
    discriminability_weight = signal_array / signal_array.sum() if signal_array.sum() > 1e-12 else np.full(4, 0.25)
    delta_matrix = np.asarray(all_delta_vectors, dtype=np.float64)
    delta_correlation = np.corrcoef(delta_matrix.T)
    consensus_correlations = []
    for metric_index in range(len(METRICS)):
        leave_one_out = np.delete(delta_matrix, metric_index, axis=1).mean(axis=1)
        correlation = float(np.corrcoef(delta_matrix[:, metric_index], leave_one_out)[0, 1])
        consensus_correlations.append(max(0.0, correlation))
        diagnostics[METRICS[metric_index]]["leave_one_out_consensus_correlation"] = correlation
    consensus_array = np.asarray(consensus_correlations, dtype=np.float64)
    consensus_weight = consensus_array / consensus_array.sum() if consensus_array.sum() > 1e-12 else np.full(4, 0.25)
    data_weight = 0.5 * discriminability_weight + 0.5 * consensus_weight
    shrinkage = 0.5
    prior_weight = np.full(4, 0.25)
    pre_cap = (1.0 - shrinkage) * data_weight + shrinkage * prior_weight
    final_weight = cap_and_normalize(pre_cap, cap=0.35)
    weight_payload = {
        "version": 1,
        "metrics": list(METRICS),
        "niqe_transform": "niqe_quality = -raw_niqe",
        "normalization": "zscore over balanced original and all single-step restoration outputs",
        "weight_method": (
            "50% targeted-vs-other discriminability + 50% leave-one-out metric consensus, "
            "then 50% uniform shrinkage and 0.35 cap"
        ),
        "data_signal_mix": {
            "targeted_discriminability": 0.5,
            "metric_consensus": 0.5,
        },
        "uniform_shrinkage": shrinkage,
        "max_weight": 0.35,
        "discriminability_weight": dict(zip(METRICS, map(float, discriminability_weight))),
        "consensus_weight": dict(zip(METRICS, map(float, consensus_weight))),
        "data_weight": dict(zip(METRICS, map(float, data_weight))),
        "pre_cap_weight": dict(zip(METRICS, map(float, pre_cap))),
        "weights": dict(zip(METRICS, map(float, final_weight))),
        "diagnostics": diagnostics,
        "delta_correlation_matrix": {
            row_metric: dict(zip(METRICS, map(float, delta_correlation[row_index])))
            for row_index, row_metric in enumerate(METRICS)
        },
        "per_class_diagnostics": per_class,
        "caveat": "Initial weakly supervised calibration; targeted tool families are semantic proxies, not human MOS labels.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "iqa_normalization_stats.json").write_text(
        json.dumps({"version": 1, "metrics": stats}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "iqa_reward_weights.json").write_text(json.dumps(weight_payload, indent=2), encoding="utf-8")
    with (args.output_dir / "iqa_scores.jsonl").open("w", encoding="utf-8") as file:
        for row in z_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "iqa_scores.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["sample_id", "degradation_type", "action", "image_path"]
        for metric in METRICS:
            fieldnames.extend([f"{metric}_raw", f"{metric}_oriented", f"{metric}_z"])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in z_rows:
            flat = {key: row[key] for key in ("sample_id", "degradation_type", "action", "image_path")}
            for metric in METRICS:
                flat[f"{metric}_raw"] = row["raw_scores"][metric]
                flat[f"{metric}_oriented"] = row["oriented_scores"][metric]
                flat[f"{metric}_z"] = row["z_scores"][metric]
            writer.writerow(flat)
    (args.output_dir / "iqa_distribution.json").write_text(json.dumps(distribution_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "iqa_distribution.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(distribution_rows[0]))
        writer.writeheader()
        writer.writerows(distribution_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, axis in zip(METRICS, axes.flat):
        original_values = [row["oriented_scores"][metric] for row in z_rows if row["action"] == "original"]
        restored_values = [row["oriented_scores"][metric] for row in z_rows if row["action"] != "original"]
        axis.hist(original_values, bins=16, alpha=0.7, density=True, label="original")
        axis.hist(restored_values, bins=32, alpha=0.55, density=True, label="restored")
        axis.set_title(metric)
        axis.set_xlabel("higher-is-better score")
        axis.set_ylabel("density")
        axis.legend()
    figure.suptitle("IQA score distributions on calibration images")
    figure.tight_layout()
    figure.savefig(args.output_dir / "iqa_score_distributions.png", dpi=180)
    plt.close(figure)

    actions = [tool["name"] for tool in load_tools(args.tools_config)]
    action_delta_matrix = []
    for action in actions:
        action_deltas = []
        for metric in METRICS:
            metric_deltas = []
            for action_scores in by_sample.values():
                metric_deltas.append(action_scores[action][metric] - action_scores["original"][metric])
            action_deltas.append(float(np.mean(metric_deltas)))
        action_delta_matrix.append(action_deltas)
    figure, axis = plt.subplots(figsize=(8, 10))
    image = axis.imshow(action_delta_matrix, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    axis.set_xticks(range(len(METRICS)), METRICS, rotation=30, ha="right")
    axis.set_yticks(range(len(actions)), actions)
    axis.set_title("Mean single-step IQA delta in z-score space")
    for row_index, row in enumerate(action_delta_matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="mean delta z")
    figure.tight_layout()
    figure.savefig(args.output_dir / "iqa_action_delta_heatmap.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Four-metric IQA reward calibration",
        "",
        f"- Original training images: {args.samples_per_class * len(LABELS)}",
        f"- Samples per class: {args.samples_per_class}",
        f"- Restoration actions: {len(load_tools(args.tools_config))}",
        f"- Total scored images: {len(score_rows)}",
        "- NIQE conversion: `niqe_quality = -raw_niqe`",
        "",
        "## Normalization",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = stats[metric]
        lines.append(
            f"| {metric} | {item['mean']:.6f} | {item['std']:.6f} | " f"{item['minimum']:.6f} | {item['maximum']:.6f} |"
        )
    lines.extend(["", "## Recommended weights", ""])
    for metric, weight in weight_payload["weights"].items():
        lines.append(f"- `{metric}`: {weight:.6f}")
    lines.extend(
        [
            "",
            "The aggregate quality score is `sum(weight_i * z_i)`. This is an initial",
            "weakly supervised calibration and should be audited against human pairwise",
            "preferences before the final GRPO run.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_process_pool(commands: list[tuple[str, list[str], dict[str, str]]], max_parallel: int) -> None:
    pending = list(commands)
    running: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    failures = []
    while pending or running:
        while pending and len(running) < max_parallel:
            busy_gpus = {gpu_id for _, gpu_id, _ in running}
            next_index = next(
                (
                    index
                    for index, (_, _, environment) in enumerate(pending)
                    if environment["CUDA_VISIBLE_DEVICES"] not in busy_gpus
                ),
                None,
            )
            if next_index is None:
                break
            name, command, environment = pending.pop(next_index)
            gpu_id = environment["CUDA_VISIBLE_DEVICES"]
            print(f"Launching {name}", flush=True)
            running.append((name, gpu_id, subprocess.Popen(command, env=environment)))
        time.sleep(1)
        still_running = []
        for name, gpu_id, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((name, gpu_id, process))
            elif return_code != 0:
                failures.append((name, return_code))
        running = still_running
        if failures:
            for _, _, process in running:
                process.terminate()
            raise RuntimeError(f"Worker failures: {failures}")


def coordinator(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.train_root = args.train_root.resolve()
    args.tools_config = args.tools_config.resolve()
    args.external_tools_root = args.external_tools_root.resolve()
    create_sample_manifest(args)
    tools = load_tools(args.tools_config)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU is required.")

    restoration_commands = []
    for index, tool in enumerate(tools):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        command = [
            str(args.python),
            str(SCRIPT_PATH),
            "--mode",
            "restore",
            "--action",
            tool["name"],
            "--output-dir",
            str(args.output_dir),
            "--tools-config",
            str(args.tools_config),
            "--external-tools-root",
            str(args.external_tools_root),
        ]
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["PYTHONUNBUFFERED"] = "1"
        restoration_commands.append((f"restoration:{tool['name']}@gpu{gpu_id}", command, environment))
    run_process_pool(restoration_commands, max_parallel=len(gpu_ids))

    score_commands = []
    for rank, gpu_id in enumerate(gpu_ids):
        command = [
            str(args.python),
            str(SCRIPT_PATH),
            "--mode",
            "score",
            "--rank",
            str(rank),
            "--world-size",
            str(len(gpu_ids)),
            "--output-dir",
            str(args.output_dir),
            "--tools-config",
            str(args.tools_config),
            "--external-tools-root",
            str(args.external_tools_root),
        ]
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["PYTHONUNBUFFERED"] = "1"
        score_commands.append((f"iqa:rank{rank}@gpu{gpu_id}", command, environment))
    run_process_pool(score_commands, max_parallel=len(gpu_ids))
    summarize(args)
    print(f"Calibration complete: {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tools_config = args.tools_config.expanduser().resolve()
    args.external_tools_root = args.external_tools_root.expanduser().resolve()
    if args.mode == "restore":
        return restoration_worker(args)
    if args.mode == "score":
        return score_worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
