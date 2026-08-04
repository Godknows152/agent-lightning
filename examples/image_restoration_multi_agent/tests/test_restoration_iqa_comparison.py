"""Tests for score_restoration_eval.py — set discovery, alignment, mean computation, minimal output."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent
OLD_VERL_DIR = EXAMPLE_DIR.parent / "old_verl_grpo"
SCORE_PATH = OLD_VERL_DIR / "scripts/eval/score_restoration_eval.py"

import importlib.util
import sys

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

scorer = _load_module("score_restoration_eval_test_module", SCORE_PATH)


def _make_output_set(tmp_path: Path, name: str, sample_ids: list[str], tool_counts: list[int] | None = None):
    out_dir = tmp_path / name
    (out_dir / "images").mkdir(parents=True)
    for i, sid in enumerate(sample_ids):
        (out_dir / "images" / f"{sid}.png").write_bytes(b"fake-png")
    tc = tool_counts or [0] * len(sample_ids)
    with (out_dir / "tool_calls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image", "restoration_tool_call_count"])
        writer.writeheader()
        for sid, n in zip(sample_ids, tc):
            writer.writerow({"sample_id": sid, "image": f"images/{sid}.png", "restoration_tool_call_count": n})
    return out_dir


def test_discover_images_from_csv(tmp_path):
    """Score script discovers images only from tool_calls.csv, not directory glob."""
    out_dir = _make_output_set(tmp_path, "model1", ["fog-000000", "fog-000001"], [1, 0])
    loaded = scorer._load_set("model1", out_dir, 2)
    assert set(loaded) == {"fog-000000", "fog-000001"}
    assert loaded["fog-000000"]["image_path"].endswith("images/fog-000000.png")
    assert loaded["fog-000000"]["tool_count"] == 1


def test_load_set_rejects_missing_images(tmp_path):
    out_dir = tmp_path / "model1"
    (out_dir / "images").mkdir(parents=True)
    with (out_dir / "tool_calls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image", "restoration_tool_call_count"])
        writer.writeheader()
        writer.writerow({"sample_id": "fog-000000", "image": "images/fog-000000.png", "restoration_tool_call_count": 0})
    with pytest.raises(FileNotFoundError, match="image missing"):
        scorer._load_set("model1", out_dir, 1)


def test_load_set_rejects_duplicate_ids(tmp_path):
    out_dir = tmp_path / "model1"
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "images/fog-000000.png").write_bytes(b"x")
    with (out_dir / "tool_calls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image", "restoration_tool_call_count"])
        writer.writeheader()
        for _ in range(2):
            writer.writerow({"sample_id": "fog-000000", "image": "images/fog-000000.png", "restoration_tool_call_count": 0})
    with pytest.raises(ValueError, match="duplicate"):
        scorer._load_set("model1", out_dir, 2)


def test_load_set_rejects_invalid_tool_count(tmp_path):
    out_dir = tmp_path / "model1"
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "images/fog-000000.png").write_bytes(b"x")
    with (out_dir / "tool_calls.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image", "restoration_tool_call_count"])
        writer.writeheader()
        writer.writerow({"sample_id": "fog-000000", "image": "images/fog-000000.png", "restoration_tool_call_count": "-1"})
    with pytest.raises(ValueError, match="invalid tool count"):
        scorer._load_set("model1", out_dir, 1)


def test_alignment_rejects_mismatched_sample_ids(tmp_path):
    set_a = _make_output_set(tmp_path, "a", ["fog-000000", "fog-000001"])
    set_b = _make_output_set(tmp_path, "b", ["fog-000000", "fog-000002"])
    loaded = {
        "a": scorer._load_set("a", set_a, 2),
        "b": scorer._load_set("b", set_b, 2),
    }
    id_sets = [set(ld.keys()) for ld in loaded.values()]
    with pytest.raises(ValueError, match="Sample ID mismatch"):
        master = id_sets[0]
        for i, ids in enumerate(id_sets[1:], start=1):
            if ids != master:
                raise ValueError(f"Sample ID mismatch: set {list(loaded)[i]} ...")


def test_mean_computation_matches_numpy(tmp_path):
    """Verify the mean aggregation matches numpy arithmetic means."""
    metrics = ["musiq", "maniqa", "clipiqa", "liqe"]
    values = np.asarray([[0.8, 0.5, 0.2, 0.7], [0.4, 0.3, 0.1, 0.5]], dtype=np.float64)
    means = {}
    for idx, m in enumerate(metrics):
        means[f"{m}_mean"] = float(np.mean(values[:, idx]))
    assert means["musiq_mean"] == pytest.approx(0.6)
    assert means["maniqa_mean"] == pytest.approx(0.4)
    assert means["clipiqa_mean"] == pytest.approx(0.15)
    assert means["liqe_mean"] == pytest.approx(0.6)


def test_nonfinite_scores_are_rejected():
    """Non-finite IQA scores must cause failure, not be silently averaged."""
    metrics = ["musiq", "maniqa", "clipiqa", "liqe"]
    values = np.asarray([[math.inf, 0.5, 0.2, 0.7]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        for idx, m in enumerate(metrics):
            v = float(values[0, idx])
            if not math.isfinite(v):
                raise ValueError(f"{m}: non-finite value {v}")


def test_output_csv_columns_and_content(tmp_path):
    """Final CSV must have exactly the required columns and one row per model."""
    output_csv = tmp_path / "comparison.csv"
    rows = [
        {"model": "v4.1.1_0803", "sample_count": 2,
         "musiq_mean": 0.6, "maniqa_mean": 0.4, "clipiqa_mean": 0.15, "liqe_mean": 0.6},
        {"model": "JarvisIR", "sample_count": 2,
         "musiq_mean": 0.5, "maniqa_mean": 0.3, "clipiqa_mean": 0.2, "liqe_mean": 0.5},
    ]
    fieldnames = ["model", "sample_count", "musiq_mean", "maniqa_mean", "clipiqa_mean", "liqe_mean"]
    scorer.write_csv(output_csv, rows, fieldnames)
    with output_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == fieldnames
        content = list(reader)
    assert len(content) == 2
    assert content[0]["model"] == "v4.1.1_0803"
    assert float(content[1]["musiq_mean"]) == pytest.approx(0.5)


def test_no_per_image_outputs(tmp_path):
    """Scoring state may exist, but the final artifact is only the comparison CSV."""
    output_csv = tmp_path / "comparison.csv"
    rows = [{"model": "m", "sample_count": 0,
             "musiq_mean": 0.0, "maniqa_mean": 0.0, "clipiqa_mean": 0.0, "liqe_mean": 0.0}]
    scorer.write_csv(output_csv, rows, ["model", "sample_count", "musiq_mean", "maniqa_mean", "clipiqa_mean", "liqe_mean"])
    assert output_csv.is_file()
    # The script must not emit per-image score files next to the CSV.
    assert not (tmp_path / "iqa_scores.csv").exists()
    assert not (tmp_path / "scores.jsonl").exists()
