#!/usr/bin/env python3
"""Directly concatenate the four expert RL parquet datasets.

The source rows are not rebuilt or rewritten. In particular, each row keeps
its original ``extra_info.expert`` and ``extra_info.expert_name`` fields so the
multi-turn runtime can construct the correct degradation-specific prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
OLD_VERL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATA_DIR = OLD_VERL_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "unified"
EXPERTS = ("fog", "rain", "snow", "low_light")
SPLITS = ("train", "val")
BALANCED_VAL_PER_EXPERT = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expert_counts(table: pa.Table) -> dict[str, int]:
    extra_info = table.column("extra_info").combine_chunks()
    experts = extra_info.field("expert").to_pylist()
    counts = {expert: 0 for expert in EXPERTS}
    for expert in experts:
        if expert not in counts:
            raise ValueError(f"Unexpected expert label in source data: {expert!r}")
        counts[expert] += 1
    return counts


def merge_split(data_dir: Path, output_dir: Path, split: str) -> dict[str, Any]:
    source_paths = [data_dir / f"{expert}_{split}.parquet" for expert in EXPERTS]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source parquet files: " + ", ".join(missing))

    tables = [pq.read_table(path) for path in source_paths]
    reference_schema = tables[0].schema
    for path, table in zip(source_paths[1:], tables[1:], strict=True):
        if not table.schema.equals(reference_schema, check_metadata=False):
            raise ValueError(f"Schema mismatch: {path}")

    merged = pa.concat_tables(tables, promote_options="none")
    counts = _expert_counts(merged)
    expected_counts = {
        expert: table.num_rows
        for expert, table in zip(EXPERTS, tables, strict=True)
    }
    if counts != expected_counts:
        raise ValueError(f"Merged expert counts differ from sources: {counts} != {expected_counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"unified_{split}.parquet"
    pq.write_table(merged, output_path, compression="snappy")
    written = pq.ParquetFile(output_path)
    if written.metadata.num_rows != merged.num_rows:
        raise ValueError(
            f"Written row count mismatch: {written.metadata.num_rows} != {merged.num_rows}"
        )

    return {
        "split": split,
        "output": str(output_path.resolve()),
        "num_rows": merged.num_rows,
        "expert_counts": counts,
        "sha256": _sha256(output_path),
        "sources": [
            {
                "expert": expert,
                "path": str(path.resolve()),
                "num_rows": table.num_rows,
                "sha256": _sha256(path),
            }
            for expert, path, table in zip(EXPERTS, source_paths, tables, strict=True)
        ],
    }


def write_balanced_validation(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write a deterministic 64-row validation view with 16 rows per expert."""

    tables = [
        pq.read_table(data_dir / f"{expert}_val.parquet").slice(
            0, BALANCED_VAL_PER_EXPERT
        )
        for expert in EXPERTS
    ]
    # Round-robin ordering keeps every contiguous group of four expert-balanced.
    batches: list[pa.Table] = []
    for index in range(BALANCED_VAL_PER_EXPERT):
        batches.extend(table.slice(index, 1) for table in tables)
    balanced = pa.concat_tables(batches, promote_options="none")
    output_path = output_dir / "unified_val_balanced64.parquet"
    pq.write_table(balanced, output_path, compression="snappy")
    counts = _expert_counts(balanced)
    expected = {expert: BALANCED_VAL_PER_EXPERT for expert in EXPERTS}
    if counts != expected:
        raise ValueError(f"Balanced validation counts are invalid: {counts}")
    return {
        "split": "val_balanced64",
        "output": str(output_path.resolve()),
        "num_rows": balanced.num_rows,
        "expert_counts": counts,
        "sha256": _sha256(output_path),
        "source": str((output_dir / "unified_val.parquet").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    results = [merge_split(data_dir, output_dir, split) for split in SPLITS]
    balanced_validation = write_balanced_validation(data_dir, output_dir)
    manifest = {
        "purpose": "unified_four_expert_old_verl_grpo",
        "merge_strategy": "direct_concatenation_without_shuffle",
        "expert_order": list(EXPERTS),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {result["split"]: result for result in results},
        "validation_view": balanced_validation,
    }
    manifest_path = output_dir / "unified_rl_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result['split']}: {result['num_rows']} rows, "
            f"counts={result['expert_counts']}, output={result['output']}"
        )
    print(
        f"val_balanced64: {balanced_validation['num_rows']} rows, "
        f"counts={balanced_validation['expert_counts']}, "
        f"output={balanced_validation['output']}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
