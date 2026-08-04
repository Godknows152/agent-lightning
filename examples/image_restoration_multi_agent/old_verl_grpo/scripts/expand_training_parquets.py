#!/usr/bin/env python3
"""Append deterministic, deduplicated source images to expert train parquets.

The original rows are copied without rebuilding their prompts. New rows clone
the corresponding parquet's row structure and replace only sample-specific
fields and embedded image bytes.

Run from the Agent Lightning repository root with the ``verl`` environment:

    /home/LXJ/anaconda3/envs/verl/bin/python \
      examples/image_restoration_multi_agent/old_verl_grpo/scripts/expand_training_parquets.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_DIRECTORIES = {
    "fog": "fog_series/fog",
    "snow": "snow_series/snow",
    "rain": "rain_series/rain",
    "low_light": "night_series/night",
}
DEFAULT_SOURCE_ROOT = Path(
    "/home/LXJ/Python_Projects/AIA_Restore\u65e7\u6570\u636e\u5b58\u653e/OpenReal_80k/train/images"
)
SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "examples/image_restoration_multi_agent/old_verl_grpo/data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "\u6269\u5145\u540e"
DEFAULT_ROUTER_DIR = REPOSITORY_ROOT / "examples/image_restoration_multi_agent/adaptive_budget/data/router_unseen_v1"


@dataclass(frozen=True)
class HashItem:
    value: int
    path: str


@dataclass
class BKNode:
    item: HashItem
    children: dict[int, "BKNode"] = field(default_factory=cast(Callable[[], dict[int, "BKNode"]], dict))


class HammingBKTree:
    """Index 64-bit perceptual hashes for bounded-distance lookup."""

    def __init__(self) -> None:
        self._root: BKNode | None = None
        self._values: set[int] = set()

    def add(self, item: HashItem) -> None:
        if item.value in self._values:
            return
        self._values.add(item.value)
        if self._root is None:
            self._root = BKNode(item=item)
            return
        node = self._root
        while True:
            distance = (item.value ^ node.item.value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(item=item)
                return
            node = child

    def find_within(self, value: int, threshold: int) -> tuple[HashItem, int] | None:
        if self._root is None:
            return None
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.item.value).bit_count()
            if distance <= threshold:
                return node.item, distance
            lower = distance - threshold
            upper = distance + threshold
            pending.extend(child for edge_distance, child in node.children.items() if lower <= edge_distance <= upper)
        return None


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    dhash64: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--router-dir", type=Path, default=DEFAULT_ROUTER_DIR)
    parser.add_argument("--additions-per-expert", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolved_file(path: Path, *, context: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{context} is not a file: {resolved}")
    return resolved


def _images(directory: Path) -> list[Path]:
    return sorted(
        path.resolve() for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _dhash64(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = grayscale.tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _fingerprint(path: Path) -> Fingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        image.load()
        dhash = _dhash64(image)
    return Fingerprint(sha256=digest.hexdigest(), dhash64=dhash)


def _image_bytes(path: Path) -> dict[str, bytes]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    with image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            yield payload


def _load_router_exclusions(router_dir: Path) -> tuple[set[Path], list[tuple[Path, Fingerprint]]]:
    paths: set[Path] = set()
    fingerprints: list[tuple[Path, Fingerprint]] = []
    for expert in CLASS_DIRECTORIES:
        manifest = router_dir / f"{expert}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"router exclusion manifest does not exist: {manifest}")
        for line_number, row in enumerate(_iter_jsonl(manifest), start=1):
            image_path = _resolved_file(
                Path(str(row.get("image_path", ""))),
                context=f"{manifest}:{line_number} image_path",
            )
            sha256 = row.get("content_sha256")
            dhash64 = row.get("dhash64")
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError(f"{manifest}:{line_number} has an invalid content_sha256")
            if not isinstance(dhash64, str) or len(dhash64) != 16:
                raise ValueError(f"{manifest}:{line_number} has an invalid dhash64")
            paths.add(image_path)
            fingerprints.append((image_path, Fingerprint(sha256=sha256, dhash64=int(dhash64, 16))))
    return paths, fingerprints


def _old_paths(table: pa.Table, *, parquet_path: Path, expert: str) -> list[Path]:
    paths: list[Path] = []
    for index, row in enumerate(table.column("extra_info").to_pylist()):
        if row.get("expert") != expert or row.get("degradation_type") != expert:
            raise ValueError(f"{parquet_path}:{index} does not belong to expert {expert}")
        paths.append(
            _resolved_file(
                Path(str(row.get("image_path", ""))),
                context=f"{parquet_path}:{index} image_path",
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError(f"{parquet_path} contains duplicate image paths")
    return paths


def _new_row(template: dict[str, Any], *, expert: str, index: int, image_path: Path) -> dict[str, Any]:
    row = copy.deepcopy(template)
    path_text = str(image_path)
    row["images"] = [_image_bytes(image_path)]
    row["reward_model"]["ground_truth"]["image_path"] = path_text
    row["reward_model"]["ground_truth"]["degradation_type"] = expert
    extra_info = row["extra_info"]
    extra_info["index"] = index
    extra_info["sample_id"] = f"{expert}-{index:06d}"
    extra_info["image_path"] = path_text
    extra_info["degradation_type"] = expert
    create_kwargs = extra_info["tools_kwargs"]["restore_image"]["create_kwargs"]
    create_kwargs["image_path"] = path_text
    create_kwargs["degradation_type"] = expert
    return row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_path.replace(path)


def main() -> int:
    args = _parse_args()
    if args.additions_per_expert <= 0:
        raise ValueError("additions-per-expert must be positive")
    if not 0 <= args.dhash_threshold <= 64:
        raise ValueError("dhash-threshold must be between 0 and 64")

    source_root = args.source_root.expanduser().resolve(strict=True)
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    router_dir = args.router_dir.expanduser().resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / f"{expert}_train.parquet" for expert in CLASS_DIRECTORIES]
    audit_paths = [output_dir / "selected_additions.jsonl", output_dir / "expansion_manifest.json"]
    existing_outputs = [path for path in (*output_paths, *audit_paths) if path.exists()]
    if existing_outputs and not args.overwrite:
        joined = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"output files already exist; pass --overwrite to replace them: {joined}")

    router_paths, router_fingerprints = _load_router_exclusions(router_dir)
    excluded_paths = set(router_paths)
    exact_hashes = {item.sha256 for _, item in router_fingerprints}
    dhash_tree = HammingBKTree()
    for path, item in router_fingerprints:
        dhash_tree.add(HashItem(value=item.dhash64, path=str(path)))

    source_images: dict[str, list[Path]] = {}
    input_tables: dict[str, pa.Table] = {}
    input_paths: dict[str, Path] = {}
    exclusion_row_paths: dict[str, list[Path]] = {}
    for expert, relative_directory in CLASS_DIRECTORIES.items():
        parquet_path = _resolved_file(input_dir / f"{expert}_train.parquet", context=f"{expert} input parquet")
        table = pq.read_table(parquet_path)
        paths = _old_paths(table, parquet_path=parquet_path, expert=expert)
        validation_path = _resolved_file(input_dir / f"{expert}_val.parquet", context=f"{expert} validation parquet")
        validation_table = pq.read_table(validation_path, columns=["extra_info"])
        validation_paths = _old_paths(validation_table, parquet_path=validation_path, expert=expert)
        input_tables[expert] = table
        input_paths[expert] = parquet_path
        exclusion_row_paths[expert] = [*paths, *validation_paths]
        excluded_paths.update(exclusion_row_paths[expert])
        source_images[expert] = _images(source_root / relative_directory)

    for expert in CLASS_DIRECTORIES:
        for image_path in exclusion_row_paths[expert]:
            item = _fingerprint(image_path)
            exact_hashes.add(item.sha256)
            dhash_tree.add(HashItem(value=item.dhash64, path=str(image_path)))

    selected: dict[str, list[tuple[Path, Fingerprint]]] = {}
    expert_audits: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for class_index, expert in enumerate(CLASS_DIRECTORIES):
        candidates = [path for path in source_images[expert] if path not in excluded_paths]
        generator = random.Random(args.seed + class_index)
        generator.shuffle(candidates)
        accepted: list[tuple[Path, Fingerprint]] = []
        rejections = {"exact_content": 0, "near_duplicate": 0}
        scanned = 0
        for image_path in candidates:
            scanned += 1
            item = _fingerprint(image_path)
            if item.sha256 in exact_hashes:
                rejections["exact_content"] += 1
                continue
            near_match = dhash_tree.find_within(item.dhash64, args.dhash_threshold)
            if near_match is not None:
                rejections["near_duplicate"] += 1
                continue
            accepted.append((image_path, item))
            excluded_paths.add(image_path)
            exact_hashes.add(item.sha256)
            dhash_tree.add(HashItem(value=item.dhash64, path=str(image_path)))
            if len(accepted) == args.additions_per_expert:
                break
        if len(accepted) != args.additions_per_expert:
            raise ValueError(
                f"not enough deduplicated {expert} candidates: selected {len(accepted)} of "
                f"{args.additions_per_expert} after scanning {scanned}"
            )
        selected[expert] = accepted
        original_count = input_tables[expert].num_rows
        for offset, (image_path, item) in enumerate(accepted):
            row_index = original_count + offset
            selected_rows.append(
                {
                    "expert": expert,
                    "index": row_index,
                    "sample_id": f"{expert}-{row_index:06d}",
                    "image_path": str(image_path),
                    "source_relative_path": str(image_path.relative_to(source_root)),
                    "content_sha256": item.sha256,
                    "dhash64": f"{item.dhash64:016x}",
                }
            )
        expert_audits[expert] = {
            "input_parquet": str(input_paths[expert]),
            "output_parquet": str(output_dir / f"{expert}_train.parquet"),
            "source_directory": str(source_root / CLASS_DIRECTORIES[expert]),
            "source_image_count": len(source_images[expert]),
            "original_row_count": original_count,
            "validation_exclusion_count": len(exclusion_row_paths[expert]) - original_count,
            "candidate_count_after_path_exclusion": len(candidates),
            "candidates_scanned": scanned,
            "rejections": rejections,
            "added_row_count": len(accepted),
            "expanded_row_count": original_count + len(accepted),
        }

    for expert in CLASS_DIRECTORIES:
        table = input_tables[expert]
        template = table.slice(0, 1).to_pylist()[0]
        new_rows = [
            _new_row(
                template,
                expert=expert,
                index=table.num_rows + offset,
                image_path=image_path,
            )
            for offset, (image_path, _) in enumerate(selected[expert])
        ]
        additions = pa.Table.from_pylist(new_rows, schema=table.schema)
        expanded = pa.concat_tables([table, additions])
        output_path = output_dir / f"{expert}_train.parquet"
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(expanded, temporary_path, compression="snappy")
        temporary_path.replace(output_path)
        print(f"{expert}: {table.num_rows} + {additions.num_rows} -> {expanded.num_rows} rows")

    _write_jsonl(output_dir / "selected_additions.jsonl", selected_rows)
    manifest = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_seed": args.seed,
        "additions_per_expert": args.additions_per_expert,
        "dhash_algorithm": "64-bit difference hash, grayscale 9x8, LANCZOS",
        "dhash_rejection_threshold": args.dhash_threshold,
        "source_root": str(source_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "router_exclusion_dir": str(router_dir),
        "router_excluded_path_count": len(router_paths),
        "expert_order": list(CLASS_DIRECTORIES),
        "experts": expert_audits,
    }
    manifest_path = output_dir / "expansion_manifest.json"
    temporary_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest_path.replace(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
