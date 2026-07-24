#!/usr/bin/env python3
"""Select deterministic expert-unseen images for difficulty-router development.

The script keeps the image files in their original OpenReal_80k location and
writes only JSONL manifests plus an audit report. It excludes the current
four-expert SFT and GRPO train/validation images by canonical path, SHA256
content, and a conservative 64-bit difference hash.

Run from the Agent Lightning repository root with an environment that contains
Pillow and PyArrow:

    conda run -n base python \
      examples/image_restoration_multi_agent/scripts/select_unseen_router_images.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, cast

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_DIRECTORIES = {
    "fog": "fog_series/fog",
    "snow": "snow_series/snow",
    "rain": "rain_series/rain",
    "low_light": "night_series/night",
}

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SOURCE_ROOT = Path(
    "/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k/train/images"
)
DEFAULT_GRPO_DATA_DIR = (
    REPOSITORY_ROOT / "examples/image_restoration_multi_agent/grpo/data"
)
DEFAULT_OLD_VERL_DATA_DIR = (
    REPOSITORY_ROOT / "examples/image_restoration_multi_agent/old_verl_grpo/data"
)
DEFAULT_SFT_DATA_DIR = (
    REPOSITORY_ROOT / "LlamaFactory/image_restoration_experts/data"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "examples/image_restoration_multi_agent/adaptive_budget/data/router_unseen_v1"
)


@dataclass
class ExclusionEntry:
    """One canonical image path and every training source that references it."""

    path: Path
    categories: set[str] = field(default_factory=cast(Callable[[], set[str]], set))
    sources: set[str] = field(default_factory=cast(Callable[[], set[str]], set))


@dataclass(frozen=True)
class HashItem:
    """Perceptual hash item stored in a BK-tree."""

    value: int
    path: str


@dataclass
class BKNode:
    """One node in a Hamming-distance BK-tree."""

    item: HashItem
    children: dict[int, "BKNode"] = field(
        default_factory=cast(Callable[[], dict[int, "BKNode"]], dict)
    )


class HammingBKTree:
    """Index 64-bit hashes for conservative near-duplicate lookup."""

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
            distance = _hamming_distance(item.value, node.item.value)
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
            distance = _hamming_distance(value, node.item.value)
            if distance <= threshold:
                return node.item, distance
            lower = distance - threshold
            upper = distance + threshold
            pending.extend(
                child
                for edge_distance, child in node.children.items()
                if lower <= edge_distance <= upper
            )
        return None


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_index, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload: object = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_index} must contain a JSON object")
            yield cast(dict[str, Any], payload)


def _required_path(value: object, *, context: str, base_dir: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must contain a non-empty image path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"{context} contains an unresolved relative path: {value}")
        path = base_dir / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} image does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{context} image is not a file: {resolved}")
    return resolved


def _category_and_split(file_name: str) -> tuple[str, str]:
    for split in ("train", "val"):
        suffix = f"_{split}.jsonl"
        if file_name.endswith(suffix):
            category = file_name.removesuffix(suffix)
            if category not in CLASS_DIRECTORIES:
                raise ValueError(f"unsupported GRPO category in {file_name}: {category}")
            return category, split
    raise ValueError(f"unsupported GRPO manifest name: {file_name}")


def _sft_category(file_name: str) -> str:
    suffix = "_expert_rl_aligned_train.jsonl"
    if not file_name.endswith(suffix):
        raise ValueError(f"unsupported SFT manifest name: {file_name}")
    category = file_name.removesuffix(suffix)
    if category not in CLASS_DIRECTORIES:
        raise ValueError(f"unsupported SFT category in {file_name}: {category}")
    return category


def _register_exclusion(
    entries: dict[Path, ExclusionEntry],
    *,
    path: Path,
    category: str,
    source: str,
) -> None:
    entry = entries.setdefault(path, ExclusionEntry(path=path))
    entry.categories.add(category)
    entry.sources.add(source)


def _load_grpo_exclusions(data_dir: Path) -> tuple[dict[Path, ExclusionEntry], dict[str, set[Path]]]:
    entries: dict[Path, ExclusionEntry] = {}
    path_sets: dict[str, set[Path]] = {}
    manifests = sorted((*data_dir.glob("*_train.jsonl"), *data_dir.glob("*_val.jsonl")))
    if len(manifests) != 8:
        raise ValueError(f"expected eight GRPO train/val manifests in {data_dir}, found {len(manifests)}")
    for manifest_path in manifests:
        category, split = _category_and_split(manifest_path.name)
        key = f"{category}_{split}"
        paths: set[Path] = set()
        for line_index, record in enumerate(_jsonl_records(manifest_path), start=1):
            record_category = record.get("degradation_type")
            if record_category != category:
                raise ValueError(
                    f"{manifest_path}:{line_index} category={record_category!r}, expected {category!r}"
                )
            image_path = _required_path(
                record.get("image_path"),
                context=f"{manifest_path}:{line_index}",
            )
            paths.add(image_path)
            _register_exclusion(
                entries,
                path=image_path,
                category=category,
                source=f"grpo_{split}:{manifest_path.relative_to(REPOSITORY_ROOT)}",
            )
        path_sets[key] = paths
    return entries, path_sets


def _load_sft_exclusions(data_dir: Path) -> dict[Path, ExclusionEntry]:
    entries: dict[Path, ExclusionEntry] = {}
    manifests = sorted(data_dir.glob("*_expert_rl_aligned_train.jsonl"))
    if len(manifests) != 4:
        raise ValueError(f"expected four expert SFT manifests in {data_dir}, found {len(manifests)}")
    for manifest_path in manifests:
        category = _sft_category(manifest_path.name)
        for line_index, record in enumerate(_jsonl_records(manifest_path), start=1):
            images_value: object = record.get("images")
            images = cast(list[object], images_value) if isinstance(images_value, list) else []
            if len(images) != 1:
                raise ValueError(f"{manifest_path}:{line_index} must reference exactly one image")
            image_path = _required_path(
                images[0],
                context=f"{manifest_path}:{line_index}",
                base_dir=data_dir,
            )
            _register_exclusion(
                entries,
                path=image_path,
                category=category,
                source=f"sft_train:{manifest_path.relative_to(REPOSITORY_ROOT)}",
            )
    return entries


def _merge_exclusions(
    destination: dict[Path, ExclusionEntry],
    additions: dict[Path, ExclusionEntry],
) -> None:
    for path, addition in additions.items():
        entry = destination.setdefault(path, ExclusionEntry(path=path))
        entry.categories.update(addition.categories)
        entry.sources.update(addition.sources)


def _path_set_digest(paths: Iterable[Path]) -> str:
    payload = "\n".join(sorted(str(path) for path in paths)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cross_check_old_verl_parquet(
    parquet_dir: Path,
    grpo_path_sets: dict[str, set[Path]],
) -> dict[str, dict[str, Any]]:
    try:
        parquet_module = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyArrow is required for the old-verl parquet cross-check; "
            "run this script in the base or verl conda environment"
        ) from error

    report: dict[str, dict[str, Any]] = {}
    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    if len(parquet_paths) != 8:
        raise ValueError(f"expected eight old-verl parquet files in {parquet_dir}, found {len(parquet_paths)}")
    for parquet_path in parquet_paths:
        stem = parquet_path.stem
        expected_paths = grpo_path_sets.get(stem)
        if expected_paths is None:
            raise ValueError(f"unexpected old-verl parquet file: {parquet_path}")
        parquet_api = cast(Any, parquet_module)
        table = parquet_api.read_table(parquet_path, columns=["extra_info"])
        actual_paths: set[Path] = set()
        rows = cast(list[object], table.to_pylist())
        for row_index, row_value in enumerate(rows, start=1):
            if not isinstance(row_value, dict):
                raise ValueError(f"{parquet_path}:{row_index} must contain an object")
            row = cast(dict[str, object], row_value)
            extra_info_value: object = row.get("extra_info")
            if not isinstance(extra_info_value, dict):
                raise ValueError(f"{parquet_path}:{row_index} is missing extra_info")
            extra_info = cast(dict[str, object], extra_info_value)
            actual_paths.add(
                _required_path(
                    extra_info.get("image_path"),
                    context=f"{parquet_path}:{row_index}",
                )
            )
        if actual_paths != expected_paths:
            missing = sorted(str(path) for path in expected_paths - actual_paths)
            extra = sorted(str(path) for path in actual_paths - expected_paths)
            raise ValueError(
                f"{parquet_path} does not match its GRPO JSONL source; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        report[stem] = {
            "path": str(parquet_path),
            "rows": table.num_rows,
            "unique_image_paths": len(actual_paths),
            "path_set_sha256": _path_set_digest(actual_paths),
            "matches_grpo_jsonl": True,
        }
    return report


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash64(path: Path) -> int:
    try:
        image_module = importlib.import_module("PIL.Image")
        image_ops_module = importlib.import_module("PIL.ImageOps")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Pillow is required for perceptual near-duplicate filtering; "
            "run this script in the base or verl conda environment"
        ) from error

    image_api = cast(Any, image_module)
    image_ops = cast(Any, image_ops_module)
    with image_api.open(path) as opened_image:
        normalized = image_ops.exif_transpose(opened_image).convert("L")
        resampling = getattr(image_api, "Resampling", image_api)
        resized = normalized.resize((9, 8), resampling.LANCZOS)
        pixels = list(resized.getdata())
    value = 0
    for row_index in range(8):
        offset = row_index * 9
        for column_index in range(8):
            value = (value << 1) | int(
                pixels[offset + column_index] > pixels[offset + column_index + 1]
            )
    return value


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"source image directory does not exist: {directory}")
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _class_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _file_digest(path: Path) -> str:
    return _sha256(path)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return _file_digest(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _file_digest(path)


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _build_exclusion_inventory(
    exclusions: dict[Path, ExclusionEntry],
) -> tuple[
    list[dict[str, Any]],
    set[str],
    dict[str, str],
    HammingBKTree,
]:
    records: list[dict[str, Any]] = []
    sha256_values: set[str] = set()
    representative_by_sha256: dict[str, str] = {}
    tree = HammingBKTree()
    total = len(exclusions)
    for index, entry in enumerate(sorted(exclusions.values(), key=lambda item: str(item.path)), start=1):
        content_sha256 = _sha256(entry.path)
        dhash_value = _dhash64(entry.path)
        path_text = str(entry.path)
        sha256_values.add(content_sha256)
        representative_by_sha256.setdefault(content_sha256, path_text)
        tree.add(HashItem(value=dhash_value, path=path_text))
        records.append(
            {
                "image_path": path_text,
                "categories": sorted(entry.categories),
                "sources": sorted(entry.sources),
                "content_sha256": content_sha256,
                "dhash64": f"{dhash_value:016x}",
            }
        )
        if index % 500 == 0 or index == total:
            print(f"exclusions: hashed {index}/{total}", flush=True)
    return records, sha256_values, representative_by_sha256, tree


def _select_category(
    *,
    category: str,
    source_root: Path,
    source_directory: Path,
    samples_per_class: int,
    seed: int,
    exclusions: dict[Path, ExclusionEntry],
    excluded_sha256: set[str],
    excluded_representative_by_sha256: dict[str, str],
    excluded_dhash_tree: HammingBKTree,
    selected_sha256: set[str],
    selected_dhash_tree: HammingBKTree,
    near_duplicate_threshold: int,
    inventory_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_images = _images(source_directory)
    source_path_exclusions = sum(image_path in exclusions for image_path in source_images)
    shuffled_images = list(source_images)
    class_seed = _class_seed(seed, category)
    random.Random(class_seed).shuffle(shuffled_images)

    selected: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    scanned_non_path_candidates = 0

    for image_path in shuffled_images:
        exclusion_entry = exclusions.get(image_path)
        if exclusion_entry is not None:
            rejection_counts["canonical_path"] += 1
            rejections.append(
                {
                    "degradation_type": category,
                    "image_path": str(image_path),
                    "reason": "canonical_path",
                    "matched_path": str(image_path),
                    "matched_sources": sorted(exclusion_entry.sources),
                }
            )
            continue

        scanned_non_path_candidates += 1
        content_sha256 = _sha256(image_path)
        if content_sha256 in excluded_sha256:
            rejection_counts["excluded_exact_content"] += 1
            rejections.append(
                {
                    "degradation_type": category,
                    "image_path": str(image_path),
                    "reason": "excluded_exact_content",
                    "matched_path": excluded_representative_by_sha256[content_sha256],
                    "content_sha256": content_sha256,
                }
            )
            continue
        if content_sha256 in selected_sha256:
            rejection_counts["selected_exact_content"] += 1
            rejections.append(
                {
                    "degradation_type": category,
                    "image_path": str(image_path),
                    "reason": "selected_exact_content",
                    "content_sha256": content_sha256,
                }
            )
            continue

        dhash_value = _dhash64(image_path)
        excluded_match = excluded_dhash_tree.find_within(dhash_value, near_duplicate_threshold)
        if excluded_match is not None:
            matched_item, distance = excluded_match
            rejection_counts["excluded_near_duplicate"] += 1
            rejections.append(
                {
                    "degradation_type": category,
                    "image_path": str(image_path),
                    "reason": "excluded_near_duplicate",
                    "matched_path": matched_item.path,
                    "dhash_distance": distance,
                }
            )
            continue
        selected_match = selected_dhash_tree.find_within(dhash_value, near_duplicate_threshold)
        if selected_match is not None:
            matched_item, distance = selected_match
            rejection_counts["selected_near_duplicate"] += 1
            rejections.append(
                {
                    "degradation_type": category,
                    "image_path": str(image_path),
                    "reason": "selected_near_duplicate",
                    "matched_path": matched_item.path,
                    "dhash_distance": distance,
                }
            )
            continue

        selection_index = len(selected)
        selected_sha256.add(content_sha256)
        selected_dhash_tree.add(HashItem(value=dhash_value, path=str(image_path)))
        selected.append(
            {
                "sample_id": f"router-{category}-{selection_index:06d}",
                "image_path": str(image_path),
                "source_relative_path": str(image_path.relative_to(source_root)),
                "source_group_id": f"{category}:{image_path.stem}",
                "degradation_type": category,
                "content_sha256": content_sha256,
                "dhash64": f"{dhash_value:016x}",
                "selection_index": selection_index,
                "selection_seed": seed,
                "class_selection_seed": class_seed,
                "unseen_status": "expert_unseen_current_lineage",
                "exclusion_inventory_sha256": inventory_sha256,
                "data_role": "difficulty_router_development",
            }
        )
        if len(selected) == samples_per_class:
            break

    if len(selected) != samples_per_class:
        raise ValueError(
            f"not enough unseen {category} images after exclusions: "
            f"selected={len(selected)} required={samples_per_class}"
        )

    stats = {
        "source_directory": str(source_directory),
        "source_images": len(source_images),
        "source_path_exclusions": source_path_exclusions,
        "source_images_after_path_exclusion": len(source_images) - source_path_exclusions,
        "selected": len(selected),
        "class_selection_seed": class_seed,
        "scanned_non_path_candidates": scanned_non_path_candidates,
        "selection_scan_rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return selected, rejections, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic expert-unseen images for difficulty-router development."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--grpo-data-dir", type=Path, default=DEFAULT_GRPO_DATA_DIR)
    parser.add_argument("--old-verl-data-dir", type=Path, default=DEFAULT_OLD_VERL_DATA_DIR)
    parser.add_argument("--sft-data-dir", type=Path, default=DEFAULT_SFT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=4,
        help="Maximum 64-bit dHash Hamming distance treated as a near duplicate.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    if not 0 <= args.near_duplicate_threshold <= 64:
        raise ValueError("near-duplicate-threshold must be between 0 and 64")

    source_root = args.source_root.expanduser().resolve(strict=True)
    grpo_data_dir = args.grpo_data_dir.expanduser().resolve(strict=True)
    old_verl_data_dir = args.old_verl_data_dir.expanduser().resolve(strict=True)
    sft_data_dir = args.sft_data_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()

    exclusions, grpo_path_sets = _load_grpo_exclusions(grpo_data_dir)
    _merge_exclusions(exclusions, _load_sft_exclusions(sft_data_dir))
    parquet_report = _cross_check_old_verl_parquet(old_verl_data_dir, grpo_path_sets)

    exclusion_records, excluded_sha256, representatives, excluded_tree = _build_exclusion_inventory(
        exclusions
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    exclusion_inventory_path = output_dir / "exclusion_inventory.jsonl"
    exclusion_inventory_sha256 = _write_jsonl(exclusion_inventory_path, exclusion_records)

    selected_sha256: set[str] = set()
    selected_tree = HammingBKTree()
    all_selected: list[dict[str, Any]] = []
    all_rejections: list[dict[str, Any]] = []
    category_report: dict[str, dict[str, Any]] = {}
    selected_files: dict[str, dict[str, Any]] = {}

    for category, relative_directory in CLASS_DIRECTORIES.items():
        selected, rejections, stats = _select_category(
            category=category,
            source_root=source_root,
            source_directory=source_root / relative_directory,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            exclusions=exclusions,
            excluded_sha256=excluded_sha256,
            excluded_representative_by_sha256=representatives,
            excluded_dhash_tree=excluded_tree,
            selected_sha256=selected_sha256,
            selected_dhash_tree=selected_tree,
            near_duplicate_threshold=args.near_duplicate_threshold,
            inventory_sha256=exclusion_inventory_sha256,
        )
        category_path = output_dir / f"{category}.jsonl"
        category_sha256 = _write_jsonl(category_path, selected)
        selected_files[category] = {
            "path": _relative_or_absolute(category_path),
            "rows": len(selected),
            "sha256": category_sha256,
        }
        category_report[category] = stats
        all_selected.extend(selected)
        all_rejections.extend(rejections)
        print(
            f"{category}: source={stats['source_images']} selected={len(selected)} "
            f"rejections={stats['selection_scan_rejection_counts']}",
            flush=True,
        )

    all_candidates_path = output_dir / "all_candidates.jsonl"
    all_candidates_sha256 = _write_jsonl(all_candidates_path, all_selected)
    rejections_path = output_dir / "selection_rejections.jsonl"
    rejections_sha256 = _write_jsonl(rejections_path, all_rejections)

    source_files = [
        *sorted(grpo_data_dir.glob("*_train.jsonl")),
        *sorted(grpo_data_dir.glob("*_val.jsonl")),
        *sorted(sft_data_dir.glob("*_expert_rl_aligned_train.jsonl")),
    ]
    manifest = {
        "version": 1,
        "purpose": "expert-unseen difficulty-router development image selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "source_root": str(source_root),
        "class_directories": CLASS_DIRECTORIES,
        "samples_per_class": args.samples_per_class,
        "total_selected": len(all_selected),
        "selection_seed": args.seed,
        "near_duplicate_method": "dhash64",
        "near_duplicate_hamming_threshold": args.near_duplicate_threshold,
        "unseen_definition": {
            "canonical_path": True,
            "sha256_content": True,
            "dhash64_near_duplicate": True,
            "excluded_training_sources": [
                "current four-expert SFT/cold-start train manifests",
                "current four-expert GRPO train and validation manifests",
            ],
            "scope_note": (
                "The inventory covers the current qwen3_5_0721 SFT lineage and current GRPO "
                "manifests. Archived model lineages without a recoverable manifest are not claimed."
            ),
        },
        "exclusion_source_files": [
            {
                "path": _relative_or_absolute(path),
                "sha256": _file_digest(path),
            }
            for path in source_files
        ],
        "excluded_unique_paths": len(exclusions),
        "excluded_unique_content_sha256": len(excluded_sha256),
        "exclusion_inventory": {
            "path": _relative_or_absolute(exclusion_inventory_path),
            "rows": len(exclusion_records),
            "sha256": exclusion_inventory_sha256,
        },
        "old_verl_parquet_cross_check": parquet_report,
        "categories": category_report,
        "selected_files": selected_files,
        "all_candidates": {
            "path": _relative_or_absolute(all_candidates_path),
            "rows": len(all_selected),
            "sha256": all_candidates_sha256,
        },
        "rejections": {
            "path": _relative_or_absolute(rejections_path),
            "rows": len(all_rejections),
            "sha256": rejections_sha256,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"wrote selection manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
