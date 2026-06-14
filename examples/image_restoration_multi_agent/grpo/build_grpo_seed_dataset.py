#!/usr/bin/env python3
"""Build deterministic image-only GRPO seed manifests for four experts.

Run from the Agent Lightning repository root:

    python examples/image_restoration_multi_agent/grpo/build_grpo_seed_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_DIRECTORIES = {
    "fog": "fog_series/fog",
    "snow": "snow_series/snow",
    "rain": "rain_series/rain",
    "low_light": "night_series/night",
}


def _images(directory: Path) -> list[Path]:
    return sorted(path.resolve() for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _write_manifest(
    *,
    image_paths: list[Path],
    degradation_type: str,
    output_path: Path,
    artifact_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, image_path in enumerate(image_paths):
            record = {
                "sample_id": f"{degradation_type}-{index:06d}",
                "image_path": str(image_path),
                "degradation_type": degradation_type,
                "output_root": str((artifact_root / degradation_type).resolve()),
                "visual_evidence": [],
            }
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/LXJ/Python_Projects/AIA_Restore旧数据存放/OpenReal_80k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/image_restoration_multi_agent/grpo/data"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("examples/image_restoration_multi_agent/grpo/artifacts/rollouts"),
    )
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_per_class <= 0 or args.val_per_class <= 0:
        raise ValueError("train-per-class and val-per-class must be positive")

    for class_index, (degradation_type, relative_directory) in enumerate(CLASS_DIRECTORIES.items()):
        train_images = _images(args.source_root / "train" / "images" / relative_directory)
        val_images = _images(args.source_root / "test" / "images" / relative_directory)
        if len(train_images) < args.train_per_class:
            raise ValueError(f"not enough {degradation_type} train images: {len(train_images)}")
        if len(val_images) < args.val_per_class:
            raise ValueError(f"not enough {degradation_type} validation images: {len(val_images)}")

        generator = random.Random(args.seed + class_index)
        sampled_train = generator.sample(train_images, args.train_per_class)
        sampled_val = generator.sample(val_images, args.val_per_class)
        _write_manifest(
            image_paths=sampled_train,
            degradation_type=degradation_type,
            output_path=args.output_dir / f"{degradation_type}_train.jsonl",
            artifact_root=args.artifact_root,
        )
        _write_manifest(
            image_paths=sampled_val,
            degradation_type=degradation_type,
            output_path=args.output_dir / f"{degradation_type}_val.jsonl",
            artifact_root=args.artifact_root,
        )
        print(
            f"{degradation_type}: train={len(sampled_train)} val={len(sampled_val)} "
            f"source_train={len(train_images)} source_val={len(val_images)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
