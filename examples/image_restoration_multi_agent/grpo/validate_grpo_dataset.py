#!/usr/bin/env python3
"""Validate one or more stage H JSONL seed manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERT_TYPES = {"fog", "snow", "rain", "low_light"}


def validate_manifest(path: Path) -> int:
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    degradation_types: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            payload = json.loads(line)
            required = {"sample_id", "image_path", "degradation_type", "output_root", "visual_evidence"}
            if set(payload) != required:
                raise ValueError(f"{path}:{line_number}: fields must be exactly {sorted(required)}")
            sample_id = payload["sample_id"]
            image_path = str(Path(payload["image_path"]).expanduser().resolve())
            degradation_type = payload["degradation_type"]
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            if image_path in seen_images:
                raise ValueError(f"{path}:{line_number}: duplicate image_path {image_path}")
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"{path}:{line_number}: missing image {image_path}")
            if degradation_type not in EXPERT_TYPES:
                raise ValueError(f"{path}:{line_number}: invalid degradation_type {degradation_type}")
            if not isinstance(payload["visual_evidence"], list):
                raise ValueError(f"{path}:{line_number}: visual_evidence must be a list")
            seen_ids.add(sample_id)
            seen_images.add(image_path)
            degradation_types.add(degradation_type)
            count += 1
    if count == 0:
        raise ValueError(f"empty manifest: {path}")
    if len(degradation_types) != 1:
        raise ValueError(f"one expert manifest must contain one degradation type: {degradation_types}")
    print(f"{path}: count={count} degradation_type={next(iter(degradation_types))}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    total = sum(validate_manifest(path.expanduser().resolve()) for path in args.manifests)
    print(f"validated_total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
