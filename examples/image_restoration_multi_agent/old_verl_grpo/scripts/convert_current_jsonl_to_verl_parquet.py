"""Convert current image-restoration JSONL data to old-verl multi-turn parquet.

Usage:
    python convert_current_jsonl_to_verl_parquet.py --expert fog --split train
    python convert_current_jsonl_to_verl_parquet.py --expert fog --split train --limit 2
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import datasets  # noqa: E402
from agents.prompts import (  # noqa: E402
    EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION,
    build_expert_single_step_sft_system_prompt,
    build_expert_single_step_sft_user_prompt,
)
from PIL import Image  # noqa: E402
from schemas import ExpertName  # noqa: E402
from tool_registry import ToolRegistry  # noqa: E402

OLD_VERL_ROOT = EXAMPLE_ROOT / "old_verl_grpo"
SUPPORTED_EXPERTS = ("fog", "low_light", "rain", "snow")
SUPPORTED_SPLITS = ("train", "val")
EXPERT_NAMES = {
    "fog": ExpertName.FOG,
    "low_light": ExpertName.LOW_LIGHT,
    "rain": ExpertName.RAIN,
    "snow": ExpertName.SNOW,
}
DEFAULT_TOOL_REGISTRY = EXAMPLE_ROOT / "config" / "tools.yaml"


def create_initial_messages(
    expert: str, tool_registry: ToolRegistry
) -> list[dict[str, str]]:
    """Build the first-turn expert messages with the current GRPO/SFT prompt."""

    expert_name = EXPERT_NAMES[expert]
    return [
        {
            "role": "system",
            "content": build_expert_single_step_sft_system_prompt(
                expert_name, tool_registry
            ),
        },
        {
            "role": "user",
            "content": build_expert_single_step_sft_user_prompt(),
        },
    ]


def load_image_as_bytes(
    image_path: str, *, placeholder_missing: bool = False
) -> dict[str, bytes]:
    """Load an image path as the verl parquet image-bytes structure."""
    path = Path(image_path)
    if not path.exists():
        if not placeholder_missing:
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.new("RGB", (1, 1), color="black")
    else:
        image = Image.open(path).convert("RGB")

    with image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


def iter_jsonl_rows(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL rows, preserving the current dataset as the source of truth."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def convert_rows(
    source_rows: list[dict[str, Any]],
    *,
    data_source: str,
    expert: str,
    tool_registry: ToolRegistry,
    tool_registry_path: Path,
    placeholder_missing: bool,
) -> list[dict[str, Any]]:
    """Convert current JSONL rows to old-verl multi-turn rows."""
    prompt = create_initial_messages(expert, tool_registry)
    converted_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        image_path = str(row["image_path"])
        degradation_type = str(row.get("degradation_type", "unknown"))
        sample_id = str(row.get("sample_id", index))

        converted_rows.append(
            {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": copy.deepcopy(prompt),
                "images": [
                    load_image_as_bytes(
                        image_path, placeholder_missing=placeholder_missing
                    )
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "image_path": image_path,
                        "degradation_type": degradation_type,
                    },
                },
                "extra_info": {
                    "index": index,
                    "sample_id": sample_id,
                    "expert": expert,
                    "expert_name": EXPERT_NAMES[expert].value,
                    "image_path": image_path,
                    "degradation_type": degradation_type,
                    "prompt_version": EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION,
                    "tool_registry_path": str(tool_registry_path),
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "restore_image": {
                            "create_kwargs": {
                                "image_path": image_path,
                                "degradation_type": degradation_type,
                            },
                        },
                    },
                },
            }
        )
    return converted_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=SUPPORTED_EXPERTS)
    parser.add_argument("--split", required=True, choices=SUPPORTED_SPLITS)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--placeholder-missing", action="store_true")
    parser.add_argument("--tool-registry", type=Path, default=DEFAULT_TOOL_REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source or (
        EXAMPLE_ROOT / "grpo" / "data" / f"{args.expert}_{args.split}.jsonl"
    )
    output = args.output or (
        OLD_VERL_ROOT / "data" / f"{args.expert}_{args.split}.parquet"
    )
    tool_registry_path = args.tool_registry.expanduser().resolve()
    tool_registry = ToolRegistry.from_yaml(tool_registry_path)

    rows = iter_jsonl_rows(source, limit=args.limit)
    converted = convert_rows(
        rows,
        data_source="restoration",
        expert=args.expert,
        tool_registry=tool_registry,
        tool_registry_path=tool_registry_path,
        placeholder_missing=bool(args.placeholder_missing),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(converted).to_parquet(str(output))

    print(f"Converted {len(converted)} rows")
    print(f"Source: {source}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
