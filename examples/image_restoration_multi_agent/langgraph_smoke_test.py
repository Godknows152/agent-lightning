"""Run the no-model LangGraph image-restoration workflow.

Usage:
    python examples/image_restoration_multi_agent/langgraph_smoke_test.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from config import load_example_config
from langgraph_factory import DeterministicLangGraphFactory
from schemas import DegradationType, RestorationTask
from tool_registry import ToolRegistry

EXAMPLE_DIR = Path(__file__).resolve().parent
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def run_smoke_test(output_dir: Path) -> dict[str, Any]:
    """Execute one checkpointed deterministic graph and return its summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.png"
    input_path.write_bytes(ONE_PIXEL_PNG)
    trajectory_id = "langgraph-smoke"
    task = RestorationTask(
        image_path=str(input_path),
        degradation_type=DegradationType.FOG,
        scripted_actions=["scunet", "s2former", "stop"],
        score_sequence=[0.40, 0.55, 0.52],
        output_dir=str(output_dir / "trajectory"),
    )
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    workflow = DeterministicLangGraphFactory(config, registry).build(task)
    result = workflow.invoke(task, trajectory_id=trajectory_id)
    checkpoint = workflow.get_checkpoint(trajectory_id)
    checkpoint_trajectory = checkpoint.get("trajectory")
    if checkpoint_trajectory is None:
        raise ValueError("LangGraph smoke checkpoint is missing trajectory state")
    return {
        **result.summary,
        "trajectory_path": result.trajectory_path,
        "graph_nodes": sorted(workflow.node_names()),
        "checkpoint_termination_reason": checkpoint_trajectory["termination_reason"],
    }


def main() -> None:
    """Parse CLI options and run the deterministic graph."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "langgraph_smoke",
        help="Directory for the input, copied intermediate images, and trajectory JSON.",
    )
    args = parser.parse_args()
    summary = run_smoke_test(args.output_dir.expanduser().resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
