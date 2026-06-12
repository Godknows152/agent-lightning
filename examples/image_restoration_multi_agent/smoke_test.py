"""Run a no-GPU traced smoke test for stages A-C.

Usage:
    python examples/image_restoration_multi_agent/smoke_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from config import load_example_config
from factory import DeterministicControllerFactory
from lit_agent import DeterministicImageRestorationAgent
from tool_registry import ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parent
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def run_smoke_test(output_dir: Path) -> dict[str, Any]:
    """Execute one traced rollout and return a printable summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.png"
    input_path.write_bytes(ONE_PIXEL_PNG)

    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    factory = DeterministicControllerFactory(config, registry)
    agent = DeterministicImageRestorationAgent(factory)
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=tracer, heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path),
        "degradation_type": "fog",
        "scripted_actions": ["restoration_model_a", "restoration_model_b", "stop"],
        "score_sequence": [0.40, 0.55, 0.52],
        "output_dir": str(output_dir / "trajectory"),
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    result = agent.results[rollout.rollout_id]
    operation_names = [
        span.attributes.get("agentlightning.operation.name")
        for span in spans
        if span.attributes and span.attributes.get("agentlightning.operation.name")
    ]
    return {
        **result.summary,
        "trajectory_path": result.trajectory_path,
        "span_names": [span.name for span in spans],
        "operation_names": operation_names,
        "final_reward_from_trace": agl.find_final_reward(spans),
    }


def main() -> None:
    """Parse CLI options and run the asynchronous smoke test."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "smoke",
        help="Directory for the input, intermediate images, and trajectory JSON.",
    )
    args = parser.parse_args()
    summary = asyncio.run(run_smoke_test(args.output_dir.expanduser().resolve()))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
