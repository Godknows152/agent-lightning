"""Run the stage D traced smoke test with real restoration and IQA models.

Usage:
    python examples/image_restoration_multi_agent/real_smoke_test.py \
        --input External_Tools/inputs/fog.png \
        --action focalnet_dehaze
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from config import load_real_example_config
from factory import RealControllerFactory
from lit_agent import RealImageRestorationAgent
from tool_registry import STOP_ACTION, ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]


async def run_smoke_test(input_path: Path, output_dir: Path, action: str) -> dict[str, Any]:
    """Execute one real traced rollout and return its high-signal summary."""

    config = load_real_example_config(EXAMPLE_DIR / "config" / "real.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    registry.validate_action(action)
    if action == STOP_ACTION:
        raise ValueError("the real smoke test action cannot be stop")
    agent = RealImageRestorationAgent(RealControllerFactory(config, registry))
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=tracer, heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path.resolve()),
        "degradation_type": "fog",
        "scripted_actions": [action, "stop"],
        "output_dir": str(output_dir.resolve()),
        "visual_evidence": ["stage D scripted routing smoke test"],
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
        "original_scores": result.state.original_evaluation.raw_scores,
        "best_scores": result.state.best_evaluation.raw_scores,
        "operation_names": operation_names,
        "final_reward_from_trace": agl.find_final_reward(spans),
    }


def main() -> None:
    """Parse command-line options and run the real-model smoke test."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "External_Tools/inputs/fog.png")
    parser.add_argument("--action", default="focalnet_dehaze")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "real_smoke",
    )
    args = parser.parse_args()
    summary = asyncio.run(
        run_smoke_test(
            args.input.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            args.action,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
