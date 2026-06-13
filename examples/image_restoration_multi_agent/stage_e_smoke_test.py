"""Run stage E with GLM-4.1V diagnosis and real restoration/IQA models.

Usage:
    python examples/image_restoration_multi_agent/stage_e_smoke_test.py \
        --input External_Tools/inputs/fog.png \
        --degradation-type fog \
        --action focalnet_dehaze \
        --routing-mode oracle_observe
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agents import VLMDegradationDiagnosisAgent
from config import load_stage_e_example_config
from diagnosis_metrics import build_diagnosis_metrics
from factory import RealControllerFactory
from lit_agent import VLMImageRestorationAgent
from schemas import DegradationType, RoutingMode
from tool_registry import STOP_ACTION, ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]


async def run_smoke_test(
    input_path: Path,
    output_dir: Path,
    degradation_type: DegradationType,
    action: str,
    routing_mode: RoutingMode,
) -> dict[str, Any]:
    """Execute one traced stage E rollout and return a concise report."""

    config = load_stage_e_example_config(EXAMPLE_DIR / "config" / "stage_e.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    registry.validate_action(action)
    if action == STOP_ACTION:
        raise ValueError("the stage E smoke-test action cannot be stop")
    agent = VLMImageRestorationAgent(
        config,
        RealControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm),
    )
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=tracer, heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path.resolve()),
        "degradation_type": degradation_type.value,
        "scripted_actions": [action, "stop"],
        "output_dir": str(output_dir.resolve()),
        "routing_mode": routing_mode.value,
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    result = agent.results[rollout.rollout_id]
    workflow = result.workflow_result
    operation_names = [
        span.attributes.get("agentlightning.operation.name")
        for span in spans
        if span.attributes and span.attributes.get("agentlightning.operation.name")
    ]
    return {
        "rollout_status": rollout.status,
        "trajectory_id": result.trajectory_id,
        "backend": result.diagnosis_attempt.backend,
        "routing_mode": result.routing_mode.value,
        "routing_source": result.routing_source.value if result.routing_source is not None else None,
        "parse_status": result.diagnosis_attempt.parse_status.value,
        "predicted_diagnosis": (
            result.diagnosis_attempt.diagnosis.model_dump(mode="json")
            if result.diagnosis_attempt.diagnosis is not None
            else None
        ),
        "actual_diagnosis": result.actual_diagnosis.model_dump(mode="json") if result.actual_diagnosis else None,
        "vlm_raw_response": result.diagnosis_attempt.raw_response,
        "vlm_latency_seconds": result.diagnosis_attempt.latency_seconds,
        "prompt_token_count": len(result.diagnosis_attempt.prompt_token_ids or []),
        "generated_token_count": len(result.diagnosis_attempt.generated_token_ids or []),
        "termination_reason": result.termination_reason,
        "final_reward": result.final_reward,
        "final_reward_from_trace": agl.find_final_reward(spans),
        "best_image": workflow.state.best_image if workflow is not None else None,
        "trajectory_path": workflow.trajectory_path if workflow is not None else None,
        "stage_e_result_path": result.result_path,
        "diagnosis_metrics": build_diagnosis_metrics([(degradation_type, result.diagnosis_attempt)]),
        "operation_names": operation_names,
    }


def main() -> None:
    """Parse command-line options and run the stage E smoke test."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "External_Tools/inputs/fog.png")
    parser.add_argument("--degradation-type", type=DegradationType, default=DegradationType.FOG)
    parser.add_argument("--action", default="focalnet_dehaze")
    parser.add_argument("--routing-mode", type=RoutingMode, default=RoutingMode.ORACLE_OBSERVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "stage_e_smoke",
    )
    args = parser.parse_args()
    summary = asyncio.run(
        run_smoke_test(
            args.input.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            args.degradation_type,
            args.action,
            args.routing_mode,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
