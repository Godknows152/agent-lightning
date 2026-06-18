"""Run stage F with replay or strict served-VLM expert decisions.

Usage:
    python examples/image_restoration_multi_agent/stage_f_smoke_test.py \
        --input External_Tools/inputs/fog.png \
        --degradation-type fog \
        --expert-decision-mode replay \
        --actions focalnet_dehaze stop
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agents import VLMDegradationDiagnosisAgent
from config import load_stage_f_example_config
from factory import RealControllerFactory
from lit_agent import StageFImageRestorationAgent
from schemas import DegradationType, ExpertDecisionMode, RoutingMode
from tool_registry import STOP_ACTION, ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]


async def run_smoke_test(
    input_path: Path,
    output_dir: Path,
    degradation_type: DegradationType,
    actions: list[str],
    routing_mode: RoutingMode,
    expert_decision_mode: ExpertDecisionMode,
) -> dict[str, Any]:
    """Execute one traced stage F rollout and return a concise report."""

    config = load_stage_f_example_config(EXAMPLE_DIR / "config" / "stage_f.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    normalized_actions = list(actions)
    if normalized_actions[-1] != STOP_ACTION:
        normalized_actions.append(STOP_ACTION)
    for action in normalized_actions:
        registry.validate_action(action)

    agent = StageFImageRestorationAgent(
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
        "scripted_actions": normalized_actions,
        "output_dir": str(output_dir.resolve()),
        "routing_mode": routing_mode.value,
        "expert_decision_mode": expert_decision_mode.value,
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    result = agent.results[rollout.rollout_id]
    workflow = result.workflow_result
    decisions = workflow.state.steps if workflow is not None else []
    operation_names = [
        span.attributes.get("agentlightning.operation.name")
        for span in spans
        if span.attributes and span.attributes.get("agentlightning.operation.name")
    ]
    return {
        "rollout_status": rollout.status,
        "trajectory_id": result.trajectory_id,
        "routing_mode": result.routing_mode.value,
        "routing_source": result.routing_source.value if result.routing_source is not None else None,
        "diagnosis_parse_status": result.diagnosis_attempt.parse_status.value,
        "expert_decision_mode": result.expert_decision_mode.value,
        "expert_decisions": [
            {
                "step_index": step.step_index,
                "source": step.expert_decision.decision_source.value,
                "parse_status": step.expert_decision.parse_status.value,
                "action": step.expert_decision.action,
                "error": step.expert_decision.error,
            }
            for step in decisions
        ],
        "termination_reason": result.termination_reason,
        "final_reward": result.final_reward,
        "final_reward_from_trace": agl.find_final_reward(spans),
        "best_image": workflow.state.best_image if workflow is not None else None,
        "trajectory_path": workflow.trajectory_path if workflow is not None else None,
        "stage_f_result_path": result.result_path,
        "operation_names": operation_names,
    }


def main() -> None:
    """Parse command-line options and run the stage F smoke test."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "External_Tools/inputs/fog.png")
    parser.add_argument("--degradation-type", type=DegradationType, default=DegradationType.FOG)
    parser.add_argument("--actions", nargs="+", default=["focalnet_dehaze", "stop"])
    parser.add_argument("--routing-mode", type=RoutingMode, default=RoutingMode.ORACLE_OBSERVE)
    parser.add_argument(
        "--expert-decision-mode",
        type=ExpertDecisionMode,
        default=ExpertDecisionMode.REPLAY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "stage_f_smoke",
    )
    args = parser.parse_args()
    summary = asyncio.run(
        run_smoke_test(
            args.input.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            args.degradation_type,
            args.actions,
            args.routing_mode,
            args.expert_decision_mode,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
