"""Run the stage G four-expert validation matrix.

Usage:
    python examples/image_restoration_multi_agent/stage_g_smoke_test.py \
        --expert-decision-mode replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agents import VLMDegradationDiagnosisAgent
from config import StageGExampleConfig, load_stage_g_example_config
from factory import RealControllerFactory
from lit_agent import StageGImageRestorationAgent
from schemas import DegradationType, ExpertDecisionMode
from tool_registry import ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]

DEFAULT_CASES: dict[DegradationType, tuple[str, str]] = {
    DegradationType.FOG: ("fog.png", "focalnet_dehaze"),
    DegradationType.SNOW: ("snow.png", "focalnet_desnow"),
    DegradationType.RAIN: ("rain.jpg", "turbo_rain"),
    DegradationType.LOW_LIGHT: ("lowlight.png", "hvicidnet"),
}


def _task_for_category(
    config: StageGExampleConfig,
    degradation_type: DegradationType,
    inputs_dir: Path,
    output_root: Path,
    expert_decision_mode: ExpertDecisionMode,
) -> dict[str, Any]:
    input_name, action = DEFAULT_CASES[degradation_type]
    input_path = (inputs_dir / input_name).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"stage G input does not exist: {input_path}")
    output_dir = (output_root / degradation_type.value).resolve()
    return {
        "image_path": str(input_path),
        "degradation_type": degradation_type.value,
        "scripted_actions": [action, "stop"],
        "output_dir": str(output_dir),
        "routing_mode": config.vlm.routing_mode.value,
        "expert_decision_mode": expert_decision_mode.value,
    }


async def run_smoke_matrix(
    inputs_dir: Path,
    output_root: Path,
    expert_decision_mode: ExpertDecisionMode,
    max_tokens_override: int | None = None,
) -> dict[str, Any]:
    """Execute all four expert routes and return a combined report."""

    config = load_stage_g_example_config(EXAMPLE_DIR / "config" / "stage_g.yaml")
    if max_tokens_override is not None:
        if max_tokens_override < 1:
            raise ValueError("max_tokens_override must be positive")
        config.vlm.max_tokens = max_tokens_override
        config.expert_vlm.max_tokens = max_tokens_override
    registry = ToolRegistry.from_yaml(config.tools_config)
    for _, action in DEFAULT_CASES.values():
        registry.validate_action(action)
    agent = StageGImageRestorationAgent(
        config,
        RealControllerFactory(config, registry),
        VLMDegradationDiagnosisAgent(config.vlm),
    )
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=agl.OtelTracer(), heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    reports: dict[str, Any] = {}

    with runner.run_context(agent=agent, store=store):
        for degradation_type in DegradationType:
            task = _task_for_category(
                config,
                degradation_type,
                inputs_dir,
                output_root,
                expert_decision_mode,
            )
            rollout = await runner.step(task, resources={}, mode="val")
            attempts = await store.query_attempts(rollout.rollout_id)
            spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)
            result = agent.results[rollout.rollout_id]
            workflow = result.workflow_result
            decisions = workflow.state.steps if workflow is not None else []
            reports[degradation_type.value] = {
                "rollout_status": rollout.status,
                "trajectory_id": result.trajectory_id,
                "routing_source": result.routing_source.value if result.routing_source is not None else None,
                "diagnosis_parse_status": result.diagnosis_attempt.parse_status.value,
                "expert_name": workflow.state.expert_name.value if workflow is not None else None,
                "resource_name": decisions[0].expert_decision.resource_name if decisions else None,
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
                "stage_g_result_path": result.result_path,
                "operation_names": [
                    span.attributes.get("agentlightning.operation.name")
                    for span in spans
                    if span.attributes and span.attributes.get("agentlightning.operation.name")
                ],
            }

    matrix_path = output_root.resolve() / f"stage_g_{expert_decision_mode.value}_matrix.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "expert_decision_mode": expert_decision_mode.value,
        "max_tokens_override": max_tokens_override,
        "shared_tool_actions": list(registry.actions),
        "expert_resources": {
            expert.value: resource.model_dump(mode="json") for expert, resource in config.expert_resources.items()
        },
        "reports": reports,
        "matrix_path": str(matrix_path),
    }
    matrix_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    """Parse command-line options and run the stage G matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=REPO_ROOT / "External_Tools" / "inputs")
    parser.add_argument(
        "--expert-decision-mode",
        type=ExpertDecisionMode,
        default=ExpertDecisionMode.REPLAY,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXAMPLE_DIR / "artifacts" / "stage_g_smoke",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    summary = asyncio.run(
        run_smoke_matrix(
            args.inputs_dir.expanduser().resolve(),
            args.output_root.expanduser().resolve(),
            args.expert_decision_mode,
            args.max_tokens,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
