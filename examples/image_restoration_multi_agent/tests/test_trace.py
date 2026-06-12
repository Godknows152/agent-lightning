"""Agent Lightning rollout and trace tests for stage C."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from config import load_example_config
from factory import DeterministicControllerFactory
from lit_agent import DeterministicImageRestorationAgent
from tool_registry import ToolRegistry

import agentlightning as agl

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_lit_agent_records_operations_and_one_final_reward(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"trace-image-placeholder")
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    agent = DeterministicImageRestorationAgent(
        DeterministicControllerFactory(config, ToolRegistry.from_yaml(config.tools_config))
    )
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[dict[str, Any]](tracer=tracer, heartbeat_interval=0)
    store = agl.InMemoryLightningStore()
    task = {
        "image_path": str(input_path),
        "degradation_type": "fog",
        "scripted_actions": ["restoration_model_a", "stop"],
        "score_sequence": [0.4, 0.6],
        "output_dir": str(tmp_path / "run"),
    }

    with runner.run_context(agent=agent, store=store):
        rollout = await runner.step(task, resources={}, mode="val")
        attempts = await store.query_attempts(rollout.rollout_id)
        spans = await store.query_spans(rollout.rollout_id, attempts[-1].attempt_id)

    operation_spans = [span for span in spans if span.name == "agentlightning.operation"]
    operation_names = {span.attributes.get("agentlightning.operation.name") for span in operation_spans}
    expert_spans = [span for span in operation_spans if span.attributes.get("agent.name") == "fog_expert"]
    non_expert_spans = [
        span
        for span in operation_spans
        if str(span.attributes.get("agentlightning.operation.name", "")).startswith(
            ("restoration_worker", "iqa_evaluator")
        )
    ]
    reward_spans = [
        span
        for span in spans
        if span.name == "agentlightning.annotation" and "agentlightning.reward.0.value" in (span.attributes or {})
    ]

    assert rollout.status == "succeeded"
    assert {
        "diagnosis_agent.decision",
        "fog_expert.decision",
        "restoration_worker.restoration_model_a",
        "iqa_evaluator.original",
        "iqa_evaluator.score",
    }.issubset(operation_names)
    assert len(expert_spans) == 2
    assert all("agent.name" not in span.attributes for span in non_expert_spans)
    assert len(reward_spans) == 1
    final_reward = agl.find_final_reward(spans)
    assert final_reward is not None
    assert abs(final_reward - 0.19) < 1e-9
    assert agent.results[rollout.rollout_id].state.termination_reason == "expert_stop"
