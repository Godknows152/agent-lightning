"""Prompt contract tests for stage F restoration experts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from agents import (
    DIAGNOSIS_PROMPT_VERSION,
    DIAGNOSIS_TOOL_SCHEMA,
    EXPERT_PROMPT_VERSION,
    build_diagnosis_system_prompt,
    build_expert_state_prompt,
    build_expert_system_prompt,
)
from agents.prompts import build_expert_single_step_sft_system_prompt
from config import load_example_config
from schemas import ExpertName
from tool_registry import ToolRegistry

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def _registry() -> ToolRegistry:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    return ToolRegistry.from_yaml(config.tools_config)


def _extract_tools_schema(prompt: str) -> dict[str, object]:
    match = re.search(r"<tools>\n(.*?)\n</tools>", prompt, flags=re.DOTALL)
    assert match is not None
    payload: object = json.loads(match.group(1))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _extract_tool_descriptions(prompt: str) -> str:
    match = re.search(r"<tool_descriptions>\n(.*?)\n</tool_descriptions>", prompt, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def test_all_expert_prompts_embed_the_same_complete_tool_schema() -> None:
    registry = _registry()
    prompts = {expert: build_expert_system_prompt(expert, registry) for expert in ExpertName}
    schemas = [_extract_tools_schema(prompt) for prompt in prompts.values()]

    assert all(schema == schemas[0] for schema in schemas[1:])
    action_schema_value = schemas[0]["parameters"]
    assert isinstance(action_schema_value, dict)
    action_schema = cast(dict[str, object], action_schema_value)
    properties_value = action_schema["properties"]
    assert isinstance(properties_value, dict)
    properties = cast(dict[str, object], properties_value)
    action_value = properties["action"]
    assert isinstance(action_value, dict)
    action = cast(dict[str, object], action_value)
    assert action["enum"] == list(registry.actions)
    assert all(expert.value not in prompts[expert] for expert in ExpertName)
    assert all("primary degradation is" not in prompt for prompt in prompts.values())
    assert all("specialized through training data" not in prompt for prompt in prompts.values())
    assert all("You are an image restoration expert." in prompt for prompt in prompts.values())


def test_expert_prompt_embeds_tool_descriptions_from_registry() -> None:
    registry = _registry()
    prompt = build_expert_system_prompt(ExpertName.FOG, registry)
    descriptions = _extract_tool_descriptions(prompt)

    assert "- focalnet_dehaze: FocalNet image dehazing model with the ITS checkpoint." in descriptions
    assert "- hvicidnet: HVI-CIDNet low-light image enhancement model." in descriptions
    assert "- stop: Stop the trajectory and keep the historical best restored image." in descriptions


def test_single_step_prompt_hides_stop_description() -> None:
    registry = _registry()
    prompt = build_expert_single_step_sft_system_prompt(ExpertName.FOG, registry)
    schema = _extract_tools_schema(prompt)
    descriptions = _extract_tool_descriptions(prompt)

    parameters = cast(dict[str, object], schema["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    action = cast(dict[str, object], properties["action"])
    assert "stop" not in action["enum"]
    assert "- stop:" not in descriptions
    assert "- focalnet_dehaze:" in descriptions


def test_diagnosis_prompt_uses_hermes_without_model_generated_route() -> None:
    prompt = build_diagnosis_system_prompt()

    assert DIAGNOSIS_PROMPT_VERSION in prompt
    assert (
        '<tool_call>\n{"name":"diagnose_degradation","arguments":{"primary_type":"fog",'
        '"visual_evidence":["global contrast is reduced","distant regions appear washed out"]}}\n</tool_call>' in prompt
    )
    assert "Do not output confidence or route_to" in prompt
    assert "diagnose_degradation" in str(DIAGNOSIS_TOOL_SCHEMA)
    assert "route_to" not in str(DIAGNOSIS_TOOL_SCHEMA)


def test_expert_prompt_locks_the_qwen_xml_wire_format() -> None:
    prompt = build_expert_system_prompt(ExpertName.FOG, _registry())

    assert EXPERT_PROMPT_VERSION in prompt
    assert "<function=restore_image>" in prompt
    assert "<parameter=action>" in prompt
    assert "<action_enum_value>" in prompt
    assert "<function=restore_image>\n<parameter=action>\nstop\n</parameter>\n</function>" in prompt
    assert "must be exactly restore_image" in prompt
    assert "arguments object must contain exactly one field named action" in prompt
    assert '"args"' not in prompt


def test_state_prompt_contains_only_natural_language_tool_feedback_and_tool_call_reminder() -> None:
    prompt = build_expert_state_prompt(
        history_feedback="Historical tool feedback: Step 0: selected action ridcp; IQA aggregate_score=0.6700.",
    )

    assert "Historical tool feedback: Step 0: selected action ridcp; IQA aggregate_score=0.6700." in prompt
    assert "Workflow state" not in prompt
    assert "action_and_evaluation_history_json" not in prompt
    assert "latest_iqa_feedback" not in prompt
    assert "tool_call" in prompt.lower()
