"""Prompt contract tests for stage F restoration experts."""

from __future__ import annotations

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
from agents.prompts import (
    EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION,
    build_expert_single_step_sft_system_prompt,
    build_expert_single_step_sft_user_prompt,
)
from config import load_example_config
from schemas import ExpertName
from tool_registry import ToolRegistry

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def _registry() -> ToolRegistry:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    return ToolRegistry.from_yaml(config.tools_config)


def test_all_expert_prompts_embed_the_same_complete_tool_schema() -> None:
    registry = _registry()
    prompts = {
        expert: build_expert_system_prompt(expert, registry) for expert in ExpertName
    }

    assert all("<tools>" not in prompt for prompt in prompts.values())
    assert all("<tool_descriptions>" not in prompt for prompt in prompts.values())
    assert all(expert.value not in prompts[expert] for expert in ExpertName)
    assert all("primary degradation is" not in prompt for prompt in prompts.values())
    assert all(
        "specialized through training data" not in prompt for prompt in prompts.values()
    )
    assert all(
        "You are an image restoration expert." in prompt for prompt in prompts.values()
    )


def test_expert_tool_schema_embeds_action_descriptions_from_registry() -> None:
    registry = _registry()
    schema = registry.build_tool_schema()
    action_schema = schema["function"]["parameters"]["properties"]["action"]
    descriptions = action_schema["description"]

    assert (
        "- focalnet_dehaze: FocalNet image dehazing model with the ITS checkpoint."
        in descriptions
    )
    assert "- hvicidnet: HVI-CIDNet low-light image enhancement model." in descriptions
    assert (
        "- stop: Stop the trajectory and keep the historical best restored image."
        in descriptions
    )


def test_single_step_prompt_hides_stop_description() -> None:
    registry = _registry()
    prompt = build_expert_single_step_sft_system_prompt(ExpertName.FOG, registry)
    schema = registry.build_tool_schema(include_stop=False)["function"]

    parameters = cast(dict[str, object], schema["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    action = cast(dict[str, object], properties["action"])
    assert "stop" not in action["enum"]
    assert "- stop:" not in str(action["description"])
    assert "- focalnet_dehaze:" in str(action["description"])
    assert "<tools>" not in prompt
    assert "<tool_descriptions>" not in prompt


def test_single_step_prompt_does_not_constrain_reasoning_text() -> None:
    prompt = build_expert_single_step_sft_system_prompt(ExpertName.RAIN, _registry())
    user_prompt = build_expert_single_step_sft_user_prompt()

    assert EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION in prompt
    assert "<think>" not in prompt
    assert "natural language" not in prompt
    assert "no other text" not in prompt
    assert "no extra text" not in user_prompt


def test_diagnosis_prompt_uses_hermes_without_model_generated_route() -> None:
    prompt = build_diagnosis_system_prompt()

    assert DIAGNOSIS_PROMPT_VERSION in prompt
    assert (
        '<tool_call>\n{"name":"diagnose_degradation","arguments":{"primary_type":"fog",'
        '"visual_evidence":["global contrast is reduced","distant regions appear washed out"]}}\n</tool_call>'
        in prompt
    )
    assert "Do not output confidence or route_to" in prompt
    assert "diagnose_degradation" in str(DIAGNOSIS_TOOL_SCHEMA)
    assert "route_to" not in str(DIAGNOSIS_TOOL_SCHEMA)


def test_expert_prompt_locks_the_qwen_xml_wire_format() -> None:
    prompt = build_expert_system_prompt(ExpertName.FOG, _registry())

    assert EXPERT_PROMPT_VERSION in prompt
    assert "<function=restore_image>" not in prompt
    assert "<parameter=action>" not in prompt
    assert "<action_enum_value>" not in prompt
    assert "Call restore_image exactly once." in prompt
    assert "action argument must be one enum value" in prompt
    assert '"args"' not in prompt


def test_state_prompt_contains_only_natural_language_tool_feedback_and_tool_call_reminder() -> (
    None
):
    prompt = build_expert_state_prompt(
        history_feedback="Historical tool feedback: Step 0: selected action ridcp; IQA aggregate_score=0.6700.",
    )

    assert (
        "Historical tool feedback: Step 0: selected action ridcp; IQA aggregate_score=0.6700."
        in prompt
    )
    assert "Workflow state" not in prompt
    assert "action_and_evaluation_history_json" not in prompt
    assert "latest_iqa_feedback" not in prompt
    assert "tool_call" in prompt.lower()
