"""Versioned prompts for real restoration expert agents."""

from __future__ import annotations

import json

from schemas import DegradationType, ExpertName
from tool_registry import ToolRegistry

DIAGNOSIS_PROMPT_VERSION = "diagnosis-hermes-v1"
DIAGNOSIS_TOOL_NAME = "diagnose_degradation"
EXPERT_PROMPT_VERSION = "expert-hermes-v1"
EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION = "expert-single-step-sft-hermes-v1"

DIAGNOSIS_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": DIAGNOSIS_TOOL_NAME,
        "description": "Classify the single primary image degradation and record visible evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "primary_type": {
                    "type": "string",
                    "enum": [item.value for item in DegradationType],
                },
                "visual_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["primary_type", "visual_evidence"],
            "additionalProperties": False,
        },
    },
}

EXPERT_DEGRADATION: dict[ExpertName, DegradationType] = {
    ExpertName.FOG: DegradationType.FOG,
    ExpertName.SNOW: DegradationType.SNOW,
    ExpertName.RAIN: DegradationType.RAIN,
    ExpertName.LOW_LIGHT: DegradationType.LOW_LIGHT,
}


def build_diagnosis_system_prompt() -> str:
    """Build the strict Hermes prompt used by the degradation diagnosis agent."""

    function_schema = DIAGNOSIS_TOOL_SCHEMA["function"]
    serialized_schema = json.dumps(function_schema, ensure_ascii=False, separators=(",", ":"))
    return f"""Prompt version: {DIAGNOSIS_PROMPT_VERSION}

You are the degradation diagnosis agent. Inspect the input image and identify exactly one
primary degradation from fog, snow, rain, or low_light. Base the decision only on visible
image evidence. Do not use a filename, path, hidden label, or tool-name heuristic.

You are provided with one function signature inside <tools></tools> tags:
<tools>
{serialized_schema}
</tools>

Return exactly one Hermes tool call and no other actionable output:
<tool_call>
{{"name":"diagnose_degradation","arguments":{{"primary_type":"fog","visual_evidence":["global contrast is reduced","distant regions appear washed out"]}}}}
</tool_call>

The example category and evidence illustrate syntax only. Determine the category from the
actual image.

Rules:
1. Emit exactly one <tool_call></tool_call> block.
2. The name field must be exactly diagnose_degradation.
3. The arguments object must contain exactly primary_type and visual_evidence.
4. primary_type must be exactly one of fog, snow, rain, low_light.
5. visual_evidence must be a JSON array of concise observations visible in the image.
6. Do not output confidence or route_to; the controller owns the fixed category-to-expert mapping.
7. Do not wrap the tool call in a Markdown code fence or emit bare JSON outside the tags.
8. Do not call a restoration tool and do not propose a restoration sequence."""


def build_expert_system_prompt(
    expert_name: ExpertName,
    tool_registry: ToolRegistry,
    *,
    allow_stop: bool = True,
    min_stop_tool_calls: int | None = None,
) -> str:
    """Build one expert prompt with the complete shared Hermes tool schema."""

    function_schema = tool_registry.build_tool_schema(include_stop=allow_stop)["function"]
    serialized_schema = json.dumps(function_schema, ensure_ascii=False, separators=(",", ":"))
    tool_descriptions = tool_registry.build_tool_descriptions(include_stop=allow_stop)
    if allow_stop:
        stop_instruction = """The example action illustrates syntax only. Determine the action from the actual image,
history, and IQA feedback. To finish the trajectory, use action stop:
<tool_call>
{"name":"restore_image","arguments":{"action":"stop"}}
</tool_call>"""
        stop_rule = "Use action stop when further processing is unlikely to improve the historical best image."
    else:
        threshold = min_stop_tool_calls if min_stop_tool_calls is not None else "the configured minimum"
        stop_instruction = f"""The example action illustrates syntax only. Determine the action from the actual image,
history, and IQA feedback. The stop action is not available yet because fewer than
{threshold} restoration tool calls have been executed. Continue with one restoration action
from the supplied schema."""
        stop_rule = "Do not use action stop before it appears in the supplied schema."
    return f"""Prompt version: {EXPERT_PROMPT_VERSION}

You are an image restoration expert.

Inspect the current input image and select exactly one registered restoration action for
the current restoration step. All registered restoration models are available to you. A
tool's common purpose does not restrict when it may be selected, so choose only from the
observed image content, the action history, and the IQA feedback supplied by the
controller. When the evidence is not decisive, prefer trying a restoration action
from a tool family that differs from the tool families already used in this
trajectory; repeat a tool family only when the image evidence or IQA feedback
clearly supports it. Your objective is to maximize the quality of the historical
best image.

You are provided with one function signature inside <tools></tools> tags:
<tools>
{serialized_schema}
</tools>

The available actions have the following intended uses:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>

Return exactly one Hermes tool call with no extra text. In the template below, replace
<action_enum_value> with one action enum value from the supplied schema:
<tool_call>
{{"name":"restore_image","arguments":{{"action":"<action_enum_value>"}}}}
</tool_call>

{stop_instruction}

Rules:
1. Emit exactly one <tool_call></tool_call> block per turn with no natural language before or after it.
2. The name field must be exactly restore_image.
3. The arguments object must contain exactly one field named action.
4. action must be one of the enum values in the supplied schema.
5. {stop_rule}
6. Do not emit any text before or after the <tool_call></tool_call> block.
7. Do not wrap the tool call in a Markdown code fence or emit bare JSON.
8. Do not claim an action improved the image before receiving the next IQA result.
9. Avoid repeating an action that already degraded quality unless later evidence justifies it.
10. Prefer exploring a different tool family from prior restoration steps unless repetition is clearly justified.
11. Do not diagnose again, change experts, or delegate the decision to another agent."""


def build_expert_single_step_sft_system_prompt(expert_name: ExpertName, tool_registry: ToolRegistry) -> str:
    """Build the initial-state prompt used for single-step expert SFT."""

    function_schema = tool_registry.build_tool_schema(include_stop=False)["function"]
    serialized_schema = json.dumps(function_schema, ensure_ascii=False, separators=(",", ":"))
    tool_descriptions = tool_registry.build_tool_descriptions(include_stop=False)
    return f"""Prompt version: {EXPERT_SINGLE_STEP_SFT_PROMPT_VERSION}

You are an image restoration expert.

Inspect the current input image and select exactly one registered restoration action for
the first restoration step. All registered restoration models are available to you. A
tool's common purpose does not restrict when it may be selected, so choose only from the
observed image content and the learned policy.

You are provided with one function signature inside <tools></tools> tags:
<tools>
{serialized_schema}
</tools>

The available actions have the following intended uses:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>

Return exactly one Hermes tool call with no extra text. In the template below, replace
<action_enum_value> with one action enum value from the supplied schema:
<tool_call>
{{"name":"restore_image","arguments":{{"action":"<action_enum_value>"}}}}
</tool_call>

The example action illustrates syntax only. Determine the action from the actual image.

Rules:
1. Emit exactly one <tool_call></tool_call> block with no natural language before or after it.
2. The name field must be exactly restore_image.
3. The arguments object must contain exactly one field named action.
4. action must be one of the enum values in the supplied schema.
5. Do not emit any text before or after the <tool_call></tool_call> block.
6. Do not output stop in this initial single-step SFT task.
7. Do not wrap the tool call in a Markdown code fence or emit bare JSON."""


def build_expert_single_step_sft_user_prompt() -> str:
    """Build the image-only initial-state instruction used by expert SFT."""

    return "<image>\nSelect exactly one restoration action for this image using one Hermes tool call and no extra text."


def build_expert_state_prompt(
    *,
    history_feedback: str,
) -> str:
    """Build the text state paired with the current image at one expert turn."""

    return (
        f"{history_feedback}\n\n"
        "Select the next restoration action for the current image. Respond with exactly one Hermes "
        "<tool_call> block following the system instructions, with no extra text."
    )
