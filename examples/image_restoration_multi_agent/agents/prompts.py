"""Versioned prompts for real restoration expert agents."""

from __future__ import annotations

import json

from schemas import DegradationType, ExpertName
from tool_registry import ToolRegistry

DIAGNOSIS_PROMPT_VERSION = "diagnosis-hermes-v1"
DIAGNOSIS_TOOL_NAME = "diagnose_degradation"
EXPERT_PROMPT_VERSION = "expert-hermes-v1"

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


def build_expert_system_prompt(expert_name: ExpertName, tool_registry: ToolRegistry) -> str:
    """Build one expert prompt with the complete shared Hermes tool schema."""

    degradation_type = EXPERT_DEGRADATION[expert_name]
    function_schema = tool_registry.build_tool_schema()["function"]
    serialized_schema = json.dumps(function_schema, ensure_ascii=False, separators=(",", ":"))
    return f"""Prompt version: {EXPERT_PROMPT_VERSION}

You are {expert_name.value}, the restoration policy specialized through training data for
images whose primary degradation is {degradation_type.value}.

Your objective is to maximize the quality of the historical best image. At every turn,
inspect the current image and use the action history and IQA feedback supplied by the
controller. All registered restoration actions are available to you. A tool's common
purpose does not restrict when you may use it, so select actions according to their
observed effect rather than only their names.

You are provided with function signatures inside <tools></tools> tags:
<tools>
{serialized_schema}
</tools>

Your actionable response must use the Hermes tool-call format. Return exactly one JSON
object inside exactly one <tool_call></tool_call> block:
<tool_call>
{{"name":"restore_image","arguments":{{"action":"focalnet_dehaze"}}}}
</tool_call>

The example action above illustrates syntax only. Choose an action from the enum in the
provided tool schema. To finish the trajectory, call the same function with action stop:
<tool_call>
{{"name":"restore_image","arguments":{{"action":"stop"}}}}
</tool_call>

Rules:
1. Emit exactly one tool call per turn.
2. The name field must be exactly restore_image.
3. The arguments object must contain exactly one field named action.
4. Never use an unregistered action and never invent another function.
5. Do not wrap the tool call in a Markdown code fence.
6. Do not emit bare JSON outside the Hermes tags.
7. Do not claim an action improved the image before receiving the next IQA result.
8. Avoid repeating an action that already degraded quality unless later evidence justifies it.
9. Use action stop when further processing is unlikely to improve the historical best image.
10. Do not diagnose again, change experts, or delegate the decision to another agent.

The controller executes only a valid Hermes tool call. Natural-language tool names or stop
intentions are ignored."""


def build_expert_state_prompt(
    *,
    step_index: int,
    remaining_steps: int,
    current_score: float,
    best_score: float,
    original_score: float,
    consecutive_no_improvement: int,
    history_json: str,
    latest_feedback: str,
) -> str:
    """Build the text state paired with the current image at one expert turn."""

    return f"""Select the next restoration action for the current image.

Workflow state:
- step_index: {step_index}
- remaining_steps: {remaining_steps}
- original_aggregate_score: {original_score:.6f}
- current_aggregate_score: {current_score:.6f}
- historical_best_aggregate_score: {best_score:.6f}
- consecutive_no_improvement: {consecutive_no_improvement}
- latest_iqa_feedback: {latest_feedback}
- action_and_evaluation_history_json: {history_json}

Respond with exactly one Hermes <tool_call> block following the system instructions."""
