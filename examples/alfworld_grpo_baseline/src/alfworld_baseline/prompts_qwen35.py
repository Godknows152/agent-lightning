"""Qwen3.5 ALFWorld state prompt and isolated tool template.

The model receives only the task goal, the latest environment observation, and
that state's admissible actions.  The tool-call wire format is owned by the
local template below; ALFWorld prompt text never repeats a second protocol.
"""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry

PROMPT_VERSION = "alfworld_qwen35_state_v4_current_state_only"
SYSTEM_PROMPT = ""

# Qwen3.5's stock template contains a generic ``example_function_name`` example
# and a normal-answer branch.  Both are harmful for this strict ALFWorld task.
# This template keeps the model's native XML call syntax while removing those
# competing instructions.  The dynamic schema remains the source of truth for
# the function name and the current action enum.
QWEN35_ALFWORLD_CHAT_TEMPLATE = r"""
{%- if tools and tools is iterable and tools is not mapping %}
{{- '<|im_start|>system\n# Tools\n\n<tools>' }}
{%- for tool in tools %}
{{- '\n' + (tool | tojson) }}
{%- endfor %}
{{- '\n</tools>\n\nCall exactly one provided function using the XML tool-call format. The action argument must be copied exactly from the current admissible action list. Do not answer in natural language.\n<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'system' %}
{{- '<|im_start|>system\n' + (message.content | string) + '<|im_end|>\n' }}
{%- elif message.role == 'user' %}
{{- '<|im_start|>user\n' + (message.content | string) + '<|im_end|>\n' }}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant\n' + (message.content | string) + '<|im_end|>\n' }}
{%- elif message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + (message.content | string) + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- raise_exception('Unexpected message role: ' ~ message.role) }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()


def _compact_observation(observation: str) -> str:
    """Remove ALFWorld's duplicated task line from the latest observation."""

    return "\n".join(
        line for line in str(observation).splitlines() if not line.strip().startswith("Your task is")
    ).strip()


def build_user_prompt(
    *,
    mission: str,
    observation: str,
    admissible_actions: Iterable[str],
) -> str:
    """Build a current-state-only prompt with an explicit goal/action split."""

    actions = tuple(str(action) for action in admissible_actions)
    action_text = "\n".join(f"- {action}" for action in actions)
    return f"""Task goal (not an executable action):
{mission}

Current observation:
{_compact_observation(observation)}

Current admissible actions (the action value must be copied exactly from this list):
{action_text}

Call `alfworld_action` exactly once. Copy one action character-for-character from the current admissible actions. Do not output the task goal, a plan, an explanation, or a natural-language answer."""


def build_messages(
    *, mission: str, observation: str, registry: ALFWorldToolRegistry
) -> tuple[list[dict[str, str]], list[dict]]:
    """Return the state-only user message and the authoritative dynamic schema."""

    messages = [
        {
            "role": "user",
            "content": build_user_prompt(
                mission=mission,
                observation=observation,
                admissible_actions=registry.available_actions(),
            ),
        }
    ]
    return messages, [registry.build_tool_schema()]
