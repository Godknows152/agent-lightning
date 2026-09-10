"""Qwen3.5 v5: action history, thinking, plain-text actions; no tool schema."""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry

PROMPT_VERSION = "alfworld_qwen35_v5_action_history_text_thinking"
SYSTEM_PROMPT = ""

# Deliberately ignores `tools`: environment tools remain internal only.
QWEN35_ALFWORLD_CHAT_TEMPLATE = r"""
{{- '<|im_start|>system\nYou are solving an ALFWorld task. Reason inside <think>...</think> about the goal, current observation, and action history. Close </think>, then output exactly one line: Action: followed by one exact command from the current admissible actions. Do not emit XML, JSON, tool calls, additional actions, or explanations. The output restrictions apply only after </think>.\n<|im_end|>\n' }}
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
{%- if enable_thinking is defined and enable_thinking %}
{{- '<think>\n' }}
{%- else %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
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
    history: Iterable[str] = (),
) -> str:
    """Build v5 input; history contains past decision actions and execution status."""

    actions = tuple(str(action) for action in admissible_actions)
    action_text = "\n".join(f"- {action}" for action in actions)
    history_text = "\n".join(f"{i}. {entry}" for i, entry in enumerate(history, 1)) or "(none)"
    return f"""Task goal (not an executable action):
{mission}

Previous actions (chronological; not actions to execute again):
{history_text}

Current observation:
{_compact_observation(observation)}

Current admissible actions (the action value must be copied exactly from this list):
{action_text}

After thinking, output exactly one line: Action: <one exact current admissible action>."""


def build_messages(
    *, mission: str, observation: str, registry: ALFWorldToolRegistry
) -> tuple[list[dict[str, str]], list[dict]]:
    """Return v5 user input and no model-facing tool schema."""

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
    return messages, []
