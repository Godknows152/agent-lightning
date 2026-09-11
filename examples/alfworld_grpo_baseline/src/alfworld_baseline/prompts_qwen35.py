"""Qwen3.5 v7: action history, brief thinking and compact XML tool schema."""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry

PROMPT_VERSION = "alfworld_qwen35_v7_compact_xml_history_thinking"
SFT_PROMPT_VERSION = "alfworld_qwen35_v7_compact_xml_history_nothinking_sft_v1"
SYSTEM_PROMPT = ""

# Render the single supported tool compactly; never serialize dynamic action enums.
QWEN35_ALFWORLD_CHAT_TEMPLATE = r"""
{%- if enable_thinking is defined and not enable_thinking %}
{{- '<|im_start|>system\nYou are solving an ALFWorld task. Choose only the next action, not a complete plan. Emit exactly one XML tool call directly. Do not emit reasoning, JSON, additional calls, or explanations. The action must exactly match one command from the current admissible actions in the user message.\n<|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>system\nYou are solving an ALFWorld task. Choose only the next action, not a complete plan. Inside <think>...</think>, use only 1-2 short sentences to connect the current observation and action history to the next useful action. Do not restate the task or observation, enumerate available actions, or speculate about a full solution. If information is missing, choose one admissible exploration action rather than prolonging reasoning. Close </think>, then emit exactly one XML tool call. No JSON, additional calls, or explanations after thinking. The action must exactly match one command from the current admissible actions in the user message.\n<|im_end|>\n' }}
{%- endif %}
{{- '<|im_start|>system\n<tools>\nalfworld_action(action: string): Execute one ALFWorld command from the current admissible actions.\n</tools>\nRequired call format:\n<tool_call>\n<function=alfworld_action>\n<parameter=action>EXACT_COMMAND</parameter>\n</function>\n</tool_call>\n<|im_end|>\n' }}
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
    enable_thinking: bool = True,
) -> str:
    """Build v7 input; the default thinking route remains RL-compatible."""

    actions = tuple(str(action) for action in admissible_actions)
    action_text = "\n".join(f"- {action}" for action in actions)
    history_text = "\n".join(f"{i}. {entry}" for i, entry in enumerate(history, 1)) or "(none)"
    decision_instruction = (
        "Decide only the next step. Keep thinking to 1-2 short sentences, then close </think> immediately.\n"
        "Do not repeat the observation or list candidate actions.\n"
        "After thinking, emit one XML tool call in the system-defined format. Copy one current admissible command exactly."
        if enable_thinking else
        "Decide only the next step. Emit one XML tool call directly in the system-defined format.\n"
        "Do not emit reasoning, explanations, the observation, or a list of candidate actions.\n"
        "Copy one current admissible command exactly."
    )
    return f"""Task goal (not an executable action):
{mission}

Previous actions (chronological; not actions to execute again):
{history_text}

Current observation:
{_compact_observation(observation)}

Current admissible actions (the action value must be copied exactly from this list):
{action_text}

{decision_instruction}"""


def build_messages(
    *, mission: str, observation: str, registry: ALFWorldToolRegistry
) -> tuple[list[dict[str, str]], list[dict]]:
    """Return user input and a compact static tool schema."""

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
    return messages, [registry.static_tool_schema()]
