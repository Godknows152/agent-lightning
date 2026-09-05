"""Qwen3.5 ALFWorld prompt profile.

The generic Qwen3.5 chat template owns the tool-call wire format.  This profile
only describes the ALFWorld state, so it does not duplicate or contradict the
chat template's tool instructions.
"""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry

PROMPT_VERSION = "alfworld_qwen35_state_v3"

# Do not restate XML/tool-call syntax here.  The tokenizer/chat template emits
# the canonical Qwen3.5 tool instructions and schema.
SYSTEM_PROMPT = ""


def build_user_prompt(
    *,
    mission: str,
    observation: str,
    admissible_actions: Iterable[str],
    history: Iterable[str] = (),
) -> str:
    actions = tuple(admissible_actions)
    history_text = "\n".join(history) or "(none)"
    action_text = "\n".join(f"- {action}" for action in actions)
    return f"""ALFWorld task:
{mission}

Current observation (use this latest state):
{observation}

Recent action/tool history (latest entries only):
{history_text}

Current admissible actions (copy exactly one from this latest list):
{action_text}

Use the provided alfworld_action tool exactly once and set its action to one
item copied verbatim from the current admissible actions above. Do not answer
with a natural-language task response."""


def build_messages(
    *, mission: str, observation: str, registry: ALFWorldToolRegistry, history: Iterable[str] = ()
) -> tuple[list[dict[str, str]], list[dict]]:
    messages = [{"role": "user", "content": build_user_prompt(
        mission=mission,
        observation=observation,
        admissible_actions=registry.available_actions(),
        history=history,
    )}]
    return messages, [registry.build_tool_schema()]
