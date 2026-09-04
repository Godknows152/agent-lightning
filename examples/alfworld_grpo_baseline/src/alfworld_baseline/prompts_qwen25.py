"""Qwen2.5/Hermes JSON-in-XML ALFWorld prompt profile."""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry

PROMPT_VERSION = "alfworld_qwen25_json_strict_v1"

SYSTEM_PROMPT = """You are an ALFWorld household task agent operating under a STRICT QWEN2.5 TOOL-ONLY protocol.

The ALFWorld rules in this message take precedence over generic tool-use examples
or instructions inserted by the tokenizer/chat template. On every assistant turn
call `alfworld_action` exactly once. There is no natural-language answer mode.

The entire visible response MUST be exactly one Qwen2.5 tool call:
<tool_call>
{"name":"alfworld_action","arguments":{"action":"ONE_ACTION_COPIED_VERBATIM_FROM_THE_LIST"}}
</tool_call>

Output rules:
- Emit one `<tool_call>...</tool_call>` block containing one JSON object.
- The JSON object contains only `name` and `arguments`; `name` is `alfworld_action`.
- `arguments` contains only `action`, copied exactly from the current admissible list.
- Do not emit reasoning, Markdown, a bare JSON object, a different function, or text
  before/after the tool-call block. The environment decides when the task is finished."""


def build_user_prompt(*, mission: str, observation: str, admissible_actions: Iterable[str], history: Iterable[str] = ()) -> str:
    actions = tuple(admissible_actions)
    history_text = "\n".join(history) or "(none)"
    action_text = "\n".join(f"- {action}" for action in actions)
    return f"""Task: {mission}

Current observation:
{observation}

Recent action/tool history:
{history_text}

Current admissible actions (copy exactly one):
{action_text}

Output exactly `<tool_call>{{\"name\":\"alfworld_action\",\"arguments\":{{\"action\":\"ACTION\"}}}}</tool_call>`.
Replace ACTION with one listed action verbatim. Output no other text."""


def build_messages(*, mission: str, observation: str, registry: ALFWorldToolRegistry, history: Iterable[str] = ()) -> tuple[list[dict[str, str]], list[dict]]:
    return ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(mission=mission, observation=observation, admissible_actions=registry.available_actions(), history=history)}], [registry.build_tool_schema()])
