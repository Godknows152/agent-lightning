# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GiGPO observation/action context with the Qwen3 XML tool-call protocol.

The environment supplies the authoritative observation and admissible action
list at every turn.  The first turn uses the compact no-history template;
later turns include the task, the recent observation/action pairs, and the
current state. Executable output keeps the baseline's Qwen3 XML format.
"""
from __future__ import annotations

from collections.abc import Iterable

from .tool_registry import ALFWorldToolRegistry
from .prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE

PROMPT_VERSION = "alfworld_gigpo_qwen3_xml_v1"
NONTHINKING_PROMPT_VERSION = "alfworld_gigpo_qwen3_xml_v1_nothinking"
SYSTEM_PROMPT = ""
HISTORY_LENGTH = 2

# Preserve the existing Qwen3 XML schema, thinking boundary and call guidance.
# The optional repeat guidance now refers to the bounded history we provide.
QWEN3_ALFWORLD_CHAT_TEMPLATE = QWEN35_ALFWORLD_CHAT_TEMPLATE.replace(
    "Previous actions", "the recent observation/action history"
).replace("already used in this episode", "already used in the recent history")

QWEN3_XML_CALL_FORMAT = """<tool_call>
<function=alfworld_action>
<parameter=action>EXACT_COMMAND</parameter>
</function>
</tool_call>"""

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
{decision_instruction}
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
{decision_instruction}
"""


def _format_actions(admissible_actions: Iterable[str]) -> str:
    return "\n ".join(f"'{str(action)}'" for action in admissible_actions if action != "help")


def build_user_prompt(
    *,
    mission: str,
    observation: str,
    admissible_actions: Iterable[str],
    history: Iterable[str] = (),
    enable_thinking: bool = True,
    avoid_repeated_actions: bool = False,
) -> str:
    """Build the current GiGPO-style ALFWorld prompt.

    History contains all completed observation/action pairs with absolute
    step numbers. Only the last two pairs enter the prompt; the step counter
    still counts the whole trajectory, as in GiGPO's SimpleMemory.
    """
    decision_instruction = (
        "First reason about the next step inside <think> </think> tags. "
        "Keep thinking to 1-2 short sentences, then close </think> and emit exactly one Qwen3 XML tool call."
        if enable_thinking else
        "Emit exactly one Qwen3 XML tool call directly, without reasoning or explanations."
    )
    decision_instruction += (
        "\nUse this exact format, replacing EXACT_COMMAND with one current admissible action verbatim:\n"
        + QWEN3_XML_CALL_FORMAT
        + "\nDo not emit JSON, additional calls, or text outside the required blocks."
    )
    if avoid_repeated_actions:
        decision_instruction += (
            "\nCheck the recent observation/action history before responding. "
            "Choose an admissible command not already used in that history when possible; do not invent commands."
        )
    actions = _format_actions(admissible_actions)
    full_history = tuple(str(item) for item in history)
    recent_history = full_history[-HISTORY_LENGTH:]
    if not recent_history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=observation,
            admissible_actions=actions,
            decision_instruction=decision_instruction,
        )

    action_history = "\n".join(recent_history)
    return ALFWORLD_TEMPLATE.format(
        task_description=mission,
        step_count=len(full_history),
        history_length=len(recent_history),
        action_history=action_history,
        current_step=len(full_history) + 1,
        current_observation=observation,
        admissible_actions=actions,
        decision_instruction=decision_instruction,
    )


def build_messages(
    *,
    mission: str,
    observation: str,
    registry: ALFWorldToolRegistry,
    history: Iterable[str] = (),
) -> tuple[list[dict[str, str]], list[dict]]:
    """Return a dataset-compatible user message and compact tool schema."""
    messages = [{
        "role": "user",
        "content": build_user_prompt(
            mission=mission,
            observation=observation,
            admissible_actions=registry.available_actions(),
            history=history,
        ),
    }]
    return messages, [registry.static_tool_schema()]
