"""ALFWorld-only old-VERL AgentLoop extension."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

from .prompts_qwen35 import QWEN35_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
from .tool_registry import ALFWorldToolRegistry
from .thinking import ThinkingToolParser, tool_output


@register("alfworld_tool_agent")
class ALFWorldToolAgentLoop(ToolAgentLoop):
    """ALFWorld loop with authoritative state prompts and bounded history.

    The full VERL trajectory remains available for policy-loss accounting, but
    every inference request uses a fresh current-state-only prompt. Previous
    observations, action lists, tool responses, and assistant turns are never
    sent as context for the next ALFWorld decision.
    """

    NO_TOOL_CALL_PENALTY = -0.05
    INVALID_TOOL_CALL_PENALTY = -0.05
    MALFORMED_TOOL_CALL_PENALTY = -0.05
    FORMAT_PENALTY = -0.05
    FORGED_ROLE_AFTER_TOOL_CALL_PENALTY = 0.0
    TRAJECTORY_REPEAT_PENALTY_SCALE = 0.0

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Replace the stock Qwen3.5 template only for Qwen3.5 ALFWorld runs.
        # The stock template contains a generic example_function_name and
        # normal-answer branch that compete with the dynamic schema. Qwen2.5
        # keeps its existing Hermes path unchanged.
        model_profile = os.environ.get("ALFWORLD_MODEL_PROFILE", "").lower()
        self._is_qwen35_alfworld = model_profile.startswith("qwen35") or self.processor is not None
        if self._is_qwen35_alfworld:
            self.apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
            self.apply_chat_template_kwargs["chat_template"] = QWEN35_ALFWORLD_CHAT_TEMPLATE
        self._thinking_enabled = self._is_qwen35_alfworld and bool(
            self.apply_chat_template_kwargs.get("enable_thinking", False)
        )
        if self._thinking_enabled:
            self.tool_parser = ThinkingToolParser(self.tool_parser, self.tokenizer)
        tool = self.tools.get("alfworld_action")
        tool_config = getattr(tool, "config", {}) or {}
        self.NO_TOOL_CALL_PENALTY = float(tool_config.get("no_tool_call_penalty", self.NO_TOOL_CALL_PENALTY))
        self.INVALID_TOOL_CALL_PENALTY = float(tool_config.get("unknown_tool_penalty", self.INVALID_TOOL_CALL_PENALTY))
        self.MALFORMED_TOOL_CALL_PENALTY = float(
            tool_config.get("malformed_tool_call_penalty", self.MALFORMED_TOOL_CALL_PENALTY)
        )
        self.FORMAT_PENALTY = float(tool_config.get("format_penalty", self.FORMAT_PENALTY))
        self.invalid_action_penalty = float(tool_config.get("invalid_action_penalty", self.FORMAT_PENALTY))

    @staticmethod
    def _mission_from_messages(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            match = re.search(
                r"(?:ALFWorld task|Task goal \(not an executable action\)|Task):\s*(.+?)(?:\n\n|$)",
                content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
            match = re.search(r"Your task is to:\s*(.+?)(?:\n\n|$)", content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return "Complete the ALFWorld task."

    @staticmethod
    def _mission_from_observation(observation: str) -> str | None:
        match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation)
        return match.group(1).strip() if match else None

    @staticmethod
    def _compact_observation(observation: str, mission: str) -> str:
        # ALFWorld repeats the task in the initial observation. Keep the outer
        # task field as the single task instruction and remove that duplicate.
        kept_lines = []
        for line in observation.splitlines():
            if line.strip().startswith("Your task is"):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _state_prompt_messages(
        self,
        *,
        mission: str,
        observation: str,
        actions: tuple[str, ...],
    ) -> list[dict[str, str]]:
        """Build the next request from the goal and latest state only."""

        return [
            {
                "role": "user",
                "content": build_user_prompt(
                    mission=mission,
                    observation=observation,
                    admissible_actions=actions,
                ),
            }
        ]

    async def _set_authoritative_initial_prompt(self, agent_data: AgentData) -> None:
        """Create the tool first and build turn zero from its actual state."""
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tool = active_tools.get("alfworld_action")
        if tool is None or not hasattr(tool, "get_state"):
            return
        instance_id = await self._get_or_create_tool_instance(
            "alfworld_action", tool, agent_data.tools_kwargs, agent_data
        )
        observation, actions = tool.get_state(instance_id)
        mission = self._mission_from_observation(observation) or self._mission_from_messages(agent_data.messages)
        agent_data.alfworld_mission = mission
        agent_data.alfworld_recent_history = []
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # Remove the dataset's possibly stale state and any custom system
        # message. The chat template supplies the canonical tool instructions.
        agent_data.messages = self._state_prompt_messages(
            mission=mission, observation=observation, actions=actions
        )
        # The action enum is part of the model-facing tool schema, not merely
        # prose in the prompt. Rebuild it from the authoritative environment
        # state. Execution still validates membership (no constrained decoder).
        agent_data._active_tool_schemas = [ALFWorldToolRegistry(actions).build_tool_schema()]

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        if getattr(agent_data, "data_source", "") == "alfworld":
            await self._set_authoritative_initial_prompt(agent_data)
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _rebuild_generation_prompt_after_tool(self, agent_data: AgentData) -> None:
        """Rebuild the next request from only the latest environment state."""
        if getattr(agent_data, "data_source", "") != "alfworld":
            return
        if agent_data.extra_fields.get("alfworld_environment_finished"):
            return
        tool_metrics = getattr(agent_data, "alfworld_last_tool_metrics", {}) or {}
        observation = str(tool_metrics.get("observation", agent_data.alfworld_current_observation))
        actions = tuple(
            str(a) for a in tool_metrics.get("admissible_commands", agent_data.alfworld_current_actions)
        )
        # Keep this compatibility field empty: history is retained internally
        # only for accounting/debugging and is never part of model context.
        agent_data.alfworld_recent_history = []
        agent_data.alfworld_current_observation = observation
        agent_data.alfworld_current_actions = actions
        # Refresh the enum after every environment transition. The next model
        # request and the tool parser must use the same latest action list.
        agent_data._active_tool_schemas = [ALFWorldToolRegistry(actions).build_tool_schema()]
        messages = self._state_prompt_messages(
            mission=agent_data.alfworld_mission,
            observation=observation,
            actions=actions,
        )
        # Only the next inference context is compacted. prompt_ids retains the
        # complete VERL trajectory and its response mask for training. Per-turn
        # contexts are replayed by FSDP for actor/ref logprobs and gradients.
        agent_data.messages = messages
        agent_data.generation_prompt_ids = await self.apply_chat_template(
            messages,
            tools=getattr(agent_data, "_active_tool_schemas", self.tool_schemas),
            images=None,
            videos=None,
        )

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Apply one protocol penalty when this turn emitted no parsed call."""
        is_alfworld = getattr(agent_data, "data_source", "") == "alfworld"
        if is_alfworld:
            prompt = list(agent_data.generation_prompt_ids or agent_data.prompt_ids)
            offset = len(agent_data.response_mask)
            previous_turns = agent_data.assistant_turns
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if is_alfworld and agent_data.assistant_turns > previous_turns:
            agent_data.extra_fields.setdefault("alfworld_turn_contexts", []).append({
                "prompt_ids": prompt,
                "response_ids": list(agent_data.response_ids),
                "response_offset": offset,
            })
        if getattr(agent_data, "data_source", "") != "alfworld" or agent_data.tool_calls:
            return state
        response_len = len(agent_data.response_ids)
        response_text, _ = tool_output(
            self.tokenizer.decode(agent_data.response_ids),
            enable_thinking=getattr(self, "_thinking_enabled", False),
        )
        if any(marker in response_text for marker in self.TOOL_CALL_ATTEMPT_MARKERS):
            self._apply_malformed_tool_call_penalty(agent_data, response_len)
        else:
            self._apply_no_tool_call_penalty(agent_data, response_len, reason="turn_without_tool_call")
        return state

    def _apply_tool_call_format_guardrails(
        self, agent_data: AgentData, token_ids: list[int], log_probs: list[float] | None
    ) -> tuple[list[int], list[float] | None]:
        """Penalize extra text or multiple complete calls exactly once."""
        if getattr(agent_data, "data_source", "") != "alfworld" or not token_ids:
            return super()._apply_tool_call_format_guardrails(agent_data, token_ids, log_probs)

        format_token_ids = self._strip_trailing_termination_tokens(token_ids)
        full_text = self.tokenizer.decode(format_token_ids)
        text, reasoning_offset = tool_output(
            full_text, enable_thinking=getattr(self, "_thinking_enabled", False)
        )
        start = text.find(self.TOOL_CALL_START_TOKEN)
        end = text.find(self.TOOL_CALL_END_TOKEN, start) if start >= 0 else -1
        if start < 0 or end < 0:
            return token_ids, log_probs

        call_end = end + len(self.TOOL_CALL_END_TOKEN)
        prefix = text[:start].strip()
        suffix = text[call_end:].strip()
        multiple_calls = text.count(self.TOOL_CALL_START_TOKEN) != 1 or text.count(self.TOOL_CALL_END_TOKEN) != 1
        if prefix or suffix or multiple_calls:
            penalty = self.FORMAT_PENALTY
            agent_data.tool_rewards.append(penalty)
            self._record_penalty(
                agent_data,
                reason="format_error",
                value=penalty,
                model_response=text,
                details={"extra_prefix": bool(prefix), "extra_suffix": bool(suffix), "multiple_calls": multiple_calls},
            )

        if getattr(self, "_thinking_enabled", False):
            # Keep the original reasoning + call IDs and aligned logprobs.
            # XML examples inside reasoning must not truncate the real call.
            target = reasoning_offset + call_end
            keep_token_count = next(
                (i for i in range(1, len(token_ids) + 1)
                 if len(self.tokenizer.decode(token_ids[:i])) >= target),
                None,
            )
        else:
            keep_token_count = self._first_complete_tool_call_token_count(token_ids)
        if keep_token_count is None or keep_token_count >= len(token_ids):
            return token_ids, log_probs
        trimmed_log_probs = log_probs[:keep_token_count] if log_probs else None
        return token_ids[:keep_token_count], trimmed_log_probs

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[Any, float, dict]:
        """Execute a call and apply ALFWorld-only protocol accounting."""
        if getattr(agent_data, "data_source", "") == "alfworld":
            # Failed parsing must not replay the previous turn's feedback.
            agent_data.alfworld_last_tool_metrics = {"action": "unknown", "error": "invalid_tool_call"}
        try:
            decoded_arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            penalty = self.FORMAT_PENALTY
            response_text = self.tokenizer.decode(agent_data.response_ids)
            self._record_invalid_tool_call(agent_data, reason="invalid_json_arguments", penalty=penalty)
            self._record_penalty(agent_data, reason="invalid_json_arguments", value=penalty, model_response=response_text)
            return (
                ToolResponse(text=f"Error when executing tool: invalid JSON arguments for '{tool_call.name}'"),
                penalty,
                {"error": "invalid_json_arguments", "skip_tool_call_reward": True},
            )

        if (
            not isinstance(decoded_arguments, dict)
            or set(decoded_arguments) != {"action"}
            or not isinstance(decoded_arguments.get("action"), str)
        ):
            penalty = self.FORMAT_PENALTY
            response_text = self.tokenizer.decode(agent_data.response_ids)
            self._record_invalid_tool_call(agent_data, reason="invalid_arguments_schema", penalty=penalty)
            self._record_penalty(
                agent_data,
                reason="format_error",
                value=penalty,
                model_response=response_text,
                details={"argument_keys": sorted(decoded_arguments) if isinstance(decoded_arguments, dict) else None},
            )
            return (
                ToolResponse(text=f"Error when executing tool: invalid arguments schema for '{tool_call.name}'"),
                penalty,
                {"error": "invalid_arguments_schema", "skip_tool_call_reward": True},
            )

        response, reward, metrics = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if isinstance(metrics, dict) and metrics.get("error") == "invalid_action":
            for record in reversed(agent_data.extra_fields.get("penalty_records", [])):
                if record.get("reason") == "invalid_restoration_action":
                    record["reason"] = "invalid_action"
                    break
        if getattr(agent_data, "data_source", "") == "alfworld" and isinstance(metrics, dict):
            agent_data.alfworld_last_tool_metrics = dict(metrics)
            if metrics.get("admissible_commands"):
                agent_data.alfworld_current_actions = tuple(str(a) for a in metrics["admissible_commands"])
            if metrics.get("observation") is not None:
                agent_data.alfworld_current_observation = str(metrics["observation"])
            if bool(metrics.get("done")) or bool(metrics.get("truncated")):
                agent_data.extra_fields["alfworld_environment_finished"] = True
                agent_data.extra_fields["alfworld_terminal_reason"] = (
                    "truncated" if metrics.get("truncated") else "done"
                )
        return response, reward, metrics

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "data_source", "") == "alfworld" and agent_data.extra_fields.get(
            "alfworld_environment_finished"
        ):
            return AgentState.TERMINATED
        return state
