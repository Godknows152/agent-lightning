"""ALFWorld-only old-VERL AgentLoop extension."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

@register("alfworld_tool_agent")
class ALFWorldToolAgentLoop(ToolAgentLoop):
    """ALFWorld loop with native rewards plus per-turn protocol penalties.

    The shared image-restoration loop is left unchanged.  This isolated
    subclass adds configurable one-time penalties for malformed/no-tool output,
    unknown tools, invalid JSON arguments, and inadmissible ALFWorld actions.
    Native ``env.step`` rewards are preserved unchanged.
    """

    NO_TOOL_CALL_PENALTY = -0.05
    INVALID_TOOL_CALL_PENALTY = -0.05
    MALFORMED_TOOL_CALL_PENALTY = -0.05
    FORMAT_PENALTY = -0.05
    FORGED_ROLE_AFTER_TOOL_CALL_PENALTY = 0.0
    TRAJECTORY_REPEAT_PENALTY_SCALE = 0.0

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Read the isolated tool configuration, while retaining safe defaults
        # for callers that construct the loop directly in tests.
        tool = self.tools.get("alfworld_action")
        tool_config = getattr(tool, "config", {}) or {}
        self.NO_TOOL_CALL_PENALTY = float(tool_config.get("no_tool_call_penalty", self.NO_TOOL_CALL_PENALTY))
        self.INVALID_TOOL_CALL_PENALTY = float(tool_config.get("unknown_tool_penalty", self.INVALID_TOOL_CALL_PENALTY))
        self.MALFORMED_TOOL_CALL_PENALTY = float(
            tool_config.get("malformed_tool_call_penalty", self.MALFORMED_TOOL_CALL_PENALTY)
        )
        self.FORMAT_PENALTY = float(tool_config.get("format_penalty", self.FORMAT_PENALTY))
        self.invalid_action_penalty = float(
            tool_config.get("invalid_action_penalty", self.FORMAT_PENALTY)
        )
        # The active Qwen2.5 configuration selects old-VERL's Hermes parser;
        # the local parser utility also accepts the Qwen3 XML form for replay.

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """Apply one protocol penalty when this turn emitted no parsed call."""

        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if getattr(agent_data, "data_source", "") != "alfworld" or agent_data.tool_calls:
            return state
        response_len = len(agent_data.response_ids)
        response_text = self.tokenizer.decode(agent_data.response_ids)
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
        text = self.tokenizer.decode(format_token_ids)
        start = text.find(self.TOOL_CALL_START_TOKEN)
        end = text.find(self.TOOL_CALL_END_TOKEN, start) if start >= 0 else -1
        if start < 0 or end < 0:
            # The generating-state handler classifies these as no-tool or
            # malformed, so do not charge a second format penalty here.
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

        keep_token_count = self._first_complete_tool_call_token_count(token_ids)
        if keep_token_count is None or keep_token_count >= len(token_ids):
            return token_ids, log_probs
        trimmed_log_probs = log_probs[:keep_token_count] if log_probs else None
        return token_ids[:keep_token_count], trimmed_log_probs

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData) -> tuple[Any, float, dict]:
        """Execute a call and apply ALFWorld-only protocol accounting."""
        try:
            decoded_arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError:
            penalty = self.FORMAT_PENALTY
            response_text = self.tokenizer.decode(agent_data.response_ids)
            self._record_invalid_tool_call(agent_data, reason="invalid_json_arguments", penalty=penalty)
            self._record_penalty(
                agent_data,
                reason="invalid_json_arguments",
                value=penalty,
                model_response=response_text,
            )
            return (
                ToolResponse(text=f"Error when executing tool: invalid JSON arguments for '{tool_call.name}'"),
                penalty,
                {"error": "invalid_json_arguments", "skip_tool_call_reward": True},
            )

        # JSON may be syntactically valid but still violate the declared
        # function schema. Treat this as one format error rather than allowing
        # the shared loop to turn it into an untracked execution exception.
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
        # The shared loop uses a restoration-specific reason for this generic
        # validator event. Rename it in-place to keep ALFWorld metrics clean.
        if isinstance(metrics, dict) and metrics.get("error") == "invalid_action":
            for record in reversed(agent_data.extra_fields.get("penalty_records", [])):
                if record.get("reason") == "invalid_restoration_action":
                    record["reason"] = "invalid_action"
                    break
        if getattr(agent_data, "data_source", "") == "alfworld" and isinstance(metrics, dict) and (bool(metrics.get("done")) or bool(metrics.get("truncated"))):
            agent_data.extra_fields["alfworld_environment_finished"] = True
            agent_data.extra_fields["alfworld_terminal_reason"] = "truncated" if metrics.get("truncated") else "done"
        return response, reward, metrics

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "data_source", "") == "alfworld" and agent_data.extra_fields.get("alfworld_environment_finished"):
            return AgentState.TERMINATED
        return state
