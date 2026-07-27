# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import yaml
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class ToolActionMaskResult:
    """Result of aligning one parsed restoration action to generated token IDs."""

    mask: list[int]
    matched: bool
    action: str | None = None
    failure_reason: str | None = None


_ACTION_PARAMETER_PATTERN = re.compile(r"<parameter=action>(?P<raw_value>.*?)</parameter>", re.DOTALL)


def _tool_action_mask_failure(response_ids: list[int], reason: str) -> ToolActionMaskResult:
    """Return a fail-closed mask for an action-token alignment failure."""

    return ToolActionMaskResult(mask=[0] * len(response_ids), matched=False, failure_reason=reason)


def build_tool_action_token_mask(
    response_ids: list[int],
    tokenizer: Any,
    parsed_tool_calls: list[FunctionCall],
    allowed_actions: set[str],
) -> ToolActionMaskResult:
    """Align a valid Qwen3 ``restore_image`` action value to its generated tokens.

    The alignment is deliberately strict. It validates the parsed call, finds the
    structured ``<parameter=action>`` value span, and requires a lossless
    decode/encode round trip with tokenizer offsets. Any ambiguity returns an
    all-zero mask instead of falling back to a fuzzy action-name search.
    """

    if len(parsed_tool_calls) == 0:
        return _tool_action_mask_failure(response_ids, "no_parsed_tool_call")
    if len(parsed_tool_calls) != 1:
        return _tool_action_mask_failure(response_ids, "multiple_tool_calls")

    tool_call = parsed_tool_calls[0]
    if tool_call.name != "restore_image":
        return _tool_action_mask_failure(response_ids, "unexpected_function_name")

    try:
        arguments = json.loads(tool_call.arguments)
    except (json.JSONDecodeError, TypeError):
        return _tool_action_mask_failure(response_ids, "invalid_tool_arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"action"}:
        return _tool_action_mask_failure(response_ids, "missing_or_extra_action_arguments")

    parsed_action = arguments.get("action")
    if not isinstance(parsed_action, str) or not parsed_action:
        return _tool_action_mask_failure(response_ids, "missing_action_argument")
    if parsed_action not in allowed_actions:
        return _tool_action_mask_failure(response_ids, "action_not_in_active_schema")

    if not getattr(tokenizer, "is_fast", False):
        return _tool_action_mask_failure(response_ids, "tokenizer_not_fast")

    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    matches = list(_ACTION_PARAMETER_PATTERN.finditer(text))
    if len(matches) == 0:
        return _tool_action_mask_failure(response_ids, "missing_action_xml_span")
    if len(matches) != 1:
        return _tool_action_mask_failure(response_ids, "multiple_action_xml_spans")

    match = matches[0]
    raw_value = match.group("raw_value")
    action = raw_value.strip()
    if action != parsed_action:
        return _tool_action_mask_failure(response_ids, "parsed_action_text_mismatch")

    leading_whitespace = len(raw_value) - len(raw_value.lstrip())
    trailing_whitespace = len(raw_value) - len(raw_value.rstrip())
    action_start = match.start("raw_value") + leading_whitespace
    action_end = match.end("raw_value") - trailing_whitespace
    if action_start >= action_end:
        return _tool_action_mask_failure(response_ids, "empty_action_xml_span")

    try:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = list(encoding["input_ids"])
        offsets = list(encoding["offset_mapping"])
    except (KeyError, TypeError, ValueError, NotImplementedError):
        return _tool_action_mask_failure(response_ids, "offset_mapping_unavailable")

    if encoded_ids != response_ids:
        return _tool_action_mask_failure(response_ids, "token_id_roundtrip_mismatch")
    if len(offsets) != len(response_ids):
        return _tool_action_mask_failure(response_ids, "offset_length_mismatch")

    selected_indices: list[int] = []
    for index, raw_offset in enumerate(offsets):
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            return _tool_action_mask_failure(response_ids, "invalid_token_offset")
        token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
        overlaps_action = token_start < action_end and token_end > action_start
        contained_in_action = action_start <= token_start and token_end <= action_end
        if overlaps_action and not contained_in_action:
            return _tool_action_mask_failure(response_ids, "token_crosses_action_boundary")
        if contained_in_action and token_end > token_start:
            selected_indices.append(index)

    if not selected_indices:
        return _tool_action_mask_failure(response_ids, "empty_action_token_span")
    first_index, last_index = selected_indices[0], selected_indices[-1]
    if selected_indices != list(range(first_index, last_index + 1)):
        return _tool_action_mask_failure(response_ids, "non_contiguous_action_tokens")
    if tokenizer.decode(response_ids[first_index : last_index + 1], skip_special_tokens=False) != action:
        return _tool_action_mask_failure(response_ids, "decoded_action_token_mismatch")

    mask = [0] * len(response_ids)
    for index in selected_indices:
        mask[index] = 1
    return ToolActionMaskResult(mask=mask, matched=True, action=action)


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.tool_action_token_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        # IQA-only reward for successful image-restoration actions. This excludes
        # all non-image reward shaping.
        self.pure_image_restoration_rewards: list[float] = []
        # Reward or penalty returned by an executed stop action.
        self.stop_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.total_tool_calls = 0
        # Track action names across the trajectory for trajectory-level penalties
        self.action_history: list[str] = []

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.tool_instances: dict[str, str] = {}
        self.tool_instance_lock = asyncio.Lock()

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    NO_TOOL_CALL_PENALTY = -10.0
    MALFORMED_TOOL_CALL_PENALTY = -5.0
    FORGED_ROLE_AFTER_TOOL_CALL_PENALTY = -1.0
    TOOL_CALL_START_TOKEN = "<tool_call>"
    TOOL_CALL_END_TOKEN = "</tool_call>"
    TOOL_CALL_ATTEMPT_MARKERS = ("<tool_call", "</tool_call>", "<function=", "<parameter=")
    FORGED_ROLE_AFTER_TOOL_CALL_PATTERN = re.compile(
        r"(?im)(?:^|\n|<\|im_end\|>)\s*(?:<\|im_start\|>\s*)?" r"(?:user|assistant)\s*(?::|(?=\n|$)|(?=<\|channel\|>))"
    )
    # Per-occurrence penalty for choosing the same tool more than once in a trajectory.
    # E.g. scale=0.5 → using scunet 3 times costs -0.5*(3-1) = -1.0 total.
    TRAJECTORY_REPEAT_PENALTY_SCALE = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_generated_response_length = (
            self.rollout_config.multi_turn.max_generated_response_length or self.response_length
        )
        self.current_restoration_prompt = self._init_current_restoration_prompt(tool_config_path)

    @staticmethod
    def _generated_response_length(agent_data: AgentData) -> int:
        """Return the number of model tokens eligible for policy loss."""
        return sum(agent_data.response_mask)

    @staticmethod
    def _allowed_actions_from_tool_schemas(tool_schemas: list[Any]) -> set[str]:
        """Return the active ``restore_image.action`` enum values."""

        allowed_actions: set[str] = set()
        for raw_schema in tool_schemas:
            schema = raw_schema.model_dump() if hasattr(raw_schema, "model_dump") else raw_schema
            if not isinstance(schema, dict):
                continue
            function = schema.get("function")
            if not isinstance(function, dict) or function.get("name") != "restore_image":
                continue
            parameters = function.get("parameters")
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            action_schema = properties.get("action") if isinstance(properties, dict) else None
            action_enum = action_schema.get("enum") if isinstance(action_schema, dict) else None
            if isinstance(action_enum, list):
                allowed_actions.update(action for action in action_enum if isinstance(action, str) and action)
        return allowed_actions

    @staticmethod
    def _record_tool_action_mask_result(agent_data: AgentData, result: ToolActionMaskResult) -> None:
        """Record compact action-mask alignment diagnostics for one assistant turn."""

        diagnostics = agent_data.extra_fields.setdefault(
            "tool_action_mask_diagnostics",
            {
                "attempted_turns": 0,
                "matched_turns": 0,
                "matched_tokens": 0,
                "failure_counts": {},
            },
        )
        diagnostics["attempted_turns"] += 1
        agent_data.metrics["tool_action_mask_attempted_turns"] = (
            agent_data.metrics.get("tool_action_mask_attempted_turns", 0) + 1
        )
        if result.matched:
            diagnostics["matched_turns"] += 1
            matched_tokens = sum(result.mask)
            diagnostics["matched_tokens"] += matched_tokens
            agent_data.metrics["tool_action_mask_matched_turns"] = (
                agent_data.metrics.get("tool_action_mask_matched_turns", 0) + 1
            )
            agent_data.metrics["tool_action_mask_matched_tokens"] = (
                agent_data.metrics.get("tool_action_mask_matched_tokens", 0) + matched_tokens
            )
            return

        failure_reason = result.failure_reason or "unknown"
        failure_counts = diagnostics["failure_counts"]
        failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
        if failure_reason == "token_id_roundtrip_mismatch":
            metric_name = "tool_action_mask_roundtrip_failures"
        elif failure_reason == "action_not_in_active_schema":
            metric_name = "tool_action_mask_invalid_schema_failures"
        elif failure_reason in {"token_crosses_action_boundary", "decoded_action_token_mismatch"}:
            metric_name = "tool_action_mask_boundary_failures"
        else:
            return
        agent_data.metrics[metric_name] = agent_data.metrics.get(metric_name, 0) + 1

    @staticmethod
    def _record_penalty(
        agent_data: AgentData,
        *,
        reason: str,
        value: float,
        occurrences: int = 1,
        model_response: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record one real negative reward contribution with a stable cause."""

        value = float(value)
        if value >= 0.0:
            return

        record: dict[str, Any] = {
            "reason": reason,
            "value": value,
            "occurrences": max(1, int(occurrences)),
        }
        if agent_data.assistant_turns > 0:
            record["assistant_turn"] = int(agent_data.assistant_turns)
        if model_response is not None:
            record["model_response"] = model_response
        if details:
            record["details"] = details
        agent_data.extra_fields.setdefault("penalty_records", []).append(record)

    @staticmethod
    def _record_pure_image_restoration_reward(agent_data: AgentData, tool_metrics: dict[str, Any]) -> None:
        """Record the IQA-only reward contribution of one restoration action."""

        if getattr(agent_data, "data_source", "") != "restoration" or tool_metrics.get("action") == "stop":
            return
        base_reward = tool_metrics.get("base_reward")
        if base_reward is None:
            return
        try:
            value = float(base_reward)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric restoration base_reward: %r", base_reward)
            return
        if not math.isfinite(value):
            logger.warning("Ignoring non-finite restoration base_reward: %r", base_reward)
            return
        agent_data.pure_image_restoration_rewards.append(value)

    @staticmethod
    def _finalize_restoration_reward_components(agent_data: AgentData) -> dict[str, float]:
        """Partition the complete trajectory reward into three exhaustive components."""

        pure_image_reward = float(sum(agent_data.pure_image_restoration_rewards))
        stop_reward = float(sum(agent_data.stop_rewards))
        trajectory_reward = float(sum(agent_data.tool_rewards)) if agent_data.tool_rewards else 0.0
        # This signed residual includes every non-IQA, non-stop contribution,
        # including penalties, affinity shaping, and any reward clipping delta.
        other_penalty = trajectory_reward - pure_image_reward - stop_reward
        components = {
            "pure_image_restoration_reward": pure_image_reward,
            "stop_reward": stop_reward,
            "other_penalty": other_penalty,
        }
        agent_data.extra_fields.update(components)
        return components

    def _apply_no_tool_call_penalty(self, agent_data: AgentData, response_len: int, reason: str) -> None:
        """Apply one fixed penalty when a response contains no tool call."""

        agent_data.tool_rewards.append(self.NO_TOOL_CALL_PENALTY)
        agent_data.extra_fields["no_tool_call_penalty"] = self.NO_TOOL_CALL_PENALTY
        agent_data.extra_fields["no_tool_call_penalty_applied"] = True
        agent_data.extra_fields["no_tool_call_penalty_reason"] = reason
        agent_data.extra_fields["no_tool_call_response_length"] = response_len
        response_text = self.tokenizer.decode(agent_data.response_ids)
        self._record_penalty(
            agent_data,
            reason="no_tool_call",
            value=self.NO_TOOL_CALL_PENALTY,
            model_response=response_text,
            details={"trigger": reason, "response_length": response_len},
        )

    @staticmethod
    def _record_invalid_tool_call(
        agent_data: AgentData,
        *,
        reason: str,
        penalty: float | None = None,
        action: str | None = None,
        response_len: int | None = None,
    ) -> None:
        """Record an invalid parsed/tool-reported call without misclassifying it as no-tool."""

        agent_data.extra_fields["invalid_tool_call_penalty_applied"] = True
        agent_data.extra_fields["invalid_tool_call_penalty_reason"] = reason
        if penalty is not None:
            agent_data.extra_fields["invalid_tool_call_penalty"] = float(penalty)
        if action is not None:
            agent_data.extra_fields["invalid_tool_call_action"] = action
        if response_len is not None:
            agent_data.extra_fields["invalid_tool_call_response_length"] = response_len

    def _apply_malformed_tool_call_penalty(self, agent_data: AgentData, response_len: int) -> None:
        """Apply the heavy penalty to an attempted tool call whose XML could not be parsed."""

        agent_data.tool_rewards.append(self.MALFORMED_TOOL_CALL_PENALTY)
        response_text = self.tokenizer.decode(agent_data.response_ids)
        self._record_penalty(
            agent_data,
            reason="malformed_tool_call_xml",
            value=self.MALFORMED_TOOL_CALL_PENALTY,
            model_response=response_text,
            details={"response_length": response_len},
        )
        self._record_invalid_tool_call(
            agent_data,
            reason="malformed_tool_call_xml",
            penalty=self.MALFORMED_TOOL_CALL_PENALTY,
            response_len=response_len,
        )

    def _classify_restoration_turn_without_parsed_tool(
        self,
        agent_data: AgentData,
        *,
        generated_budget_exhausted: bool,
    ) -> None:
        """Classify an empty parser result as malformed XML or a genuine no-tool response."""

        response_len = len(agent_data.response_ids)
        response_text = self.tokenizer.decode(agent_data.response_ids)
        if any(marker in response_text for marker in self.TOOL_CALL_ATTEMPT_MARKERS):
            self._apply_malformed_tool_call_penalty(agent_data, response_len=response_len)
            return

        reason = (
            "generated_budget_exhausted_without_tool_call" if generated_budget_exhausted else "turn_without_tool_call"
        )
        self._apply_no_tool_call_penalty(agent_data, response_len=response_len, reason=reason)

    def _first_complete_tool_call_token_count(self, token_ids: list[int]) -> int | None:
        """Return the shortest generated prefix containing the first complete tool call."""

        for index in range(1, len(token_ids) + 1):
            text = self.tokenizer.decode(token_ids[:index])
            tool_call_start = text.find(self.TOOL_CALL_START_TOKEN)
            if tool_call_start >= 0 and text.find(self.TOOL_CALL_END_TOKEN, tool_call_start) >= 0:
                return index
        return None

    def _strip_trailing_termination_tokens(self, token_ids: list[int]) -> list[int]:
        """Remove trailing EOS/Pad tokens before checking tool-call format.

        SGLang includes the matched stop token in ``output_ids``. For Qwen this
        produces a raw suffix such as ``</tool_call><|im_end|>`` even though the
        model emitted no user-visible text after the tool call. Terminal
        EOS/Pad tokens are ignored before checking specifically for forged
        user/assistant role output.
        """

        termination_token_ids: set[int] = set()
        for attribute_name in ("eos_token_id", "pad_token_id"):
            token_id = getattr(self.tokenizer, attribute_name, None)
            if token_id is None:
                continue
            if isinstance(token_id, int):
                termination_token_ids.add(token_id)
            else:
                termination_token_ids.update(int(item) for item in token_id)

        format_token_ids = list(token_ids)
        while format_token_ids and format_token_ids[-1] in termination_token_ids:
            format_token_ids.pop()
        return format_token_ids

    def _apply_tool_call_format_guardrails(
        self, agent_data: AgentData, token_ids: list[int], log_probs: list[float] | None
    ) -> tuple[list[int], list[float] | None]:
        """Trim after the first tool call and penalize only forged user/assistant role output."""

        if not token_ids:
            return token_ids, log_probs
        if getattr(agent_data, "data_source", "") != "restoration":
            return token_ids, log_probs

        format_token_ids = self._strip_trailing_termination_tokens(token_ids)
        text = self.tokenizer.decode(format_token_ids)
        first_tool_call_start = text.find(self.TOOL_CALL_START_TOKEN)
        first_tool_call_end = (
            text.find(self.TOOL_CALL_END_TOKEN, first_tool_call_start) if first_tool_call_start >= 0 else -1
        )
        has_complete_tool_call = first_tool_call_start >= 0 and first_tool_call_end >= 0

        forged_role_match: re.Match[str] | None = None
        if has_complete_tool_call:
            after_tool_call = text[first_tool_call_end + len(self.TOOL_CALL_END_TOKEN) :]
            forged_role_match = self.FORGED_ROLE_AFTER_TOOL_CALL_PATTERN.search(after_tool_call)

        if forged_role_match is not None:
            penalty = self.FORGED_ROLE_AFTER_TOOL_CALL_PENALTY
            agent_data.tool_rewards.append(penalty)
            self._record_penalty(
                agent_data,
                reason="forged_user_or_assistant_role_after_tool_call",
                value=penalty,
                model_response=text,
                details={"matched_role_marker": forged_role_match.group(0).strip()},
            )

        keep_token_count = self._first_complete_tool_call_token_count(token_ids) if has_complete_tool_call else None
        if keep_token_count is None or keep_token_count >= len(token_ids):
            return token_ids, log_probs

        agent_data.extra_fields.setdefault("format_truncations", []).append(
            {
                "original_tokens": len(token_ids),
                "kept_tokens": keep_token_count,
                "dropped_tokens": len(token_ids) - keep_token_count,
            }
        )
        trimmed_log_probs = log_probs[:keep_token_count] if log_probs else None
        return token_ids[:keep_token_count], trimmed_log_probs

    @staticmethod
    def _count_image_markers_in_messages(messages: list[dict[str, Any]]) -> int:
        """Count image markers from message content for alignment diagnostics."""
        count = 0
        for message in messages or []:
            content = message.get("content")
            if isinstance(content, str):
                count += content.count("<image>")
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image":
                        count += 1
                    elif item.get("type") == "text" and isinstance(item.get("text"), str):
                        count += item["text"].count("<image>")
        return count

    @staticmethod
    def _safe_len(multimodal_data: Any) -> int:
        if multimodal_data is None:
            return 0
        if isinstance(multimodal_data, list):
            return len(multimodal_data)
        return 1

    def _log_image_alignment(self, stage: str, messages: list[dict[str, Any]], images: Any, request_id: str) -> None:
        marker_count = self._count_image_markers_in_messages(messages)
        image_count = self._safe_len(images)
        if marker_count != image_count:
            logger.warning(
                "Image marker/image count mismatch at %s: markers=%d images=%d request_id=%s",
                stage,
                marker_count,
                image_count,
                request_id,
            )
        else:
            logger.info(
                "Image marker/image alignment at %s: markers=%d images=%d request_id=%s",
                stage,
                marker_count,
                image_count,
                request_id,
            )

    def _init_current_restoration_prompt(self, tool_config_path: str | None) -> dict[str, Any] | None:
        """Load the current image-restoration prompt builders for restoration rollouts."""

        try:
            example_root = Path(__file__).resolve().parents[4]
            if str(example_root) not in sys.path:
                sys.path.insert(0, str(example_root))

            from agents.prompts import (  # type: ignore[import]
                build_expert_single_step_sft_system_prompt,
                build_expert_single_step_sft_user_prompt,
                build_expert_state_prompt,
                build_expert_system_prompt,
            )
            from schemas import ExpertName  # type: ignore[import]
            from tool_registry import ToolRegistry  # type: ignore[import]

            tool_config: dict[str, Any] = {}
            if tool_config_path:
                config_path = Path(tool_config_path).expanduser()
                if not config_path.is_absolute():
                    config_path = Path.cwd() / config_path
                if config_path.is_file():
                    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        tools = payload.get("tools")
                        if isinstance(tools, list) and tools and isinstance(tools[0], dict):
                            maybe_config = tools[0].get("config")
                            if isinstance(maybe_config, dict):
                                tool_config = maybe_config

            registry_path = Path(tool_config.get("tool_registry_path", example_root / "config" / "tools.yaml"))
            if not registry_path.is_absolute():
                registry_path = (example_root / registry_path).resolve()
            registry = ToolRegistry.from_yaml(registry_path)

            return {
                "ExpertName": ExpertName,
                "registry": registry,
                "registry_path": str(registry_path),
                "min_stop_tool_calls": int(
                    tool_config.get("prompt_min_stop_tool_calls", tool_config.get("stop_min_step", 3))
                ),
                "build_initial_system": build_expert_single_step_sft_system_prompt,
                "build_initial_user": build_expert_single_step_sft_user_prompt,
                "build_state": build_expert_state_prompt,
                "build_system": build_expert_system_prompt,
            }
        except Exception as exc:
            logger.warning("Current restoration prompt compatibility disabled: %s", exc)
            return None

    def _uses_current_restoration_prompt(self, agent_data: AgentData) -> bool:
        return bool(self.current_restoration_prompt) and getattr(agent_data, "data_source", "") == "restoration"

    def _restoration_expert_name(self, agent_data: AgentData) -> Any:
        prompt_config = self.current_restoration_prompt
        if prompt_config is None:
            return None

        ExpertName = prompt_config["ExpertName"]
        extra_info = getattr(agent_data, "sample_extra_info", {}) or {}
        raw_expert = extra_info.get("expert_name") or extra_info.get("expert") or "fog"
        aliases = {
            "fog": ExpertName.FOG,
            "fog_expert": ExpertName.FOG,
            "low_light": ExpertName.LOW_LIGHT,
            "low-light": ExpertName.LOW_LIGHT,
            "low_light_expert": ExpertName.LOW_LIGHT,
            "rain": ExpertName.RAIN,
            "rain_expert": ExpertName.RAIN,
            "snow": ExpertName.SNOW,
            "snow_expert": ExpertName.SNOW,
        }
        raw_expert = str(raw_expert)
        if raw_expert in aliases:
            return aliases[raw_expert]
        return ExpertName(raw_expert)

    def _current_image_user_content(self, text: str, include_image: bool) -> str | list[dict[str, Any]]:
        if self.processor is None or not include_image:
            return text
        image_text = text.removeprefix("<image>\n")
        return [{"type": "image"}, {"type": "text", "text": image_text}]

    def _current_history_feedback_text(self, agent_data: AgentData) -> str:
        entries = getattr(agent_data, "current_prompt_history", [])
        if not entries:
            return "No historical restoration actions have been executed yet."
        return "Historical tool feedback: " + " ".join(entries)

    def _build_current_restoration_decision_prompt(
        self, agent_data: AgentData
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prompt_config = self.current_restoration_prompt
        if prompt_config is None:
            return agent_data.messages, getattr(agent_data, "_active_tool_schemas", self.tool_schemas)

        registry = prompt_config["registry"]
        expert_name = self._restoration_expert_name(agent_data)
        step_index = len(getattr(agent_data, "current_prompt_history", []))
        min_stop_tool_calls = int(prompt_config["min_stop_tool_calls"])
        completed_restoration_actions = step_index

        if step_index == 0:
            include_stop = False
            system_prompt = prompt_config["build_initial_system"](expert_name, registry)
            user_prompt = prompt_config["build_initial_user"]()
            include_image = True
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._current_image_user_content(user_prompt, include_image)},
            ]
        else:
            include_stop = completed_restoration_actions >= min_stop_tool_calls
            user_prompt = prompt_config["build_state"](history_feedback=self._current_history_feedback_text(agent_data))
            if include_stop:
                user_prompt += (
                    "\n\nThe stop action is now available. Use action stop only when further processing is unlikely "
                    "to improve the historical best image."
                )
            else:
                remaining_tool_calls = max(0, min_stop_tool_calls - completed_restoration_actions)
                user_prompt += (
                    "\n\nThe stop action is not available yet. Continue with one non-stop restoration action; "
                    f"{remaining_tool_calls} more restoration tool call(s) are required before stop becomes available."
                )
            include_image = False
            messages = [
                {"role": "user", "content": self._current_image_user_content(user_prompt, include_image)},
            ]

        tool_schemas = [registry.build_tool_schema(include_stop=include_stop)]
        return messages, tool_schemas

    @staticmethod
    def _append_current_history_feedback(agent_data: AgentData, tool_name: str, tool_metrics: dict[str, Any]) -> None:
        action = str(tool_metrics.get("action") or tool_name)
        if action == "stop":
            return
        step_value = tool_metrics.get("step")
        try:
            step_index = max(0, int(step_value) - 1)
        except Exception:
            step_index = len(getattr(agent_data, "current_prompt_history", []))

        aggregate_score = tool_metrics.get("aggregate_score")
        if isinstance(aggregate_score, (int, float)):
            score_text = f"IQA aggregate_score={float(aggregate_score):.4f}"
        else:
            score_text = "IQA score unavailable"
        entry = f"Step {step_index}: selected action {action}; {score_text}."
        history = getattr(agent_data, "current_prompt_history", None)
        if history is None:
            history = []
            agent_data.current_prompt_history = history
        history.append(entry)

    async def _get_or_create_tool_instance(
        self,
        tool_name: str,
        tool: Any,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
    ) -> str:
        async with agent_data.tool_instance_lock:
            instance_id = agent_data.tool_instances.get(tool_name)
            if instance_id is not None:
                return instance_id

            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            agent_data.tool_instances[tool_name] = instance_id
            return instance_id

    async def _release_tool_instances(self, agent_data: AgentData) -> None:
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        for tool_name, instance_id in list(agent_data.tool_instances.items()):
            tool = active_tools.get(tool_name)
            if tool is None:
                continue
            try:
                await tool.release(instance_id)
            except Exception as e:
                logger.warning("Error when releasing tool '%s' instance '%s': %s", tool_name, instance_id, e)
        agent_data.tool_instances.clear()

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], round_barrier=None, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        agent_data.data_source = kwargs.get("data_source", "")
        agent_data.sample_extra_info = extra_info
        agent_data.current_prompt_history = []
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        try:
            # State machine loop
            state = AgentState.PENDING
            generation_round = 0  # Track which generation round we're on
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    # Before submitting the generation request, wait for the
                    # round barrier if this is round 2+ (i.e., after at least
                    # one tool call has been processed).  Round 1 starts
                    # naturally for all trajectories at the same time, so no
                    # barrier is needed there.
                    if round_barrier is not None and generation_round > 0:
                        await round_barrier.wait_for_next_round()
                    generation_round += 1
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED

            # Final reward shaping guardrail for restoration tasks.
            # This ensures no-tool penalty is applied regardless of how the loop terminated
            # (e.g., max response length, max turns, or regular stop).
            if getattr(agent_data, "data_source", "") == "restoration":
                response_len = len(agent_data.response_ids or [])
                no_tool_call = agent_data.total_tool_calls == 0

                failure_already_classified = agent_data.extra_fields.get(
                    "no_tool_call_penalty_applied", False
                ) or agent_data.extra_fields.get("invalid_tool_call_penalty_applied", False)
                if no_tool_call and not failure_already_classified:
                    self._apply_no_tool_call_penalty(
                        agent_data,
                        response_len=response_len,
                        reason="trajectory_ended_without_tool_call",
                    )
                # Expose decomposed reward parts for easier diagnosis in logs.
                # Trajectory-level duplicate penalty: penalize choosing the same
                # tool multiple times across the whole trajectory (not just
                # consecutively).  The penalty grows with the number of repeats:
                #   penalty = -scale * sum(count - 1 for each tool with count > 1)
                # i.e. each extra occurrence of a repeated tool costs `scale`.
                # "stop" is excluded — calling stop multiple times is impossible
                # because the loop terminates on the first stop.
                action_counts: dict[str, int] = {}
                for a in agent_data.action_history:
                    if a != "stop":
                        action_counts[a] = action_counts.get(a, 0) + 1
                total_repeats = sum(c - 1 for c in action_counts.values() if c > 1)
                trajectory_repeat_penalty = -self.TRAJECTORY_REPEAT_PENALTY_SCALE * total_repeats
                if trajectory_repeat_penalty < 0:
                    agent_data.tool_rewards.append(trajectory_repeat_penalty)
                    agent_data.extra_fields["trajectory_repeat_penalty"] = trajectory_repeat_penalty
                    agent_data.extra_fields["trajectory_repeat_counts"] = {
                        a: c for a, c in action_counts.items() if c > 1
                    }
                    self._record_penalty(
                        agent_data,
                        reason="repeated_restoration_action",
                        value=trajectory_repeat_penalty,
                        occurrences=total_repeats,
                        details={"repeated_action_counts": agent_data.extra_fields["trajectory_repeat_counts"]},
                    )

                components = self._finalize_restoration_reward_components(agent_data)
                pure_image_reward = components["pure_image_restoration_reward"]
                stop_reward = components["stop_reward"]
                other_penalty = components["other_penalty"]
                trajectory_reward = float(sum(agent_data.tool_rewards)) if agent_data.tool_rewards else 0.0
                agent_data.extra_fields["reward_components"] = {
                    "no_tool_call_penalty": float(agent_data.extra_fields.get("no_tool_call_penalty", 0.0)),
                    "invalid_tool_call_penalty": float(agent_data.extra_fields.get("invalid_tool_call_penalty", 0.0)),
                    "response_length": response_len,
                    "trajectory_repeat_penalty": trajectory_repeat_penalty,
                    "pure_image_restoration_reward": pure_image_reward,
                    "stop_reward": stop_reward,
                    "other_penalty": other_penalty,
                    "tool_reward_sum": trajectory_reward,
                }

            # Finalize output
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            multi_modal_data = {}
            if agent_data.image_data is not None:
                multi_modal_data["images"] = agent_data.image_data
            if agent_data.video_data is not None:
                multi_modal_data["videos"] = agent_data.video_data

            output: AgentLoopOutput = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=agent_data.response_mask[: self.response_length],
                tool_action_token_mask=agent_data.tool_action_token_mask[: self.response_length],
                multi_modal_data=multi_modal_data,
                response_logprobs=(
                    agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
                ),
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                metrics=agent_data.metrics,
                routed_experts=agent_data.routed_experts,
                extra_fields=agent_data.extra_fields,
            )
            output.extra_fields.update(
                {
                    "turn_scores": agent_data.turn_scores,
                    "tool_rewards": agent_data.tool_rewards,
                    "action_history": list(agent_data.action_history),
                    "pure_image_restoration_rewards": agent_data.pure_image_restoration_rewards,
                    "stop_rewards": agent_data.stop_rewards,
                }
            )
            return output
        finally:
            # Always depart from the round barrier, even if an exception occurred.
            # Without this, remaining trajectories would wait forever for a
            # departed trajectory that never arrives at the barrier.
            if round_barrier is not None:
                round_barrier.depart()
            await self._release_tool_instances(agent_data)

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        if self._uses_current_restoration_prompt(agent_data):
            agent_data.messages, schemas = self._build_current_restoration_decision_prompt(agent_data)
            agent_data._active_tool_schemas = schemas
        self._log_image_alignment(
            stage="pending_before_apply_chat_template",
            messages=agent_data.messages,
            images=agent_data.image_data,
            request_id=agent_data.request_id,
        )
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        generated_tokens = self._generated_response_length(agent_data)
        remaining_generated_tokens = self.max_generated_response_length - generated_tokens
        if not ignore_termination and remaining_generated_tokens <= 0:
            return AgentState.TERMINATED

        generation_params = dict(sampling_params)
        generation_params["max_new_tokens"] = max(0, remaining_generated_tokens)
        generation_params.pop("max_generated_response_length", None)
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=generation_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        response_ids, response_logprobs = self._apply_tool_call_format_guardrails(
            agent_data,
            output.token_ids,
            output.log_probs,
        )
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        tool_action_mask_start = len(agent_data.tool_action_token_mask)
        agent_data.tool_action_token_mask += [0] * len(agent_data.response_ids)
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Only model-generated tokens consume the generation budget. Zero-mask
        # tool observations and reconstructed prompts still count toward the
        # trajectory tensor capacity checked after they are appended.
        generated_budget_exhausted = (
            not ignore_termination and self._generated_response_length(agent_data) >= self.max_generated_response_length
        )

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        if getattr(agent_data, "data_source", "") == "restoration":
            active_tool_schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            allowed_actions = self._allowed_actions_from_tool_schemas(active_tool_schemas)
            if self.tool_parser_name == "qwen3_coder":
                mask_result = build_tool_action_token_mask(
                    agent_data.response_ids,
                    self.tokenizer,
                    agent_data.tool_calls,
                    allowed_actions,
                )
            else:
                mask_result = _tool_action_mask_failure(
                    agent_data.response_ids,
                    f"unsupported_tool_parser:{self.tool_parser_name}",
                )
            agent_data.tool_action_token_mask[
                tool_action_mask_start : tool_action_mask_start + len(agent_data.response_ids)
            ] = mask_result.mask
            self._record_tool_action_mask_result(agent_data, mask_result)

        # Check soft termination conditions (max turns) AFTER tool call extraction.
        # This ensures the final assistant turn's tool call (e.g. "stop") is still
        # processed and rewarded, rather than being silently discarded.
        at_max_assistant_turns = self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns
        at_max_user_turns = self.max_user_turns and agent_data.user_turns >= self.max_user_turns

        if agent_data.tool_calls:
            # If we are at max turns, still execute the tool call (e.g. "stop")
            # so its reward is computed, but mark that the loop should terminate
            # after processing the tool response.
            if at_max_assistant_turns or at_max_user_turns or generated_budget_exhausted:
                agent_data.extra_fields["_terminate_after_tool"] = True
            return AgentState.PROCESSING_TOOLS
        else:
            # No tool call — terminate.
            # Enforce tool usage per assistant turn for restoration.
            # Malformed XML is an invalid attempted call; only responses without
            # any tool-call marker are classified as genuine no-tool turns.
            if getattr(agent_data, "data_source", "") == "restoration":
                self._classify_restoration_turn_without_parsed_tool(
                    agent_data,
                    generated_budget_exhausted=generated_budget_exhausted,
                )
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        returned_images_this_turn: list[Any] = []
        use_latest_image_context = getattr(agent_data, "data_source", "") == "restoration"

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)
        agent_data.total_tool_calls += len(tool_call_names)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Check if any tool response signals a stop action — this must terminate the loop.
        # We detect it via the metrics dict returned by _call_tool (action == "stop").
        stop_triggered = any(res.get("action") == "stop" for _, _, res in responses)

        use_current_restoration_prompt = self._uses_current_restoration_prompt(agent_data)

        # Process tool responses and update multi_modal_data.
        for tool_call_name, (tool_response, tool_reward, tool_metrics) in zip(tool_call_names, responses, strict=False):
            if not use_current_restoration_prompt:
                # Create message from tool response
                if tool_response.image or tool_response.video:
                    # Multi-modal content with structured format
                    if not getattr(self.processor, "image_processor", None):
                        raise ValueError(
                            "Multimedia data can only be processed by `processor`, but the processor is None. "
                            "This error is often caused if you are using a LLM model but your tool returns multimodal "
                            "data. Plase use a vlm as the base model."
                        )
                    content = []
                    if tool_response.image and not use_latest_image_context:
                        content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    if tool_response.image and use_latest_image_context and not content:
                        content.append({"type": "text", "text": "Current image updated."})
                    message = {"role": "tool", "content": content}
                else:
                    # Text-only content
                    message = {"role": "tool", "content": tool_response.text or ""}

                add_messages.append(message)

            # Handle image data
            if tool_response.image:
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            returned_images_this_turn.append(img)
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        returned_images_this_turn.append(tool_response.image)

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                tool_metrics = tool_metrics or {}
                self._record_pure_image_restoration_reward(agent_data, tool_metrics)
                reward_value = float(tool_reward)
                agent_data.tool_rewards.append(reward_value)
                # Record action name for trajectory-level duplicate penalty
                action_name = tool_metrics.get("action")
                if getattr(agent_data, "data_source", "") == "restoration" and action_name == "stop":
                    agent_data.stop_rewards.append(reward_value)
                if action_name:
                    agent_data.action_history.append(action_name)
                if use_current_restoration_prompt and not tool_metrics.get("error"):
                    self._append_current_history_feedback(agent_data, tool_call_name, tool_metrics)

        if returned_images_this_turn:
            if use_latest_image_context:
                # Preserve one visual slot and bind it to the latest restoration
                # result. Later decision prompts carry textual feedback only, so
                # no additional image tokens enter the training trajectory.
                agent_data.image_data = self._limit_image_pixels([returned_images_this_turn[-1]], self.max_image_pixels)
            else:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]
                for img in returned_images_this_turn:
                    agent_data.image_data.append(img)

        # If the model called stop, terminate the loop now so the trajectory ends
        # at the model's chosen stopping point rather than being forced by max_turns.
        if stop_triggered:
            return AgentState.TERMINATED

        # If we reached max turns before this tool call, the tool was still executed
        # (so its reward is counted), but we should not continue the loop.
        if agent_data.extra_fields.get("_terminate_after_tool"):
            return AgentState.TERMINATED

        if use_current_restoration_prompt:
            add_messages, schemas = self._build_current_restoration_decision_prompt(agent_data)
            agent_data._active_tool_schemas = schemas
            agent_data.messages.extend(add_messages)
            images = None
            videos = None
            self._log_image_alignment(
                stage="processing_tools_current_prompt_before_apply_chat_template",
                messages=add_messages,
                images=images,
                request_id=agent_data.request_id,
            )
            response_ids = await self.apply_chat_template(
                add_messages,
                tools=None,
                images=images,
                videos=videos,
                remove_system_prompt=False,
            )
        else:
            agent_data.messages.extend(add_messages)

            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                if use_latest_image_context:
                    # Tool-returned restoration images replace the current visual
                    # state instead of adding new image markers to the text history.
                    # The prompt keeps the original image slot and binds it to the
                    # latest image for the next generation round, while text
                    # feedback preserves action history.
                    images = None
                else:
                    # Pass None when there are no new images / videos to stay
                    # compatible with downstream image processing logic.
                    images = returned_images_this_turn if returned_images_this_turn else None
                videos = None
                self._log_image_alignment(
                    stage="processing_tools_tool_response_before_apply_chat_template",
                    messages=add_messages,
                    images=images,
                    request_id=agent_data.request_id,
                )
                response_ids = await self.apply_chat_template(
                    add_messages,
                    images=images,
                    videos=videos,
                    remove_system_prompt=True,
                )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        self._log_image_alignment(
            stage="processing_tools_after_message_and_image_update",
            messages=agent_data.messages,
            images=agent_data.image_data,
            request_id=agent_data.request_id,
        )

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        agent_data.tool_action_token_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = active_tools[tool_name]
            instance_id = await self._get_or_create_tool_instance(tool_name, tool, tools_kwargs, agent_data)
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            if isinstance(res, dict) and res.get("error") == "invalid_action":
                action = str(tool_args.get("action", ""))
                self._record_invalid_tool_call(
                    agent_data,
                    reason="invalid_action",
                    penalty=tool_reward,
                    action=action,
                )
                self._record_penalty(
                    agent_data,
                    reason="invalid_restoration_action",
                    value=tool_reward,
                    model_response=self.tokenizer.decode(agent_data.response_ids),
                    details={"action": action},
                )
        except json.JSONDecodeError as e:
            logger.warning("Error when executing tool: invalid tool arguments for '%s': %s", tool_call.name, e)
            self._record_invalid_tool_call(agent_data, reason="invalid_json_arguments")
            return (
                ToolResponse(
                    text=f"Error when executing tool: invalid tool arguments for '{tool_call.name}'",
                ),
                0.0,
                {"skip_tool_call_reward": True},
            )
        except KeyError:
            logger.warning(f"Error when executing tool: unknown tool '{tool_call.name}'")
            agent_data.extra_fields["unknown_tool_call_penalty_applied"] = True
            agent_data.extra_fields["unknown_tool_call_penalty_reason"] = "unknown_tool_name"
            self._record_penalty(
                agent_data,
                reason="unknown_tool_name",
                value=-1.0,
                model_response=self.tokenizer.decode(agent_data.response_ids),
                details={"tool_name": tool_call.name},
            )
            return (
                ToolResponse(
                    text=f"Error when executing tool: unknown tool '{tool_call.name}'",
                ),
                -1.0,
                {"skip_tool_call_reward": True},
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            agent_data.extra_fields["tool_execution_error_penalty_applied"] = True
            agent_data.extra_fields["tool_execution_error_penalty_reason"] = type(e).__name__
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {"skip_tool_call_reward": True},
            )

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
