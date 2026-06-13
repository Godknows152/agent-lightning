"""Strict Hermes restoration expert inference through an OpenAI-compatible VLM."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Protocol, cast

from config import ExpertVLMSettings
from exceptions import UnknownActionError
from openai import APITimeoutError, OpenAI
from schemas import (
    ExpertDecisionRecord,
    ExpertDecisionSource,
    ExpertName,
    ExpertParseStatus,
    RestorationTrajectoryState,
    ValidationStatus,
)
from tool_registry import RESTORE_FUNCTION_NAME, ToolRegistry

from .prompts import build_expert_state_prompt, build_expert_system_prompt


class _DumpableResponse(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


class _CompletionsClient(Protocol):
    def create(self, **request: Any) -> _DumpableResponse: ...


class _ChatNamespace(Protocol):
    completions: _CompletionsClient


class _OpenAICompatibleClient(Protocol):
    chat: _ChatNamespace


def _tool_call_from_raw_hermes(
    raw_response: str | None,
) -> tuple[ExpertParseStatus | None, dict[str, Any] | None, str | None, str | None]:
    """Extract exactly one raw Hermes restoration call."""

    if raw_response is None or not raw_response.strip():
        return ExpertParseStatus.EMPTY_RESPONSE, None, None, "expert returned no tool call or assistant content"
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_response, flags=re.DOTALL)
    if not matches:
        return ExpertParseStatus.INVALID_TOOL_CALL, None, None, "no Hermes <tool_call> block found"
    if len(matches) != 1:
        return ExpertParseStatus.MULTIPLE_TOOL_CALLS, None, None, "expert must contain exactly one tool call"
    try:
        payload: object = json.loads(matches[0])
    except json.JSONDecodeError as error:
        return ExpertParseStatus.INVALID_JSON, None, None, f"invalid Hermes tool-call JSON: {error}"
    if not isinstance(payload, dict):
        return ExpertParseStatus.INVALID_TOOL_CALL, None, None, "Hermes tool call must contain a JSON object"
    return None, cast(dict[str, Any], payload), None, None


def _tool_call_from_openai(
    tool_calls_value: object | None,
) -> tuple[ExpertParseStatus | None, dict[str, Any] | None, str | None, str | None]:
    """Normalize vLLM's parsed OpenAI tool-call representation."""

    if not isinstance(tool_calls_value, list) or not tool_calls_value:
        return None, None, None, None
    tool_calls = cast(list[object], tool_calls_value)
    if len(tool_calls) != 1:
        return ExpertParseStatus.MULTIPLE_TOOL_CALLS, None, None, "expert must contain exactly one tool call"
    tool_call_value = tool_calls[0]
    if not isinstance(tool_call_value, dict):
        return ExpertParseStatus.INVALID_TOOL_CALL, None, None, "parsed tool call must be an object"
    tool_call = cast(dict[str, Any], tool_call_value)
    function_value: object = tool_call.get("function")
    if not isinstance(function_value, dict):
        return ExpertParseStatus.INVALID_TOOL_CALL, None, None, "parsed tool call is missing function"
    function = cast(dict[str, Any], function_value)
    arguments_value: object = function.get("arguments")
    if isinstance(arguments_value, str):
        try:
            arguments_value = json.loads(arguments_value)
        except json.JSONDecodeError as error:
            return ExpertParseStatus.INVALID_JSON, None, None, f"invalid function arguments JSON: {error}"
    tool_call_id = tool_call.get("id") if isinstance(tool_call.get("id"), str) else None
    return None, {"name": function.get("name"), "arguments": arguments_value}, tool_call_id, None


def parse_expert_response(
    raw_response: str | None,
    tool_registry: ToolRegistry,
    tool_calls: object | None = None,
) -> tuple[ExpertParseStatus, dict[str, Any] | None, str | None, str | None, str | None]:
    """Validate one parsed or raw Hermes restore_image tool call."""

    status, payload, tool_call_id, error = _tool_call_from_openai(tool_calls)
    if status is not None:
        return status, payload, None, tool_call_id, error
    if payload is None:
        status, payload, tool_call_id, error = _tool_call_from_raw_hermes(raw_response)
        if status is not None:
            return status, payload, None, tool_call_id, error
    if payload is None:
        return ExpertParseStatus.EMPTY_RESPONSE, None, None, None, "expert returned no restoration tool call"

    if payload.get("name") != RESTORE_FUNCTION_NAME:
        return ExpertParseStatus.UNKNOWN_FUNCTION, payload, None, tool_call_id, "unexpected expert function name"
    arguments_value: object = payload.get("arguments")
    if not isinstance(arguments_value, dict):
        return ExpertParseStatus.INVALID_TOOL_CALL, payload, None, tool_call_id, "tool arguments must be an object"
    arguments = cast(dict[str, Any], arguments_value)
    if "action" not in arguments:
        return ExpertParseStatus.MISSING_FIELD, payload, None, tool_call_id, "missing required field: action"
    extra_fields = sorted(set(arguments) - {"action"})
    if extra_fields:
        return (
            ExpertParseStatus.INVALID_TOOL_CALL,
            payload,
            None,
            tool_call_id,
            f"unexpected argument fields: {', '.join(extra_fields)}",
        )
    action = arguments.get("action")
    if not isinstance(action, str) or not action:
        return ExpertParseStatus.MISSING_FIELD, payload, None, tool_call_id, "action must be a non-empty string"
    try:
        tool_registry.validate_action(action)
    except UnknownActionError as parse_error:
        return ExpertParseStatus.UNKNOWN_ACTION, payload, None, tool_call_id, str(parse_error)
    return ExpertParseStatus.VALID, payload, action, tool_call_id, None


def _int_list(value: Any) -> list[int] | None:
    value_object: object = value
    items = cast(list[object], value_object) if isinstance(value_object, list) else None
    if items is None or not all(isinstance(item, int) for item in items):
        return None
    return cast(list[int], items)


def _history_payload(state: RestorationTrajectoryState) -> list[dict[str, Any]]:
    """Build a compact, serializable action and feedback history."""

    history: list[dict[str, Any]] = []
    for step in state.steps:
        history.append(
            {
                "step_index": step.step_index,
                "action": step.expert_decision.action,
                "success": step.success,
                "aggregate_score": step.evaluation.aggregate_score if step.evaluation is not None else None,
                "delta_from_previous": step.evaluation.delta_from_previous if step.evaluation is not None else None,
                "feedback": step.evaluation.feedback if step.evaluation is not None else None,
                "error": step.error,
            }
        )
    return history


def _history_messages(state: RestorationTrajectoryState) -> list[dict[str, Any]]:
    """Reconstruct prior assistant/tool messages without resending old images."""

    messages: list[dict[str, Any]] = []
    for step in state.steps:
        decision = step.expert_decision
        if decision.action is None or decision.parse_status != ExpertParseStatus.VALID:
            continue
        tool_call_id = decision.tool_call_id or f"history-call-{step.step_index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": RESTORE_FUNCTION_NAME,
                            "arguments": json.dumps({"action": decision.action}, separators=(",", ":")),
                        },
                    }
                ],
            }
        )
        tool_result = {
            "success": step.success,
            "aggregate_score": step.evaluation.aggregate_score if step.evaluation is not None else None,
            "delta_from_previous": step.evaluation.delta_from_previous if step.evaluation is not None else None,
            "delta_from_best": step.evaluation.delta_from_best if step.evaluation is not None else None,
            "feedback": step.evaluation.feedback if step.evaluation is not None else None,
            "error": step.error,
        }
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return messages


class VLMRestorationExpertAgent:
    """Call one served VLM once per restoration decision and retain the response."""

    def __init__(
        self,
        settings: ExpertVLMSettings,
        expert_name: ExpertName,
        tool_registry: ToolRegistry,
        *,
        max_steps: int,
        client: object | None = None,
    ) -> None:
        self.settings = settings
        self.expert_name = expert_name
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.client = cast(
            _OpenAICompatibleClient,
            client
            or OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                max_retries=0,
            ),
        )

    def decide(self, state: RestorationTrajectoryState) -> ExpertDecisionRecord:
        """Request exactly one restoration action for the current trajectory state."""

        step_index = len(state.steps)
        started_at = time.perf_counter()
        try:
            image_url = self._encode_image(state.current_image)
            state_prompt = build_expert_state_prompt(
                step_index=step_index,
                remaining_steps=max(self.max_steps - step_index, 0),
                current_score=state.current_evaluation.aggregate_score,
                best_score=state.best_evaluation.aggregate_score,
                original_score=state.original_evaluation.aggregate_score,
                consecutive_no_improvement=state.consecutive_no_improvement,
                history_json=json.dumps(_history_payload(state), ensure_ascii=False, separators=(",", ":")),
                latest_feedback=state.current_evaluation.feedback,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": build_expert_system_prompt(self.expert_name, self.tool_registry)},
                *_history_messages(state),
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": state_prompt},
                    ],
                },
            ]
            request: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "tools": [self.tool_registry.build_tool_schema()],
                "tool_choice": "auto",
                "max_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "seed": self.settings.seed,
            }
            if self.settings.return_token_ids:
                request["extra_body"] = {"return_token_ids": True}
            response: _DumpableResponse = self.client.chat.completions.create(**request)
        except APITimeoutError as error:
            return self._failed_decision(
                ExpertParseStatus.TIMEOUT,
                step_index,
                started_at,
                f"expert VLM request timed out: {error}",
            )
        except Exception as error:
            return self._failed_decision(
                ExpertParseStatus.REQUEST_FAILED,
                step_index,
                started_at,
                f"expert VLM request failed: {error}",
            )

        response_payload = response.model_dump(mode="json")
        choices_value: object = response_payload.get("choices")
        choices = cast(list[object], choices_value) if isinstance(choices_value, list) else []
        first_choice_value = choices[0] if choices else None
        first_choice = cast(dict[str, Any], first_choice_value) if isinstance(first_choice_value, dict) else {}
        message_value: object = first_choice.get("message")
        message = cast(dict[str, Any], message_value) if isinstance(message_value, dict) else {}
        raw_response = message.get("content") if isinstance(message.get("content"), str) else None
        reasoning_content = (
            message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None
        )
        parse_status, parsed_payload, action, tool_call_id, parse_error = parse_expert_response(
            raw_response,
            self.tool_registry,
            message.get("tool_calls"),
        )
        usage_value: object = response_payload.get("usage")
        usage_payload = cast(dict[object, object], usage_value) if isinstance(usage_value, dict) else {}
        usage = {key: value for key, value in usage_payload.items() if isinstance(key, str) and isinstance(value, int)}
        return ExpertDecisionRecord(
            expert_name=self.expert_name,
            step_index=step_index,
            action=action,
            decision_source=ExpertDecisionSource.VLM,
            parse_status=parse_status,
            api_succeeded=True,
            tool_call_id=tool_call_id,
            llm_response_id=(response_payload.get("id") if isinstance(response_payload.get("id"), str) else None),
            validation_status=self._validation_status(parse_status),
            raw_assistant_output=raw_response,
            reasoning_content=reasoning_content,
            parsed_payload=parsed_payload,
            model=response_payload.get("model") if isinstance(response_payload.get("model"), str) else None,
            finish_reason=(
                first_choice.get("finish_reason") if isinstance(first_choice.get("finish_reason"), str) else None
            ),
            latency_seconds=time.perf_counter() - started_at,
            prompt_token_ids=_int_list(response_payload.get("prompt_token_ids")),
            generated_token_ids=_int_list(first_choice.get("token_ids")),
            usage=usage,
            response_payload=response_payload,
            error=parse_error,
        )

    def _failed_decision(
        self,
        status: ExpertParseStatus,
        step_index: int,
        started_at: float,
        error: str,
    ) -> ExpertDecisionRecord:
        return ExpertDecisionRecord(
            expert_name=self.expert_name,
            step_index=step_index,
            action=None,
            decision_source=ExpertDecisionSource.VLM,
            parse_status=status,
            api_succeeded=False,
            validation_status=ValidationStatus.INVALID_TOOL_CALL,
            latency_seconds=time.perf_counter() - started_at,
            error=error,
        )

    @staticmethod
    def _validation_status(parse_status: ExpertParseStatus) -> ValidationStatus:
        if parse_status == ExpertParseStatus.VALID:
            return ValidationStatus.VALID
        if parse_status == ExpertParseStatus.UNKNOWN_ACTION:
            return ValidationStatus.UNKNOWN_ACTION
        return ValidationStatus.INVALID_TOOL_CALL

    @staticmethod
    def _encode_image(image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"expert image does not exist: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
