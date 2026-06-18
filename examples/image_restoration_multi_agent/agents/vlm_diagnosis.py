"""Single-call VLM diagnosis through a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Protocol, cast

from config import VLMSettings
from openai import APITimeoutError, OpenAI
from schemas import (
    DEGRADATION_TO_EXPERT,
    DegradationType,
    DiagnosisParseStatus,
    DiagnosisResult,
    VLMDiagnosisAttempt,
)

from .prompts import DIAGNOSIS_TOOL_NAME, DIAGNOSIS_TOOL_SCHEMA, build_diagnosis_system_prompt


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
) -> tuple[DiagnosisParseStatus | None, dict[str, Any] | None, str | None]:
    """Extract exactly one raw Hermes tool call without accepting bare JSON."""

    if raw_response is None or not raw_response.strip():
        return DiagnosisParseStatus.EMPTY_RESPONSE, None, "VLM returned no tool call or assistant content"
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_response, flags=re.DOTALL)
    if not matches:
        return DiagnosisParseStatus.INVALID_TOOL_CALL, None, "no Hermes <tool_call> block found"
    if len(matches) != 1:
        return DiagnosisParseStatus.MULTIPLE_TOOL_CALLS, None, "diagnosis must contain exactly one tool call"
    try:
        payload: object = json.loads(matches[0])
    except json.JSONDecodeError as error:
        return DiagnosisParseStatus.INVALID_JSON, None, f"invalid Hermes tool-call JSON: {error}"
    if not isinstance(payload, dict):
        return DiagnosisParseStatus.INVALID_TOOL_CALL, None, "Hermes tool call must contain a JSON object"
    return None, cast(dict[str, Any], payload), None


def _tool_call_from_openai(
    tool_calls_value: object | None,
) -> tuple[DiagnosisParseStatus | None, dict[str, Any] | None, str | None]:
    """Normalize vLLM's parsed OpenAI tool-call representation."""

    if not isinstance(tool_calls_value, list) or not tool_calls_value:
        return None, None, None
    tool_calls = cast(list[object], tool_calls_value)
    if len(tool_calls) != 1:
        return DiagnosisParseStatus.MULTIPLE_TOOL_CALLS, None, "diagnosis must contain exactly one tool call"
    tool_call_value = tool_calls[0]
    if not isinstance(tool_call_value, dict):
        return DiagnosisParseStatus.INVALID_TOOL_CALL, None, "parsed tool call must be an object"
    tool_call = cast(dict[str, Any], tool_call_value)
    function_value: object = tool_call.get("function")
    if not isinstance(function_value, dict):
        return DiagnosisParseStatus.INVALID_TOOL_CALL, None, "parsed tool call is missing function"
    function = cast(dict[str, Any], function_value)
    arguments_value: object = function.get("arguments")
    if isinstance(arguments_value, str):
        try:
            arguments_value = json.loads(arguments_value)
        except json.JSONDecodeError as error:
            return DiagnosisParseStatus.INVALID_JSON, None, f"invalid function arguments JSON: {error}"
    return None, {"name": function.get("name"), "arguments": arguments_value}, None


def parse_diagnosis_response(
    raw_response: str | None,
    tool_calls: object | None = None,
) -> tuple[DiagnosisParseStatus, dict[str, Any] | None, DiagnosisResult | None, str | None]:
    """Validate one parsed or raw Hermes diagnosis tool call."""

    status, payload, error = _tool_call_from_openai(tool_calls)
    if status is not None:
        return status, payload, None, error
    if payload is None:
        status, payload, error = _tool_call_from_raw_hermes(raw_response)
        if status is not None:
            return status, payload, None, error
    if payload is None:
        return DiagnosisParseStatus.EMPTY_RESPONSE, None, None, "VLM returned no diagnosis tool call"

    if payload.get("name") != DIAGNOSIS_TOOL_NAME:
        return DiagnosisParseStatus.UNKNOWN_FUNCTION, payload, None, "unexpected diagnosis function name"
    arguments_value: object = payload.get("arguments")
    if not isinstance(arguments_value, dict):
        return DiagnosisParseStatus.INVALID_TOOL_CALL, payload, None, "tool arguments must be a JSON object"
    arguments = cast(dict[str, Any], arguments_value)
    required_fields = {"primary_type", "visual_evidence"}
    missing_fields = sorted(required_fields - set(arguments))
    if missing_fields:
        return (
            DiagnosisParseStatus.MISSING_FIELD,
            payload,
            None,
            f"missing required fields: {', '.join(missing_fields)}",
        )
    extra_fields = sorted(set(arguments) - required_fields)
    if extra_fields:
        return (
            DiagnosisParseStatus.INVALID_TOOL_CALL,
            payload,
            None,
            f"unexpected argument fields: {', '.join(extra_fields)}",
        )
    evidence_value: object = arguments.get("visual_evidence")
    evidence_items = cast(list[object], evidence_value) if isinstance(evidence_value, list) else None
    if evidence_items is None or not all(isinstance(item, str) for item in evidence_items):
        return (
            DiagnosisParseStatus.MISSING_FIELD,
            payload,
            None,
            "visual_evidence must be a JSON array of strings",
        )

    try:
        primary_type = DegradationType(arguments["primary_type"])
    except (KeyError, TypeError, ValueError):
        return (
            DiagnosisParseStatus.INVALID_CATEGORY,
            payload,
            None,
            "primary_type is outside the supported category set",
        )

    diagnosis = DiagnosisResult(
        primary_type=primary_type,
        visual_evidence=cast(list[str], evidence_items),
        route_to=DEGRADATION_TO_EXPERT[primary_type],
    )
    return DiagnosisParseStatus.VALID, payload, diagnosis, None


def _int_list(value: Any) -> list[int] | None:
    value_object: object = value
    items = cast(list[object], value_object) if isinstance(value_object, list) else None
    if items is None or not all(isinstance(item, int) for item in items):
        return None
    return cast(list[int], items)


class VLMDegradationDiagnosisAgent:
    """Call a served VLM exactly once and retain the complete response."""

    def __init__(self, settings: VLMSettings, *, client: object | None = None) -> None:
        self.settings = settings
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

    def diagnose(self, image_path: str) -> VLMDiagnosisAttempt:
        """Send one image-only diagnosis request without exposing its path or label."""

        started_at = time.perf_counter()
        try:
            image_url = self._encode_image(image_path)
            request: dict[str, Any] = {
                "model": self.settings.model,
                "messages": [
                    {"role": "system", "content": build_diagnosis_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {
                                "type": "text",
                                "text": "Diagnose this image using exactly one diagnose_degradation Hermes tool call.",
                            },
                        ],
                    },
                ],
                "tools": [DIAGNOSIS_TOOL_SCHEMA],
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
            return self._failed_attempt(
                DiagnosisParseStatus.TIMEOUT,
                started_at,
                f"VLM request timed out: {error}",
            )
        except Exception as error:
            return self._failed_attempt(
                DiagnosisParseStatus.REQUEST_FAILED,
                started_at,
                f"VLM request failed: {error}",
            )

        response_payload = response.model_dump(mode="json")
        choices_value: object = response_payload.get("choices")
        choices = cast(list[object], choices_value) if isinstance(choices_value, list) else []
        first_choice_value = choices[0] if choices else None
        first_choice = cast(dict[str, Any], first_choice_value) if isinstance(first_choice_value, dict) else {}
        message_value: object = first_choice.get("message")
        message = cast(dict[str, Any], message_value) if isinstance(message_value, dict) else {}
        raw_response = message.get("content") if isinstance(message.get("content"), str) else None
        tool_calls_value: object = message.get("tool_calls")
        reasoning_content = (
            message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None
        )
        parse_status, parsed_payload, diagnosis, parse_error = parse_diagnosis_response(
            raw_response,
            tool_calls_value,
        )
        usage_value: object = response_payload.get("usage")
        usage_payload = cast(dict[object, object], usage_value) if isinstance(usage_value, dict) else {}
        usage = {key: value for key, value in usage_payload.items() if isinstance(key, str) and isinstance(value, int)}
        return VLMDiagnosisAttempt(
            backend=self.settings.backend,
            parse_status=parse_status,
            api_succeeded=True,
            raw_response=raw_response,
            reasoning_content=reasoning_content,
            parsed_payload=parsed_payload,
            diagnosis=diagnosis,
            response_id=response_payload.get("id") if isinstance(response_payload.get("id"), str) else None,
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

    def _failed_attempt(
        self,
        status: DiagnosisParseStatus,
        started_at: float,
        error: str,
    ) -> VLMDiagnosisAttempt:
        return VLMDiagnosisAttempt(
            backend=self.settings.backend,
            parse_status=status,
            api_succeeded=False,
            latency_seconds=time.perf_counter() - started_at,
            error=error,
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"diagnosis image does not exist: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
