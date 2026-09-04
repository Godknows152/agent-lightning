"""Qwen2.5/Hermes and Qwen3 native tool-call parser for the isolated baseline."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import Any
from .tool_registry import TOOL_NAME

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

class ParseStatus(str, Enum):
    VALID = "VALID"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    INVALID_JSON = "INVALID_JSON"
    MULTIPLE_TOOL_CALLS = "MULTIPLE_TOOL_CALLS"
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"

@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: Mapping[str, Any]
    raw_text: str
    tool_call_id: str | None = None

@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    call: ParsedToolCall | None
    error: str | None = None

def _from_payload(payload: object, raw_text: str, tool_call_id: str | None = None) -> ParseResult:
    if not isinstance(payload, Mapping):
        return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "tool call must be an object")
    name, arguments = payload.get("name"), payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "tool call requires name and object arguments")
    return ParseResult(ParseStatus.VALID, ParsedToolCall(name, dict(arguments), raw_text, tool_call_id))

def parse_tool_call(raw_text: str | None = None, tool_calls: object | None = None) -> ParseResult:
    """Parse one OpenAI call or one Qwen tool-call block."""
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
        if len(tool_calls) > 1:
            return ParseResult(ParseStatus.MULTIPLE_TOOL_CALLS, None, "exactly one tool call is required")
        if len(tool_calls) == 1:
            item = tool_calls[0]
            if not isinstance(item, Mapping):
                return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "tool call must be an object")
            function = item.get("function", item)
            if not isinstance(function, Mapping):
                return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "missing function object")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    return ParseResult(ParseStatus.INVALID_JSON, None, str(exc))
            return _from_payload({"name": function.get("name"), "arguments": arguments}, raw_text or "", item.get("id") if isinstance(item.get("id"), str) else None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ParseResult(ParseStatus.EMPTY_RESPONSE, None, "model returned no tool call")
    matches = _TOOL_CALL_RE.findall(raw_text)
    if len(matches) != 1:
        status = ParseStatus.MULTIPLE_TOOL_CALLS if len(matches) > 1 else ParseStatus.EMPTY_RESPONSE
        return ParseResult(status, None, "expected exactly one <tool_call> JSON block")
    body = matches[0].strip()
    if body.startswith("{"):
        try:
            return _from_payload(json.loads(body), raw_text)
        except json.JSONDecodeError as exc:
            return ParseResult(ParseStatus.INVALID_JSON, None, str(exc))

    function_matches = _FUNCTION_RE.findall(body)
    if len(function_matches) != 1:
        return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "expected exactly one <function=name> block")
    name, function_body = function_matches[0]
    arguments: dict[str, Any] = {}
    for parameter_name, value in _PARAMETER_RE.findall(function_body):
        arguments[parameter_name.strip()] = value.strip()
    if not arguments:
        return ParseResult(ParseStatus.INVALID_TOOL_CALL, None, "function call has no parameters")
    return _from_payload({"name": name.strip(), "arguments": arguments}, raw_text)
