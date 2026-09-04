"""Runtime validation before an action is sent to ALFWorld."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .parser import ParseResult, ParseStatus
from .tool_registry import ACTION_FIELD, ALFWorldToolRegistry, TOOL_NAME, UnknownActionError

class ValidationStatus(str, Enum):
    VALID = "VALID"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    INVALID_JSON = "INVALID_JSON"
    MULTIPLE_TOOL_CALLS = "MULTIPLE_TOOL_CALLS"
    UNKNOWN_FUNCTION = "UNKNOWN_FUNCTION"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"

@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    action: str | None
    error: str | None = None
    @property
    def is_valid(self) -> bool:
        return self.status is ValidationStatus.VALID

def validate_tool_call(result: ParseResult, registry: ALFWorldToolRegistry) -> ValidationResult:
    mapped = {ParseStatus.EMPTY_RESPONSE: ValidationStatus.EMPTY_RESPONSE, ParseStatus.INVALID_JSON: ValidationStatus.INVALID_JSON, ParseStatus.MULTIPLE_TOOL_CALLS: ValidationStatus.MULTIPLE_TOOL_CALLS}
    if result.status in mapped:
        return ValidationResult(mapped[result.status], None, result.error)
    if result.call is None or result.status is not ParseStatus.VALID:
        return ValidationResult(ValidationStatus.INVALID_ARGUMENTS, None, result.error or "invalid parsed call")
    if result.call.name != TOOL_NAME:
        return ValidationResult(ValidationStatus.UNKNOWN_FUNCTION, None, f"unexpected function: {result.call.name!r}")
    args = result.call.arguments
    if ACTION_FIELD not in args:
        return ValidationResult(ValidationStatus.MISSING_FIELD, None, "missing required field: action")
    extra = sorted(set(args) - {ACTION_FIELD})
    if extra:
        return ValidationResult(ValidationStatus.INVALID_ARGUMENTS, None, f"unexpected fields: {extra}")
    try:
        action = registry.validate_action(args[ACTION_FIELD])
    except UnknownActionError as exc:
        return ValidationResult(ValidationStatus.UNKNOWN_ACTION, None, str(exc))
    return ValidationResult(ValidationStatus.VALID, action)
