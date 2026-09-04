"""Dynamic model-facing schema for ALFWorld's text action space."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

TOOL_NAME = "alfworld_action"
ACTION_FIELD = "action"

class UnknownActionError(ValueError):
    pass

@dataclass(frozen=True)
class ALFWorldToolRegistry:
    admissible_actions: tuple[str, ...]

    def __init__(self, admissible_actions: Iterable[str]):
        actions = tuple(str(action) for action in admissible_actions)
        if not actions or any(not action.strip() for action in actions):
            raise ValueError("ALFWorld returned no/empty admissible action")
        if len(set(actions)) != len(actions):
            raise ValueError("ALFWorld returned duplicate admissible actions")
        object.__setattr__(self, "admissible_actions", actions)

    def available_actions(self) -> tuple[str, ...]:
        return self.admissible_actions

    def validate_action(self, action: object) -> str:
        if not isinstance(action, str) or not action:
            raise UnknownActionError("action must be a non-empty string")
        if action not in self.admissible_actions:
            raise UnknownActionError(f"action is not admissible in the current state: {action!r}")
        return action

    def to_runtime_action(self, action: object) -> str:
        return self.validate_action(action)

    def build_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Execute exactly one currently admissible ALFWorld text action.",
                "parameters": {
                    "type": "object",
                    "properties": {ACTION_FIELD: {"type": "string", "description": "Copy one action exactly from the current admissible action list.", "enum": list(self.admissible_actions)}},
                    "required": [ACTION_FIELD],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def static_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Execute exactly one ALFWorld text action.",
                "parameters": {"type": "object", "properties": {ACTION_FIELD: {"type": "string"}}, "required": [ACTION_FIELD], "additionalProperties": False},
            },
        }
