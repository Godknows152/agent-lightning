"""Shared restoration tool registry and OpenAI-compatible schema generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from exceptions import UnknownActionError
from pydantic import Field, model_validator
from schemas import StrictModel

STOP_ACTION = "stop"
RESTORE_FUNCTION_NAME = "restore_image"


class ToolRuntime(StrictModel):
    """Runtime information required by the isolated restoration process."""

    adapter: Literal["verl_toolkit", "candidate"]
    model: str = Field(min_length=1)
    repo: str | None = None
    checkpoint: str | None = None

    @model_validator(mode="after")
    def validate_candidate_paths(self) -> ToolRuntime:
        if self.adapter == "candidate" and (not self.repo or not self.checkpoint):
            raise ValueError("candidate tools require repo and checkpoint paths")
        return self


class ToolDefinition(StrictModel):
    """One registered restoration action."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    enabled: bool = True
    runtime: ToolRuntime | None = None

    @model_validator(mode="after")
    def reject_reserved_stop(self) -> ToolDefinition:
        if self.name == STOP_ACTION:
            raise ValueError("stop is reserved and added by the registry")
        return self


class ToolRegistryConfig(StrictModel):
    """YAML representation of one shared registry."""

    registry_name: str = Field(min_length=1)
    tools: list[ToolDefinition]

    @model_validator(mode="after")
    def validate_unique_tools(self) -> ToolRegistryConfig:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if not any(tool.enabled for tool in self.tools):
            raise ValueError("at least one restoration tool must be enabled")
        return self


class ToolRegistry:
    """Runtime registry shared by every restoration expert."""

    def __init__(self, config: ToolRegistryConfig) -> None:
        self.config = config
        self._tools = {tool.name: tool for tool in config.tools if tool.enabled}

    @classmethod
    def from_yaml(cls, path: str | Path) -> ToolRegistry:
        """Load one registry from YAML."""

        registry_path = Path(path).expanduser().resolve()
        with registry_path.open("r", encoding="utf-8") as registry_file:
            payload = yaml.safe_load(registry_file)
        if not isinstance(payload, dict):
            raise ValueError(f"tool registry must be a mapping: {registry_path}")
        return cls(ToolRegistryConfig.model_validate(payload))

    @classmethod
    def from_actions(cls, actions: list[str]) -> ToolRegistry:
        """Construct a small registry for tests and local deterministic runs."""

        config = ToolRegistryConfig(
            registry_name="test_restoration_tools",
            tools=[ToolDefinition(name=action, description=f"Deterministic {action} action") for action in actions],
        )
        return cls(config)

    @property
    def actions(self) -> tuple[str, ...]:
        """Return all enabled restoration actions plus the reserved stop action."""

        return (*self._tools.keys(), STOP_ACTION)

    def validate_action(self, action: str) -> None:
        """Reject actions that are not available to every expert."""

        if action not in self.actions:
            raise UnknownActionError(f"unknown restoration action: {action}")

    def get_tool(self, action: str) -> ToolDefinition:
        """Return one enabled tool definition after validating its action."""

        self.validate_action(action)
        if action == STOP_ACTION:
            raise UnknownActionError("stop does not have a restoration runtime")
        return self._tools[action]

    def build_tool_schema(self) -> dict[str, Any]:
        """Build the canonical OpenAI-compatible restore_image tool definition."""

        return {
            "type": "function",
            "function": {
                "name": RESTORE_FUNCTION_NAME,
                "description": "Apply one registered restoration action or stop the trajectory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(self.actions),
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
