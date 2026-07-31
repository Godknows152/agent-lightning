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
    model_name: str | None = Field(default=None, min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
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
        self._runtime_to_model = {
            tool.name: tool.model_name or tool.name for tool in self._tools.values()
        }
        self._runtime_to_model[STOP_ACTION] = STOP_ACTION
        self._model_to_runtime = {
            model_action: runtime_action
            for runtime_action, model_action in self._runtime_to_model.items()
        }
        if len(self._model_to_runtime) != len(self._runtime_to_model):
            raise ValueError("model-facing restoration action names must be unique")

        explicit_model_names = [tool.model_name for tool in self._tools.values()]
        if any(explicit_model_names):
            if not all(explicit_model_names):
                raise ValueError("either every enabled restoration tool must define model_name or none may define it")
            initials = [action[0] for action in self._model_to_runtime]
            if len(initials) != len(set(initials)):
                raise ValueError("model-facing restoration actions must have unique first characters")

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

    @property
    def model_actions(self) -> tuple[str, ...]:
        """Return the action values exposed to the model in tool schemas."""

        return tuple(self._runtime_to_model[action] for action in self.actions)

    def validate_action(self, action: str) -> None:
        """Reject actions that are not available to every expert."""

        if action not in self.actions:
            raise UnknownActionError(f"unknown restoration action: {action}")

    def to_model_action(self, action: str) -> str:
        """Translate one canonical runtime action to its model-facing name."""

        self.validate_action(action)
        return self._runtime_to_model[action]

    def to_runtime_action(self, model_action: str) -> str:
        """Translate a model-facing schema value to its canonical runtime action."""

        try:
            return self._model_to_runtime[model_action]
        except KeyError as error:
            raise UnknownActionError(f"unknown restoration action: {model_action}") from error

    def validate_model_action(self, model_action: str) -> None:
        """Reject values that are not in the model-facing action vocabulary."""

        self.to_runtime_action(model_action)

    def get_tool(self, action: str) -> ToolDefinition:
        """Return one enabled tool definition after validating its action."""

        self.validate_action(action)
        if action == STOP_ACTION:
            raise UnknownActionError("stop does not have a restoration runtime")
        return self._tools[action]

    def build_tool_schema(self, *, include_stop: bool = True) -> dict[str, Any]:
        """Build the model-facing OpenAI-compatible restore_image definition."""

        runtime_actions = self.actions if include_stop else tuple(self._tools)
        actions = [self.to_model_action(action) for action in runtime_actions]
        description = (
            "Apply exactly one registered restoration action or stop the trajectory."
            if include_stop
            else "Apply exactly one registered restoration action."
        )
        action_description = (
            "Select exactly one action enum value. Action meanings:\n"
            f"{self.build_tool_descriptions(include_stop=include_stop)}"
        )
        return {
            "type": "function",
            "function": {
                "name": RESTORE_FUNCTION_NAME,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": action_description,
                            "enum": actions,
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def build_tool_descriptions(self, *, include_stop: bool = True) -> str:
        """Build a compact action-to-purpose list for model-facing prompts."""

        lines = [
            f"- {self.to_model_action(tool.name)}: {tool.description}"
            for tool in self._tools.values()
        ]
        if include_stop:
            lines.append("- stop: Stop the trajectory and keep the historical best restored image.")
        return "\n".join(lines)
