"""ALFWorld adapter with a strict parser/validator boundary."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .parser import ParseResult, parse_tool_call
from .tool_registry import ALFWorldToolRegistry
from .validator import ValidationResult, validate_tool_call

@dataclass
class StepResult:
    observation: str
    executed_action: str | None
    is_valid: bool
    step_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    available_actions_hint: str = ""
    parse: ParseResult | None = None
    validation: ValidationResult | None = None

class ALFWorldAdapter:
    def __init__(self, manager: Any):
        self.manager = manager
        self.observation = ""
        self.info: dict[str, Any] = {}
        self.available_actions: tuple[str, ...] = ()
        self.mission = ""

    def reset(self) -> tuple[str, dict[str, Any], ALFWorldToolRegistry]:
        observation, info, _ = self.manager.reset()
        self.observation, self.info = str(observation), dict(info)
        self.available_actions = tuple(self.manager.env.available_actions)
        self.mission = self.manager.get_mission(self.observation, self.info)
        return self.observation, self.info, ALFWorldToolRegistry(self.available_actions)

    def step_tool_call(self, raw_text: str | None = None, tool_calls: object | None = None) -> StepResult:
        registry = ALFWorldToolRegistry(self.available_actions)
        parsed = parse_tool_call(raw_text, tool_calls)
        validation = validate_tool_call(parsed, registry)
        if not validation.is_valid:
            return StepResult(self.observation, None, False, -1.0, True, False, {"validation_status": validation.status.value, "error": validation.error}, "\n".join(self.available_actions), parsed, validation)
        env_obs, action, is_valid, reward, terminated, truncated, info, hint = self.manager.step(validation.action, use_reasoning=False)
        self.observation, self.info = str(env_obs), dict(info)
        self.available_actions = tuple(self.manager.env.available_actions)
        return StepResult(self.observation, action, bool(is_valid), float(reward), bool(terminated), bool(truncated), self.info, hint, parsed, validation)

    def close(self) -> None:
        self.manager.close()
