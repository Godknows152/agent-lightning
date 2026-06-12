"""Configuration models and YAML loading for the restoration example."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator
from schemas import ExpertName, StrictModel


class WorkflowSettings(StrictModel):
    """Controller limits and reward coefficients."""

    max_steps: int = Field(default=6, ge=1)
    max_consecutive_failures: int = Field(default=2, ge=1)
    no_improvement_limit: int = Field(default=3, ge=1)
    improvement_epsilon: float = Field(default=1e-6, ge=0.0)
    tool_call_cost: float = Field(default=0.01, ge=0.0)
    invalid_action_penalty: float = Field(default=1.0, ge=0.0)
    failure_penalty: float = Field(default=0.25, ge=0.0)


class ExpertConfig(StrictModel):
    """Reference from one expert to the shared tool registry."""

    tool_registry: str


class ExampleConfig(StrictModel):
    """Top-level example configuration."""

    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    tools_config: str
    experts: dict[ExpertName, ExpertConfig]

    @model_validator(mode="after")
    def validate_experts(self) -> ExampleConfig:
        expected = set(ExpertName)
        if set(self.experts) != expected:
            missing = sorted(item.value for item in expected - set(self.experts))
            extra = sorted(str(item) for item in set(self.experts) - expected)
            raise ValueError(f"experts must define exactly four stable experts; missing={missing}, extra={extra}")
        registries = {expert.tool_registry for expert in self.experts.values()}
        if len(registries) != 1:
            raise ValueError("all experts must reference the same tool registry")
        return self


def load_example_config(path: str | Path) -> ExampleConfig:
    """Load and validate the example YAML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = ExampleConfig.model_validate(payload)
    tools_path = Path(config.tools_config)
    if not tools_path.is_absolute():
        config.tools_config = str((config_path.parent / tools_path).resolve())
    return config
