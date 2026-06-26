"""Configuration models and YAML loading for the restoration example."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator
from schemas import ExpertDecisionMode, ExpertName, RoutingMode, StrictModel


class WorkflowSettings(StrictModel):
    """Controller limits and reward coefficients."""

    max_steps: int = Field(default=6, ge=1)
    max_consecutive_failures: int = Field(default=2, ge=1)
    no_improvement_limit: int | None = Field(default=3, ge=1)
    improvement_epsilon: float = Field(default=1e-6, ge=0.0)
    tool_call_cost: float = Field(default=0.01, ge=0.0)
    tool_call_reward: float = Field(default=0.0, ge=0.0)
    invalid_action_penalty: float = Field(default=1.0, ge=0.0)
    failure_penalty: float = Field(default=0.25, ge=0.0)
    reward_mode: Literal["best_gain_v1", "step_iqa_sum_v1"] = "best_gain_v1"
    reward_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    reward_scale: float = Field(default=1.0, gt=0.0)
    step_reward_clip: float = Field(default=10.0, gt=0.0)
    repeated_action_penalty: float = Field(default=0.0, ge=0.0)
    stop_min_tool_calls: int = Field(default=1, ge=0)
    stop_min_best_gain: float = 0.0
    valid_stop_reward: float = 0.0
    premature_stop_penalty: float = Field(default=0.0, ge=0.0)
    forced_termination_penalty: float = Field(default=0.0, ge=0.0)


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


class SubprocessSettings(StrictModel):
    """Shared settings for a model process isolated in another conda environment."""

    environment_name: str = Field(default="verl", min_length=1)
    python_executable: str | None = None
    service_url: str | None = None
    entrypoint: str
    external_tools_root: str
    device: str = "cuda:0"
    timeout_seconds: float = Field(default=300.0, gt=0.0)


class IQAMetricConfig(StrictModel):
    """Normalization and aggregation configuration for one IQA metric."""

    name: str = Field(min_length=1)
    weight: float = Field(gt=0.0)
    minimum: float
    maximum: float
    higher_is_better: bool = True

    @model_validator(mode="after")
    def validate_range(self) -> IQAMetricConfig:
        if self.maximum <= self.minimum:
            raise ValueError("IQA metric maximum must be greater than minimum")
        return self


class EvaluatorSettings(SubprocessSettings):
    """Configuration for the isolated IQA evaluator."""

    iqa_repo: str
    metrics: list[IQAMetricConfig] = Field(default_factory=list)
    reward_calibration_path: str | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> EvaluatorSettings:
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("IQA metric names must be unique")
        if not names and self.reward_calibration_path is None:
            raise ValueError("metrics or reward_calibration_path must configure at least one IQA metric")
        return self


class RealRuntimeConfig(StrictModel):
    """Stage D restoration and IQA subprocess configuration."""

    restoration: SubprocessSettings
    evaluator: EvaluatorSettings


class RealExampleConfig(ExampleConfig):
    """Top-level configuration for the stage D real-model workflow."""

    runtime: RealRuntimeConfig


class VLMEndpointSettings(StrictModel):
    """Shared OpenAI-compatible VLM endpoint settings."""

    backend: Literal["vllm_openai"] = "vllm_openai"
    base_url: str = Field(min_length=1)
    api_key: str = Field(default="EMPTY", min_length=1)
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0
    return_token_ids: bool = True


class VLMSettings(VLMEndpointSettings):
    """OpenAI-compatible VLM endpoint and stage E routing settings."""

    routing_mode: RoutingMode = RoutingMode.ORACLE_OBSERVE
    diagnosis_failure_penalty: float = Field(default=1.0, ge=0.0)


class ExpertVLMSettings(VLMEndpointSettings):
    """Stage F expert VLM endpoint and default decision path."""

    decision_mode: ExpertDecisionMode = ExpertDecisionMode.REPLAY
    enable_thinking: bool = False


class ExpertResourceConfig(StrictModel):
    """Independent policy resource reserved for one restoration expert."""

    resource_name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    served_model_name: str = Field(min_length=1)
    policy_path: str | None = None


class StageEExampleConfig(RealExampleConfig):
    """Top-level configuration for real VLM diagnosis and stage D tools."""

    vlm: VLMSettings


class StageFExampleConfig(StageEExampleConfig):
    """Top-level configuration for stage F diagnosis and expert inference."""

    expert_vlm: ExpertVLMSettings


class StageGExampleConfig(StageFExampleConfig):
    """Top-level configuration for four parallel restoration experts."""

    expert_resources: dict[ExpertName, ExpertResourceConfig]

    @model_validator(mode="after")
    def validate_expert_resources(self) -> StageGExampleConfig:
        expected = set(ExpertName)
        if set(self.expert_resources) != expected:
            missing = sorted(item.value for item in expected - set(self.expert_resources))
            extra = sorted(str(item) for item in set(self.expert_resources) - expected)
            raise ValueError(
                f"expert_resources must define exactly four stable experts; missing={missing}, extra={extra}"
            )
        resource_names = [resource.resource_name for resource in self.expert_resources.values()]
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("each expert must use a unique resource_name")
        return self


def _resolve_runtime_path(config_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


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


def load_real_example_config(path: str | Path) -> RealExampleConfig:
    """Load stage D configuration and resolve all local runtime paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = RealExampleConfig.model_validate(payload)
    config.tools_config = _resolve_runtime_path(config_path, config.tools_config)
    config.runtime.restoration.entrypoint = _resolve_runtime_path(config_path, config.runtime.restoration.entrypoint)
    config.runtime.restoration.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.restoration.external_tools_root
    )
    config.runtime.evaluator.entrypoint = _resolve_runtime_path(config_path, config.runtime.evaluator.entrypoint)
    config.runtime.evaluator.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.evaluator.external_tools_root
    )
    config.runtime.evaluator.iqa_repo = _resolve_runtime_path(config_path, config.runtime.evaluator.iqa_repo)
    if config.runtime.evaluator.reward_calibration_path is not None:
        config.runtime.evaluator.reward_calibration_path = _resolve_runtime_path(
            config_path, config.runtime.evaluator.reward_calibration_path
        )
    return config


def load_stage_e_example_config(path: str | Path) -> StageEExampleConfig:
    """Load stage E configuration and resolve all local runtime paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = StageEExampleConfig.model_validate(payload)
    config.tools_config = _resolve_runtime_path(config_path, config.tools_config)
    config.runtime.restoration.entrypoint = _resolve_runtime_path(config_path, config.runtime.restoration.entrypoint)
    config.runtime.restoration.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.restoration.external_tools_root
    )
    config.runtime.evaluator.entrypoint = _resolve_runtime_path(config_path, config.runtime.evaluator.entrypoint)
    config.runtime.evaluator.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.evaluator.external_tools_root
    )
    config.runtime.evaluator.iqa_repo = _resolve_runtime_path(config_path, config.runtime.evaluator.iqa_repo)
    if config.runtime.evaluator.reward_calibration_path is not None:
        config.runtime.evaluator.reward_calibration_path = _resolve_runtime_path(
            config_path, config.runtime.evaluator.reward_calibration_path
        )
    return config


def load_stage_f_example_config(path: str | Path) -> StageFExampleConfig:
    """Load stage F configuration and resolve all local runtime paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = StageFExampleConfig.model_validate(payload)
    config.tools_config = _resolve_runtime_path(config_path, config.tools_config)
    config.runtime.restoration.entrypoint = _resolve_runtime_path(config_path, config.runtime.restoration.entrypoint)
    config.runtime.restoration.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.restoration.external_tools_root
    )
    config.runtime.evaluator.entrypoint = _resolve_runtime_path(config_path, config.runtime.evaluator.entrypoint)
    config.runtime.evaluator.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.evaluator.external_tools_root
    )
    config.runtime.evaluator.iqa_repo = _resolve_runtime_path(config_path, config.runtime.evaluator.iqa_repo)
    if config.runtime.evaluator.reward_calibration_path is not None:
        config.runtime.evaluator.reward_calibration_path = _resolve_runtime_path(
            config_path, config.runtime.evaluator.reward_calibration_path
        )
    return config


def load_stage_g_example_config(path: str | Path) -> StageGExampleConfig:
    """Load stage G four-expert configuration and resolve local paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = StageGExampleConfig.model_validate(payload)
    config.tools_config = _resolve_runtime_path(config_path, config.tools_config)
    config.runtime.restoration.entrypoint = _resolve_runtime_path(config_path, config.runtime.restoration.entrypoint)
    config.runtime.restoration.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.restoration.external_tools_root
    )
    config.runtime.evaluator.entrypoint = _resolve_runtime_path(config_path, config.runtime.evaluator.entrypoint)
    config.runtime.evaluator.external_tools_root = _resolve_runtime_path(
        config_path, config.runtime.evaluator.external_tools_root
    )
    config.runtime.evaluator.iqa_repo = _resolve_runtime_path(config_path, config.runtime.evaluator.iqa_repo)
    if config.runtime.evaluator.reward_calibration_path is not None:
        config.runtime.evaluator.reward_calibration_path = _resolve_runtime_path(
            config_path, config.runtime.evaluator.reward_calibration_path
        )
    for resource in config.expert_resources.values():
        if resource.policy_path is not None:
            resource.policy_path = _resolve_runtime_path(config_path, resource.policy_path)
    return config
