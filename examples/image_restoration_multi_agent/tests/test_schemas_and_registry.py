"""Protocol and shared-tool-registry tests for stage A."""

from __future__ import annotations

from pathlib import Path

import pytest
from config import load_example_config
from pydantic import ValidationError
from schemas import DegradationType, DiagnosisResult, ExpertName
from tool_registry import RESTORE_FUNCTION_NAME, STOP_ACTION, ToolRegistry

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def test_config_uses_one_registry_for_all_experts() -> None:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registries = {expert.tool_registry for expert in config.experts.values()}

    assert set(config.experts) == set(ExpertName)
    assert registries == {"all_restoration_tools"}


def test_tool_schema_contains_all_actions_and_stop() -> None:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    schema = registry.build_tool_schema()

    assert schema["function"]["name"] == RESTORE_FUNCTION_NAME
    action_schema = schema["function"]["parameters"]["properties"]["action"]
    assert action_schema["enum"] == list(registry.actions)
    assert action_schema["enum"][-1] == STOP_ACTION
    assert schema["function"]["parameters"]["additionalProperties"] is False
    assert set(registry.actions[:-1]) == {
        "real_esrgan",
        "scunet",
        "retinexformer_fivek",
        "hvicidnet",
        "lightdiff",
        "turbo_rain",
        "s2former",
        "idt",
        "ridcp",
        "kanet",
        "turbo_snow",
        "snowmaster",
        "nafnet_denoise",
        "focalnet_dehaze",
        "focalnet_desnow",
        "mb_taylorformer_dehaze",
    }
    assert all(registry.get_tool(action).runtime is not None for action in registry.actions[:-1])


def test_diagnosis_rejects_mismatched_route() -> None:
    with pytest.raises(ValidationError, match="must route"):
        DiagnosisResult(primary_type=DegradationType.FOG, route_to=ExpertName.RAIN)


def test_schema_round_trip() -> None:
    diagnosis = DiagnosisResult(
        primary_type=DegradationType.LOW_LIGHT,
        route_to=ExpertName.LOW_LIGHT,
        visual_evidence=["dark regions"],
    )

    restored = DiagnosisResult.model_validate_json(diagnosis.model_dump_json())

    assert restored == diagnosis
