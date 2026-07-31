"""Protocol and shared-tool-registry tests for stage A."""

from __future__ import annotations

from pathlib import Path

import pytest
from config import load_example_config
from exceptions import UnknownActionError
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
    assert action_schema["enum"] == list(registry.model_actions)
    assert action_schema["enum"][-1] == STOP_ACTION
    assert "- N_focalnet_dehaze: FocalNet image dehazing model with the ITS checkpoint." in action_schema["description"]
    assert "- stop: Stop the trajectory and keep the historical best restored image." in action_schema["description"]
    assert schema["function"]["parameters"]["additionalProperties"] is False
    assert set(registry.model_actions[:-1]) == {
        "A_real_esrgan",
        "B_scunet",
        "C_retinexformer_fivek",
        "D_hvicidnet",
        "E_lightdiff",
        "F_turbo_rain",
        "G_s2former",
        "H_idt",
        "I_ridcp",
        "J_kanet",
        "K_turbo_snow",
        "L_snowmaster",
        "M_nafnet_denoise",
        "N_focalnet_dehaze",
        "O_focalnet_desnow",
        "P_mb_taylorformer_dehaze",
    }
    assert all(registry.get_tool(action).runtime is not None for action in registry.actions[:-1])


def test_tool_schema_can_hide_stop_for_early_training_turns() -> None:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)
    schema = registry.build_tool_schema(include_stop=False)

    action_enum = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert STOP_ACTION not in action_enum
    assert action_enum == list(registry.model_actions[:-1])


def test_tool_descriptions_follow_stop_visibility() -> None:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)

    full_descriptions = registry.build_tool_descriptions()
    early_descriptions = registry.build_tool_descriptions(include_stop=False)

    assert "- N_focalnet_dehaze: FocalNet image dehazing model with the ITS checkpoint." in full_descriptions
    assert "- stop: Stop the trajectory and keep the historical best restored image." in full_descriptions
    assert "- stop:" not in early_descriptions


def test_model_action_mapping_is_reversible_and_rejects_canonical_model_output() -> None:
    config = load_example_config(EXAMPLE_DIR / "config" / "default.yaml")
    registry = ToolRegistry.from_yaml(config.tools_config)

    assert registry.to_model_action("scunet") == "B_scunet"
    assert registry.to_runtime_action("B_scunet") == "scunet"
    with pytest.raises(UnknownActionError):
        registry.to_runtime_action("scunet")


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
