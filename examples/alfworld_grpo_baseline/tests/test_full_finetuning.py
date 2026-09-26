"""Prevent accidental adapter training after full-SFT migration."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from alfworld_baseline.full_finetuning import validate_full_finetuning

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["qwen35_2b", "qwen35_9b", "qwen25_1_5b"])
@pytest.mark.parametrize("backend", ["trajectory", "gigpo_grpo"])
def test_every_backend_uses_full_sft_model(profile, backend, tmp_path, monkeypatch):
    monkeypatch.setenv("ALFWORLD_SFT_MODEL", str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    with initialize_config_dir(config_dir=str(ROOT / "config/alfworld" / profile / "v1"), version_base=None):
        overrides = ["+training_backend=gigpo_grpo"] if backend == "gigpo_grpo" else []
        config = compose(config_name="alfworld_config_2gpu", overrides=overrides)
    assert config.actor_rollout_ref.model.path == str(tmp_path)
    validate_full_finetuning(config)


@pytest.mark.parametrize("override", [
    "lora_rank=16", "lora_adapter_path=/tmp/adapter", "lora.rank=16", "lora.adapter_path=/tmp/adapter",
])
def test_adapter_overrides_are_rejected(override):
    config = OmegaConf.create({"actor_rollout_ref": {"model": {"lora_rank": 0, "lora_adapter_path": None}}})
    config = OmegaConf.merge(config, OmegaConf.from_dotlist([f"actor_rollout_ref.model.{override}"]))
    with pytest.raises(ValueError, match="full-parameter"):
        validate_full_finetuning(config)


def test_adapter_directory_is_not_a_full_model(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    config = OmegaConf.create({"actor_rollout_ref": {"model": {
        "lora_rank": 0, "lora_adapter_path": None, "path": str(tmp_path),
    }}})
    with pytest.raises(ValueError, match="adapter directory"):
        validate_full_finetuning(config)
