"""Validate standalone full-model initialization before allocating training GPUs."""
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_full_finetuning(config: DictConfig) -> None:
    """Reject adapter training and missing or adapter-only SFT checkpoints."""
    model = config.actor_rollout_ref.model
    if (
        int(model.lora_rank) != 0
        or model.lora_adapter_path is not None
        or int(OmegaConf.select(model, "lora.rank", default=0)) != 0
        or OmegaConf.select(model, "lora.adapter_path") is not None
    ):
        raise ValueError("ALFWorld requires full-parameter training: merge the SFT adapter before launching")
    path = Path(model.path)
    if (path / "adapter_config.json").exists():
        raise ValueError(f"Expected a standalone full SFT model, got an adapter directory: {path}")
    if not (path / "config.json").is_file() or not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(
            f"Missing full SFT model: {path}. Run scripts/export_sft_model.py --model-profile "
            f"{config.variables.MODEL_PROFILE}, or set ALFWORLD_SFT_MODEL to an existing full SFT model."
        )
    print(f"full_finetuning model={path} lora_rank=0 lora_adapter_path=None", flush=True)
