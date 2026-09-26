"""Merge an ALFWorld SFT adapter into a standalone full model on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=("qwen35_2b", "qwen35_9b"), default="qwen35_2b")
    args = parser.parse_args()
    profile = OmegaConf.load(ROOT / "config/model_profiles" / f"{args.model_profile}.yaml").variables
    base = Path(profile.BASE_MODEL).resolve()
    adapter = Path(profile.SFT_ADAPTER).resolve()
    output = Path(profile.SFT_MODEL).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing export: {output}")
    for source in (base / "config.json", adapter / "adapter_config.json", adapter / "adapter_model.safetensors"):
        if not source.is_file():
            raise FileNotFoundError(f"Missing SFT export input: {source}")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if Path(adapter_config["base_model_name_or_path"]).resolve() != base:
        raise ValueError("SFT adapter and profile must use the same base model")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    torch.set_num_threads(8)
    print(f"Merging base={base} adapter={adapter} output={output}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("The exported model still contains adapter parameters")
    model.requires_grad_(True)
    count = sum(parameter.numel() for parameter in model.parameters())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        model.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
        # Preserve the base tokenizer/processor used by the existing RL rollout.
        AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(staging)
        AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(staging)
        with (adapter / "adapter_model.safetensors").open("rb") as adapter_file:
            adapter_sha256 = hashlib.file_digest(adapter_file, "sha256").hexdigest()
        manifest = {
            "base_model": str(base), "sft_adapter": str(adapter), "dtype": "bfloat16",
            "parameter_count": count, "merge_method": "peft.merge_and_unload(safe_merge=True)",
            "adapter_sha256": adapter_sha256,
        }
        (staging / "sft_merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    print(f"Exported full SFT model: {output} parameters={count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
