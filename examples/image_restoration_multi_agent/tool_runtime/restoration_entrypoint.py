#!/usr/bin/env python3
"""Run one restoration model inside the isolated `verl` environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.io import write_png
from torchvision.transforms.functional import pil_to_tensor

RESULT_PREFIX = "RESULT_JSON="


class RestorationToolkitInstance(Protocol):
    """Runtime subset used from the copied verl restoration toolkit."""

    def load_single_model(self, model_name: str) -> object | None: ...

    def process_image_with_models(self, model_list: list[str], img_path: str, output_dir: str) -> str: ...


class RestorationToolkitFactory(Protocol):
    """Constructor interface for the dynamically imported toolkit."""

    def __call__(
        self,
        *,
        models: list[str],
        device: str,
        load_iqa: bool,
        preload: bool,
        auto_unload: bool,
    ) -> RestorationToolkitInstance: ...


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)
    return image.unsqueeze(0).to(device)


def _save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = image.squeeze(0).detach().clamp(0, 1).mul(255).byte().cpu()
    write_png(output, str(path))


def _pad_to_multiple(image: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return F.pad(image, (0, pad_width, 0, pad_height), mode="reflect"), height, width


def _load_nafnet(repo: Path, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    module = importlib.import_module("basicsr.models.archs.NAFNet_arch")
    model = module.NAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["params"], strict=True)
    return model


def _load_focalnet(repo: Path, checkpoint: Path, model_name: str) -> torch.nn.Module:
    branch = repo / ("Desnowing" if model_name == "focal_desnow" else "Dehazing/ITS")
    package_name = "stage_d_focalnet_models"
    package = ModuleType(package_name)
    package.__path__ = [str(branch / "models")]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.FocalNet", branch / "models/FocalNet.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load FocalNet module from {branch}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.build_net()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    return model


def _load_mb_taylorformer(repo: Path, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    module = importlib.import_module("basicsr.models.archs.MB_TaylorFormer")
    with (repo / "Dehazing/Options/MB-TaylorFormer-B.yml").open("r", encoding="utf-8") as config_file:
        network_config = yaml.safe_load(config_file)["network_g"]
    network_config.pop("type", None)
    model = module.MB_TaylorFormer(**network_config)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["params"], strict=True)
    return model


def _run_candidate(
    model_name: str,
    repo: Path,
    checkpoint: Path,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    if model_name == "nafnet_denoise":
        model = _load_nafnet(repo, checkpoint)
    elif model_name in {"focal_dehaze", "focal_desnow"}:
        model = _load_focalnet(repo, checkpoint, model_name)
    elif model_name == "mb_taylorformer_dehaze":
        model = _load_mb_taylorformer(repo, checkpoint)
    else:
        raise ValueError(f"unsupported candidate model: {model_name}")

    model = model.eval().to(device)
    image = _load_image(input_path, device)
    original_height, original_width = image.shape[-2:]
    if model_name.startswith("focal_"):
        image, original_height, original_width = _pad_to_multiple(image, 4)
    elif model_name == "mb_taylorformer_dehaze":
        image, original_height, original_width = _pad_to_multiple(image, 8)

    with torch.inference_mode():
        restored = model(image)
        if model_name.startswith("focal_"):
            restored = restored[2]
    restored = restored[:, :, :original_height, :original_width]
    _save_image(restored, output_path)


def _run_verl_toolkit(
    model_name: str,
    external_tools_root: Path,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    bundle = external_tools_root / "verl_bundle"
    agent_tools_dir = bundle / "agent_tools"
    source_directories = {
        "retinexformer_fivek": "Retinexformer",
        "lightdiff": "LightenDiffusion",
        "idt": "IDT",
        "ridcp": "RIDCP",
        "turbo_rain": "img2img_turbo",
        "turbo_snow": "img2img_turbo",
    }
    sys.path.insert(0, str(bundle))
    selected_source = source_directories.get(model_name)
    if selected_source:
        source_dir = agent_tools_dir / selected_source
        sys.path.insert(0, str(source_dir))
        if (source_dir / "src").is_dir():
            sys.path.insert(0, str(source_dir / "src"))

    toolkit_module = importlib.import_module("agent_tools.restoration_toolkit")
    toolkit_factory = cast(RestorationToolkitFactory, getattr(toolkit_module, "RestorationToolkit"))
    toolkit = toolkit_factory(
        models=[model_name],
        device=str(device),
        load_iqa=False,
        preload=False,
        auto_unload=True,
    )
    if toolkit.load_single_model(model_name) is None:
        raise RuntimeError(f"failed to load verl restoration model: {model_name}")
    with tempfile.TemporaryDirectory(prefix=f"{model_name}-", dir=output_path.parent) as temporary_dir:
        generated = Path(toolkit.process_image_with_models([model_name], str(input_path), temporary_dir)).resolve()
        if not generated.is_file():
            raise RuntimeError(f"verl restoration model did not produce an image: {generated}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, choices=["verl_toolkit", "candidate"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--external-tools-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    external_tools_root = args.external_tools_root.expanduser().resolve()
    local_packages = external_tools_root / "python_packages"
    if local_packages.is_dir():
        sys.path.insert(0, str(local_packages))
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    if args.adapter == "verl_toolkit":
        _run_verl_toolkit(args.model, external_tools_root, input_path, output_path, device)
    else:
        if args.repo is None or args.checkpoint is None:
            raise ValueError("candidate adapter requires --repo and --checkpoint")
        _run_candidate(
            args.model,
            args.repo.expanduser().resolve(),
            args.checkpoint.expanduser().resolve(),
            input_path,
            output_path,
            device,
        )
    torch.cuda.synchronize(device)
    result = {
        "status": "success",
        "adapter": args.adapter,
        "model": args.model,
        "output_path": str(output_path),
        "inference_seconds": time.perf_counter() - started,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "torch_version": torch.__version__,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
