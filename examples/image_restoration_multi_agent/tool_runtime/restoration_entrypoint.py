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
READY_PREFIX = "READY_JSON="


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
    model = _load_candidate_model(model_name, repo, checkpoint, device)
    _run_loaded_candidate(model_name, model, input_path, output_path, device)


def _load_candidate_model(
    model_name: str,
    repo: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "nafnet_denoise":
        model = _load_nafnet(repo, checkpoint)
    elif model_name in {"focal_dehaze", "focal_desnow"}:
        model = _load_focalnet(repo, checkpoint, model_name)
    elif model_name == "mb_taylorformer_dehaze":
        model = _load_mb_taylorformer(repo, checkpoint)
    else:
        raise ValueError(f"unsupported candidate model: {model_name}")
    return model.eval().to(device)


def _run_loaded_candidate(
    model_name: str,
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
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


def _load_verl_toolkit(
    model_names: list[str],
    external_tools_root: Path,
    device: torch.device,
    *,
    preload: bool,
    auto_unload: bool,
) -> RestorationToolkitInstance:
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
    selected_sources = list(
        dict.fromkeys(source_directories[name] for name in model_names if name in source_directories)
    )
    for selected_source in selected_sources:
        source_dir = agent_tools_dir / selected_source
        sys.path.insert(0, str(source_dir))
        if (source_dir / "src").is_dir():
            sys.path.insert(0, str(source_dir / "src"))

    toolkit_module = importlib.import_module("agent_tools.restoration_toolkit")
    toolkit_factory = cast(RestorationToolkitFactory, getattr(toolkit_module, "RestorationToolkit"))
    toolkit = toolkit_factory(
        models=model_names,
        device=str(device),
        load_iqa=False,
        preload=preload,
        auto_unload=auto_unload,
    )
    missing = [model_name for model_name in model_names if toolkit.load_single_model(model_name) is None]
    if missing:
        raise RuntimeError(f"failed to load verl restoration models: {missing}")
    return toolkit


def _run_loaded_verl_toolkit(
    toolkit: RestorationToolkitInstance,
    model_name: str,
    input_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{model_name}-", dir=output_path.parent) as temporary_dir:
        generated = Path(toolkit.process_image_with_models([model_name], str(input_path), temporary_dir)).resolve()
        if not generated.is_file():
            raise RuntimeError(f"verl restoration model did not produce an image: {generated}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, output_path)


def _run_verl_toolkit(
    model_name: str,
    external_tools_root: Path,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    toolkit = _load_verl_toolkit(
        [model_name],
        external_tools_root,
        device,
        preload=False,
        auto_unload=True,
    )
    _run_loaded_verl_toolkit(toolkit, model_name, input_path, output_path)


def _emit(prefix: str, payload: dict[str, object]) -> None:
    try:
        print(f"{prefix}{json.dumps(payload, separators=(',', ':'))}", flush=True)
    except BrokenPipeError:
        sys.exit(0)


def _serve_jsonl(
    *,
    adapter: str,
    model_names: list[str],
    external_tools_root: Path,
    repo: Path | None,
    checkpoint: Path | None,
    device: torch.device,
) -> None:
    if adapter == "verl_toolkit":
        toolkit = _load_verl_toolkit(
            model_names,
            external_tools_root,
            device,
            preload=True,
            auto_unload=False,
        )
        candidate_model = None
    else:
        if len(model_names) != 1 or repo is None or checkpoint is None:
            raise ValueError("persistent candidate worker requires one model, --repo, and --checkpoint")
        toolkit = None
        candidate_model = _load_candidate_model(model_names[0], repo, checkpoint, device)

    torch.cuda.synchronize(device)
    _emit(
        READY_PREFIX,
        {
            "status": "ready",
            "adapter": adapter,
            "models": model_names,
            "device": str(device),
            "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        },
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id: object = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            model_name = str(request["model"])
            if model_name not in model_names:
                raise ValueError(f"model is not loaded by this worker: {model_name}")
            input_path = Path(request["input"]).expanduser().resolve()
            output_path = Path(request["output"]).expanduser().resolve()
            started = time.perf_counter()
            if toolkit is not None:
                _run_loaded_verl_toolkit(toolkit, model_name, input_path, output_path)
            else:
                assert candidate_model is not None
                _run_loaded_candidate(model_name, candidate_model, input_path, output_path, device)
            torch.cuda.synchronize(device)
            _emit(
                RESULT_PREFIX,
                {
                    "request_id": request_id,
                    "status": "success",
                    "adapter": adapter,
                    "model": model_name,
                    "output_path": str(output_path),
                    "inference_seconds": time.perf_counter() - started,
                    "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "torch_version": torch.__version__,
                },
            )
        except Exception as error:
            _emit(
                RESULT_PREFIX,
                {
                    "request_id": request_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, choices=["verl_toolkit", "candidate"])
    parser.add_argument("--model")
    parser.add_argument("--models")
    parser.add_argument("--external-tools-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()

    external_tools_root = args.external_tools_root.expanduser().resolve()
    local_packages = external_tools_root / "python_packages"
    if local_packages.is_dir():
        sys.path.insert(0, str(local_packages))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model_names = [item.strip() for item in (args.models or args.model or "").split(",") if item.strip()]
    if not model_names:
        raise ValueError("--model or --models must specify at least one model")
    repo = args.repo.expanduser().resolve() if args.repo is not None else None
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    if args.serve_jsonl:
        _serve_jsonl(
            adapter=args.adapter,
            model_names=model_names,
            external_tools_root=external_tools_root,
            repo=repo,
            checkpoint=checkpoint,
            device=device,
        )
        return
    if len(model_names) != 1 or args.input is None or args.output is None:
        raise ValueError("one-shot mode requires one model, --input, and --output")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_name = model_names[0]
    started = time.perf_counter()
    if args.adapter == "verl_toolkit":
        _run_verl_toolkit(model_name, external_tools_root, input_path, output_path, device)
    else:
        if repo is None or checkpoint is None:
            raise ValueError("candidate adapter requires --repo and --checkpoint")
        _run_candidate(
            model_name,
            repo,
            checkpoint,
            input_path,
            output_path,
            device,
        )
    torch.cuda.synchronize(device)
    result = {
        "status": "success",
        "adapter": args.adapter,
        "model": model_name,
        "output_path": str(output_path),
        "inference_seconds": time.perf_counter() - started,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "torch_version": torch.__version__,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
