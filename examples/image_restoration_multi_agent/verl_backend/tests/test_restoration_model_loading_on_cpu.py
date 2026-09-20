"""Restoration must not continue with missing or unloaded model weights."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "restoration_tools" / "agent_tools"


def load_module(name, path):
    if not path.is_file():
        pytest.skip(f"Optional restoration backend is not installed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["IDT", "LightenDiffusion"])
def test_diffusion_rejects_missing_checkpoint(tmp_path, backend):
    module = load_module(f"test_{backend}_restoration", TOOLS / backend / "models" / "restoration.py")
    diffusion = Mock()
    checkpoint = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError, match=str(checkpoint)):
        module.DiffusiveRestoration(diffusion, SimpleNamespace(resume=str(checkpoint)), SimpleNamespace())

    diffusion.load_ddm_ckpt.assert_not_called()
    diffusion.model.eval.assert_not_called()


@pytest.mark.parametrize("backend,ema", [("IDT", True), ("LightenDiffusion", False)])
def test_diffusion_loads_weights_before_evaluation(tmp_path, backend, ema):
    module = load_module(f"test_{backend}_restoration", TOOLS / backend / "models" / "restoration.py")
    diffusion = Mock()
    checkpoint = tmp_path / "weights.pth"
    checkpoint.touch()

    module.DiffusiveRestoration(diffusion, SimpleNamespace(resume=str(checkpoint)), SimpleNamespace())

    diffusion.load_ddm_ckpt.assert_called_once_with(str(checkpoint), ema=ema)
    diffusion.model.eval.assert_called_once_with()
    assert [call[0] for call in diffusion.mock_calls] == ["load_ddm_ckpt", "model.eval"]


@pytest.mark.parametrize("preload", [False, True])
def test_toolkit_preserves_model_loading_error(tmp_path, monkeypatch, preload):
    monkeypatch.setenv("VERL_LOG_DIR", str(tmp_path))
    module = load_module("test_toolkit.restoration_toolkit", TOOLS / "restoration_toolkit.py")
    inference = ModuleType("test_toolkit.SCUNet.inference")
    failure = FileNotFoundError("missing SCUNet checkpoint")
    inference.load_scu_model = Mock(side_effect=failure)
    monkeypatch.setitem(sys.modules, "test_toolkit.SCUNet.inference", inference)
    with pytest.raises(RuntimeError, match="scunet.*missing SCUNet checkpoint") as raised:
        toolkit = module.RestorationToolkit(models=["scunet"], device="cpu", load_iqa=False, preload=preload)
        toolkit.load_single_model("scunet")

    assert raised.value.__cause__ is failure
