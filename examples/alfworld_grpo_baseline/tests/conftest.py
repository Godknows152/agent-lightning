"""Select native veRL before collection and detect later import contamination."""
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("ALFWORLD_VERL_ROOT", "/home/LXJ/Python_Projects/verl")).resolve()
sys.path[:0] = [str(ROOT / "src"), str(BACKEND)]

from alfworld_baseline.backend import assert_native_verl

assert_native_verl()


def pytest_collection_finish(session):
    assert_native_verl()


def pytest_runtest_setup(item):
    assert_native_verl()


import pytest

@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    import torch
    def fail(*args, **kwargs):
        pytest.fail("ALFWorld CPU tests must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)
