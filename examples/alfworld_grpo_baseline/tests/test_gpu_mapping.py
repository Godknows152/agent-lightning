"""Physical GPU selections must never be used as remapped CUDA ordinals."""

from types import SimpleNamespace

import pytest

from alfworld_baseline.gpu_mapping import install_cuda_device_mapping, logical_cuda_index


@pytest.mark.parametrize(
    "visible,physical,logical", [("0,3", "3", 1), ("0,3", "0", 0), ("3,0", "0", 1), ("0,1", "1", 1)]
)
def test_logical_cuda_index(visible, physical, logical):
    assert logical_cuda_index(visible, physical) == logical


def test_unassigned_device_is_rejected():
    with pytest.raises(RuntimeError, match="outside"):
        logical_cuda_index("0,3", "1")


def test_worker_mapping_preserves_shared_visible_list(monkeypatch):
    from verl.single_controller.base.worker import Worker

    calls = []
    monkeypatch.setattr(Worker, "_setup_env_cuda_visible_devices", lambda self: calls.append("original"))
    monkeypatch.setattr("ray.get_runtime_context", lambda: SimpleNamespace(get_accelerator_ids=lambda: {"GPU": ["3"]}))
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: SimpleNamespace(set_device=calls.append))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3")
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    install_cuda_device_mapping()
    setup = Worker._setup_env_cuda_visible_devices
    install_cuda_device_mapping()
    assert Worker._setup_env_cuda_visible_devices is setup
    setup(SimpleNamespace())
    import os

    assert calls == [1]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,3"
    assert os.environ["LOCAL_RANK"] == "1"
