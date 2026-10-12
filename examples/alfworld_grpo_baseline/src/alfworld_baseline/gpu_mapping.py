"""Keep Ray's physical GPU assignments consistent with a restricted CUDA view."""

from __future__ import annotations

import functools
import os


def logical_cuda_index(visible: str, assigned: str) -> int:
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if assigned not in devices:
        raise RuntimeError(f"Ray assigned GPU {assigned}, outside CUDA_VISIBLE_DEVICES={visible}")
    return devices.index(assigned)


def install_cuda_device_mapping() -> None:
    """Install only in ALFWorld processes; do not edit shared veRL or Ray files.

    With Ray NOSET enabled, all colocated FSDP/SGLang processes share the same
    visible GPU list for CUDA IPC. Native veRL uses Ray's physical ID directly
    as a torch device ordinal, which fails for subsets such as ``0,3``.
    """
    import ray
    from verl.single_controller.base.worker import Worker
    from verl.utils.device import get_torch_device
    from verl.utils.ray_utils import ray_noset_visible_devices

    original = Worker._setup_env_cuda_visible_devices
    if getattr(original, "_alfworld_device_mapping", False):
        return

    @functools.wraps(original)
    def setup(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if (
            not visible
            or not ray_noset_visible_devices()
            or os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("ROCR_VISIBLE_DEVICES")
        ):
            return original(self)
        assigned = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
        if not assigned:
            return original(self)
        local_rank = logical_cuda_index(visible, str(assigned[0]))
        os.environ["LOCAL_RANK"] = str(local_rank)
        get_torch_device().set_device(local_rank)

    setup._alfworld_device_mapping = True
    Worker._setup_env_cuda_visible_devices = setup
