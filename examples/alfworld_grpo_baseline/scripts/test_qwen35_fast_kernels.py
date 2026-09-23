"""GPU regression checks for Qwen3.5 kernels and native veRL packed sequence boundaries.

Run with the training PYTHONPATH and CUDA_VISIBLE_DEVICES=0. This is separate from
the CPU-only test suite, whose conftest intentionally forbids CUDA initialization.
"""
from __future__ import annotations

import argparse
import copy
import json
from importlib.metadata import version

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from verl.models.transformers.qwen3_5 import qwen3_5_gated_delta_net_forward


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow BF16 rounding while detecting nonfinite results and boundary leakage."""
    assert torch.isfinite(actual).all(), f"{name}: nonfinite result"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    a, b = actual.float(), expected.float()
    relative_l2 = ((a - b).norm() / b.norm().clamp_min(1e-8)).item()
    assert relative_l2 < 0.03, f"{name}: relative L2 error {relative_l2} >= 0.03"
    print(json.dumps({"check": name, "relative_l2": relative_l2, "max_abs": (a - b).abs().max().item()}))


def check_delta_rule() -> None:
    lengths = (97, 65)
    shape = (1, sum(lengths), 4, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2 for _ in range(3)]
    g = -torch.rand(shape[:-1], device="cuda", dtype=torch.float32)
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
    fast_inputs = [x.detach().requires_grad_() for x in (q, k, v, g, beta)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
    cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
    out, _ = modeling.chunk_gated_delta_rule(
        *fast_inputs[:3], g=fast_inputs[3], beta=fast_inputs[4],
        cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
    )
    references = []
    start = 0
    for length in lengths:
        segment = [x[:, start:start + length] for x in ref_inputs]
        ref, _ = modeling.torch_chunk_gated_delta_rule(
            *segment[:3], g=segment[3], beta=segment[4], use_qk_l2norm_in_kernel=True,
        )
        references.append(ref)
        start += length
    reference = torch.cat(references, dim=1)
    compare("delta_rule/forward", out, reference)
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (reference * upstream).sum().backward()
    for name, fast, ref in zip(("q", "k", "v", "g", "beta"), fast_inputs, ref_inputs, strict=True):
        compare(f"delta_rule/grad_{name}", fast.grad, ref.grad)


def check_native_layer(model_path: str) -> None:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True).text_config
    config.dtype = torch.bfloat16
    packed_layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).cuda().to(torch.bfloat16).train()
    separate_layer = copy.deepcopy(packed_layer)
    lengths = (97, 65)
    x = torch.randn(1, sum(lengths), config.hidden_size, device="cuda", dtype=torch.bfloat16)
    packed_input = x.detach().requires_grad_()
    separate_input = x.detach().clone().requires_grad_()
    cu_cpu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
    packed = qwen3_5_gated_delta_net_forward(
        packed_layer, packed_input, cu_seqlens=cu_cpu.cuda(), cu_seqlens_cpu=cu_cpu,
    )
    separate = torch.cat([
        qwen3_5_gated_delta_net_forward(separate_layer, segment)
        for segment in separate_input.split(lengths, dim=1)
    ], dim=1)
    compare("native_layer/packed_forward", packed, separate)
    upstream = torch.randn_like(packed) / sum(lengths)
    (packed * upstream).sum().backward()
    (separate * upstream).sum().backward()
    compare("native_layer/input_gradient", packed_input.grad, separate_input.grad)
    for (name, p), (other_name, reference) in zip(
        packed_layer.named_parameters(), separate_layer.named_parameters(), strict=True,
    ):
        assert name == other_name and p.grad is not None and reference.grad is not None
        compare(f"native_layer/parameter_gradient/{name}", p.grad, reference.grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/LXJ/Python_Projects/Models/Qwen3.5-2B")
    args = parser.parse_args()
    assert modeling.is_fast_path_available, "Qwen3.5 fast path is not available"
    assert modeling.chunk_gated_delta_rule.__module__.startswith("fla.")
    torch.manual_seed(7)
    for package in ("flash-linear-attention", "fla-core", "causal-conv1d"):
        print(f"{package}={version(package)}")
    check_delta_rule()
    check_native_layer(args.model)
    torch.cuda.synchronize()
    print("qwen35_fast_kernels_gpu_test=passed")


if __name__ == "__main__":
    main()
