# Copyright (c) Microsoft. All rights reserved.

"""Compatibility shims for model-specific VERL worker behavior."""

from __future__ import annotations

from typing import Any, Callable

import torch
from omegaconf import DictConfig

try:
    from verl.workers.fsdp_workers import (
        ActorRolloutRefWorker,
        AsyncActorRolloutRefWorker,
    )
except ModuleNotFoundError:
    # VERL 0.8 moved the unified FSDP actor/rollout/ref worker to
    # engine_workers and folded async rollout handling into the same class.
    from verl.workers.engine_workers import ActorRolloutRefWorker

    AsyncActorRolloutRefWorker = ActorRolloutRefWorker


def normalize_glm4v_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Remove Agent Lightning's auxiliary text row before GLM-4.1V forward."""

    if position_ids.ndim != 3 or position_ids.size(0) != 4:
        raise ValueError("position_ids should have shape (4, batch_size, sequence_length)")
    return position_ids[1:]


def build_glm4v_input_embed_patch(
    original: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Avoid VERL's in-place dummy vision dependency for text-only transitions."""

    def compatible_get_input_embeds(
        model: Any,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if pixel_values is not None or pixel_values_videos is not None:
            return original(
                model,
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        inputs_embeds = model.get_input_embeddings()(input_ids)
        dummy_pixels = torch.zeros(
            (16, 1176),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        dummy_grid = torch.tensor(
            [[1, 4, 4]],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        image_embeds = model.visual(dummy_pixels, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        return inputs_embeds, attention_mask

    return compatible_get_input_embeds


def build_activation_offload_sync_patch(
    original: Callable[[Any, int], None],
) -> Callable[[Any, int], None]:
    """Ignore final layer commits after every configured offload window."""

    def compatible_sync(handler: Any, current_group: int) -> None:
        if handler.offloaded_group_count >= handler.num_offload_group:
            return
        original(handler, current_group)

    return compatible_sync


def patch_verl_activation_offload_window() -> None:
    """Guard VERL 0.6.x against indexing past the final offload window."""

    try:
        from verl.utils import activation_offload
    except ImportError:
        return

    handler_class = activation_offload.AsyncDoubleBufferGroupOffloadHandler
    original = handler_class.synchronize_on_group_commit_forward
    if getattr(original, "_agentlightning_compatible", False):
        return
    compatible_sync = build_activation_offload_sync_patch(original)
    compatible_sync._agentlightning_compatible = True  # type: ignore[attr-defined]
    handler_class.synchronize_on_group_commit_forward = compatible_sync


def patch_flash_attn_padding_fallback() -> None:
    """Provide torch fallbacks for VERL padding helpers when flash-attn is unavailable.

    VERL 0.8 routes old-log-prob/ref-log-prob/update through no-padding helpers.
    Those helpers import ``flash_attn.bert_padding`` even when model attention itself
    uses SDPA. Qwen3.5 smoke tests only need the padding utilities, so falling back
    to equivalent PyTorch indexing avoids a fragile flash-attn ABI dependency.
    """

    try:
        import flash_attn.bert_padding  # noqa: F401

        return
    except Exception:
        pass

    try:
        from einops import rearrange as einops_rearrange
        from verl.utils import attention_utils
    except ImportError:
        return

    def torch_index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return input_tensor[indices]

    def torch_unpad_input(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        unused_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        del unused_mask
        batch_size, seqlen = attention_mask.shape
        flat_mask = attention_mask.reshape(-1).to(torch.bool)
        indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        hidden_states_flat = hidden_states.reshape(batch_size * seqlen, *hidden_states.shape[2:])
        hidden_states_unpad = torch_index_first_axis(hidden_states_flat, indices)
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
        return hidden_states_unpad, indices, cu_seqlens, max_seqlen

    def torch_pad_input(
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
    ) -> torch.Tensor:
        output = hidden_states.new_zeros((batch_size * seqlen, *hidden_states.shape[1:]))
        output[indices] = hidden_states
        return output.reshape(batch_size, seqlen, *hidden_states.shape[1:])

    def torch_get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
        return torch_index_first_axis, torch_pad_input, einops_rearrange, torch_unpad_input

    if getattr(attention_utils._get_attention_functions, "_agentlightning_torch_fallback", False):
        return
    torch_get_attention_functions._agentlightning_torch_fallback = True  # type: ignore[attr-defined]
    attention_utils._get_attention_functions = torch_get_attention_functions


def patch_verl_glm4v_position_ids() -> None:
    """Patch VERL 0.6.x GLM-4.1V handling in the worker process."""

    try:
        from verl.models.transformers import glm4v
    except ImportError:
        return

    glm4v.process_position_ids = normalize_glm4v_position_ids
    if not getattr(glm4v._get_input_embeds, "_agentlightning_compatible", False):
        compatible_get_input_embeds = build_glm4v_input_embed_patch(glm4v._get_input_embeds)
        compatible_get_input_embeds._agentlightning_compatible = True  # type: ignore[attr-defined]
        glm4v._get_input_embeds = compatible_get_input_embeds


class AgentLightningActorRolloutRefWorker(ActorRolloutRefWorker):
    """Synchronous VERL worker with Agent Lightning model compatibility."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        patch_flash_attn_padding_fallback()
        patch_verl_glm4v_position_ids()
        patch_verl_activation_offload_window()
        super().__init__(config, role, **kwargs)


class AgentLightningAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Asynchronous VERL worker with Agent Lightning model compatibility."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        patch_flash_attn_padding_fallback()
        patch_verl_glm4v_position_ids()
        patch_verl_activation_offload_window()
        super().__init__(config, role, **kwargs)
