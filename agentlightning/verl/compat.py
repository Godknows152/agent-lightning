# Copyright (c) Microsoft. All rights reserved.

"""Compatibility shims for model-specific VERL worker behavior."""

from __future__ import annotations

from typing import Any, Callable

import torch
from omegaconf import DictConfig
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
)


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

    from verl.utils import activation_offload

    handler_class = activation_offload.AsyncDoubleBufferGroupOffloadHandler
    original = handler_class.synchronize_on_group_commit_forward
    if getattr(original, "_agentlightning_compatible", False):
        return
    compatible_sync = build_activation_offload_sync_patch(original)
    compatible_sync._agentlightning_compatible = True  # type: ignore[attr-defined]
    handler_class.synchronize_on_group_commit_forward = compatible_sync


def patch_verl_glm4v_position_ids() -> None:
    """Patch VERL 0.6.x GLM-4.1V handling in the worker process."""

    from verl.models.transformers import glm4v

    glm4v.process_position_ids = normalize_glm4v_position_ids
    if not getattr(glm4v._get_input_embeds, "_agentlightning_compatible", False):
        compatible_get_input_embeds = build_glm4v_input_embed_patch(glm4v._get_input_embeds)
        compatible_get_input_embeds._agentlightning_compatible = True  # type: ignore[attr-defined]
        glm4v._get_input_embeds = compatible_get_input_embeds


class AgentLightningActorRolloutRefWorker(ActorRolloutRefWorker):
    """Synchronous VERL worker with Agent Lightning model compatibility."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        patch_verl_glm4v_position_ids()
        patch_verl_activation_offload_window()
        super().__init__(config, role, **kwargs)


class AgentLightningAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Asynchronous VERL worker with Agent Lightning model compatibility."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        patch_verl_glm4v_position_ids()
        patch_verl_activation_offload_window()
        super().__init__(config, role, **kwargs)
