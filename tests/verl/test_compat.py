"""Tests for VERL model compatibility shims."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from agentlightning.verl.compat import (
    build_activation_offload_sync_patch,
    build_glm4v_input_embed_patch,
    normalize_glm4v_position_ids,
)


def test_normalize_glm4v_position_ids_removes_auxiliary_text_row() -> None:
    position_ids = torch.arange(4 * 2 * 5).reshape(4, 2, 5)

    normalized = normalize_glm4v_position_ids(position_ids)

    assert normalized.shape == (3, 2, 5)
    assert torch.equal(normalized, position_ids[1:])


def test_normalize_glm4v_position_ids_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="position_ids should have shape"):
        normalize_glm4v_position_ids(torch.zeros(3, 2, 5, dtype=torch.long))


def test_glm4v_input_embed_patch_avoids_in_place_leaf_update() -> None:
    class LeafEmbedding(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                (*input_ids.shape, 4),
                dtype=torch.float32,
                requires_grad=True,
            )

    class FakeVisual(nn.Module):
        def forward(self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
            assert pixel_values.shape == (16, 1176)
            assert grid_thw.tolist() == [[1, 4, 4]]
            return pixel_values.mean(dim=1, keepdim=True)

    class FakeModel:
        visual = FakeVisual()

        @staticmethod
        def get_input_embeddings() -> nn.Module:
            return LeafEmbedding()

    def unused_original(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        raise AssertionError("text-only transitions should use the compatibility path")

    get_input_embeds = build_glm4v_input_embed_patch(unused_original)
    embeds, attention_mask = get_input_embeds(
        FakeModel(),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.ones((1, 2), dtype=torch.long),
    )

    embeds.sum().backward()
    assert embeds.shape == (1, 2, 4)
    assert attention_mask is not None
    assert attention_mask.shape == (1, 2)


def test_activation_offload_patch_ignores_exhausted_window() -> None:
    calls: list[int] = []

    class FakeHandler:
        num_offload_group = 3
        offloaded_group_count = 3

    def original(handler: FakeHandler, current_group: int) -> None:
        calls.append(current_group)

    compatible_sync = build_activation_offload_sync_patch(original)
    compatible_sync(FakeHandler(), 63)

    assert calls == []


def test_activation_offload_patch_preserves_active_window() -> None:
    calls: list[int] = []

    class FakeHandler:
        num_offload_group = 3
        offloaded_group_count = 2

    def original(handler: FakeHandler, current_group: int) -> None:
        calls.append(current_group)

    compatible_sync = build_activation_offload_sync_patch(original)
    compatible_sync(FakeHandler(), 2)

    assert calls == [2]
