"""Replay compact ALFWorld decision contexts without changing trajectory loss weights."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def _select_expanded_rows(expanded: TensorDict, row_indices: list[int]) -> TensorDict:
    """Select nested replay rows while preserving the scalar non-tensor metadata."""

    device = expanded["input_ids"].device
    chunk = TensorDict({}, batch_size=[len(row_indices)], device=device)
    for key in ("input_ids", "position_ids"):
        rows = [expanded[key][index] for index in row_indices]
        chunk[key] = torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    temperature = tu.get_non_tensor_data(expanded, "temperature", None)
    if temperature is not None:
        # Uniform temperatures are stored as scalar non-tensor metadata so the
        # fused PPO head can consume them without a per-sample tensor.
        tu.assign_non_tensor_data(chunk, "temperature", temperature)
    elif "temperature" in expanded.keys():
        chunk["temperature"] = expanded["temperature"][row_indices]

    for key, default in (
        ("use_remove_padding", True),
        ("use_fused_kernels", False),
        ("calculate_entropy", False),
        ("pad_mode", "no_padding"),
    ):
        tu.assign_non_tensor_data(chunk, key, tu.get_non_tensor_data(expanded, key, default))
    return chunk


def iter_turn_context_chunks(
    batch: TensorDict, max_tokens: int | None
) -> list[tuple[TensorDict, torch.Tensor, torch.Tensor]]:
    """Split replay turns into bounded token chunks.

    A turn is kept atomic so its prompt and generated action are always scored
    together. A turn longer than ``max_tokens`` is emitted as a single chunk;
    splitting inside an action would invalidate the replay mapping.
    """

    expanded, source_indices, target_indices = expand_turn_contexts(batch)
    if max_tokens is None or max_tokens <= 0:
        return [(expanded, source_indices, target_indices)]

    sequence_lengths = expanded["input_ids"].offsets().diff().tolist()
    chunks: list[tuple[TensorDict, torch.Tensor, torch.Tensor]] = []
    current_rows: list[int] = []
    current_tokens = 0
    source_sequence_ids = torch.bucketize(
        source_indices,
        expanded["input_ids"].offsets()[1:-1],
        right=True,
    )
    for row, sequence_length in enumerate(sequence_lengths):
        if current_rows and current_tokens + sequence_length > max_tokens:
            row_mask = (source_sequence_ids >= current_rows[0]) & (source_sequence_ids <= current_rows[-1])
            # Source indices address the flattened full expanded batch. The
            # selected TensorDict starts at zero, so rebase them to this chunk.
            source_start = expanded["input_ids"].offsets()[current_rows[0]]
            chunks.append(
                (
                    _select_expanded_rows(expanded, current_rows),
                    source_indices[row_mask] - source_start,
                    target_indices[row_mask],
                )
            )
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += sequence_length

    if current_rows:
        row_mask = (source_sequence_ids >= current_rows[0]) & (source_sequence_ids <= current_rows[-1])
        source_start = expanded["input_ids"].offsets()[current_rows[0]]
        chunks.append(
            (
                _select_expanded_rows(expanded, current_rows),
                source_indices[row_mask] - source_start,
                target_indices[row_mask],
            )
        )
    return chunks


def chunk_response_mask(batch: TensorDict, target_indices: torch.Tensor) -> torch.Tensor:
    """Build the response mask for one replay chunk in original trajectory space."""

    response_mask = batch["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(0)
    chunk_mask = torch.zeros_like(response_mask)
    records = batch["alfworld_turn_contexts"]
    if hasattr(records, "tolist"):
        records = records.tolist()
    sequence_offsets = batch["input_ids"].offsets()
    for row, turns in enumerate(records):
        row_start = int(sequence_offsets[row])
        row_end = int(sequence_offsets[row + 1])
        row_targets = target_indices[(target_indices >= row_start) & (target_indices < row_end)]
        if row_targets.numel() == 0:
            continue
        prompt_length = len(turns[0]["prompt_ids"])
        response_positions = row_targets - row_start - prompt_length + 1
        valid = (response_positions >= 0) & (response_positions < chunk_mask.shape[1])
        chunk_mask[row, response_positions[valid].long()] = 1
    return chunk_mask


def expand_turn_contexts(batch: TensorDict) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    """Pack each recorded prompt+answer independently; map predictions back to the trajectory.

    The original batch remains the loss/reward batch. Only model inputs expand,
    so GRPO groups, token masks and trajectory-level normalization are unchanged.
    Position IDs restart at zero for every decision (also resetting packed
    attention/recurrent state). Mapping uses next-token prediction positions.
    """
    original = batch["input_ids"]
    if not original.is_nested:
        raise ValueError("Turn-context replay requires padding-free inputs")
    if not tu.get_non_tensor_data(batch, "use_remove_padding", True):
        raise ValueError("Turn-context replay requires use_remove_padding=True")
    # ALFWorld is text-only, but the shared dataset pipeline may still add an
    # empty ``multi_modal_inputs`` placeholder. Ignore only placeholders; fail
    # closed if actual image/video payloads are present.
    if "multi_modal_inputs" in batch:
        multimodal = batch["multi_modal_inputs"]
        if hasattr(multimodal, "tolist"):
            multimodal = multimodal.tolist()

        def _has_payload(value):
            if value is None:
                return False
            if isinstance(value, dict):
                return any(_has_payload(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(_has_payload(item) for item in value)
            return True

        if _has_payload(multimodal):
            raise ValueError("Turn-context replay does not support non-empty multi_modal_inputs")
    for key in ("decision_action_token_ids", "decision_first_token_ids"):
        if key in batch:
            raise ValueError(f"Turn-context replay does not support {key}")
    records = batch["alfworld_turn_contexts"]
    if hasattr(records, "tolist"):
        records = records.tolist()
    device = original.device
    sequences, positions, temperatures = [], [], []
    source_indices, target_indices = [], []
    source_offset = 0
    temperature = batch["temperature"]
    for row, turns in enumerate(records):
        if not turns:
            raise ValueError("Missing ALFWorld per-turn contexts")
        row_tokens = original[row]
        prompt_length = len(turns[0]["prompt_ids"])
        target_offset = int(original.offsets()[row])
        covered = torch.zeros(row_tokens.numel(), dtype=torch.bool, device=device)
        for turn in turns:
            prompt, response = turn["prompt_ids"], turn["response_ids"]
            start = prompt_length + int(turn["response_offset"])
            # The trajectory output can truncate its final response to capacity.
            count = min(len(response), row_tokens.numel() - start)
            if count <= 0:
                continue
            response = response[:count]
            expected = torch.tensor(response, dtype=torch.long, device=device)
            if not torch.equal(row_tokens[start : start + count], expected):
                raise ValueError("Recorded response does not match trajectory tokens")
            if not prompt or start < 1 or covered[start - 1 : start + count - 1].any():
                raise ValueError("Invalid/overlapping turn context mapping")
            tokens = torch.tensor(prompt + response, dtype=torch.long, device=device)
            sequences.append(tokens)
            positions.append(torch.arange(tokens.numel(), device=device))
            temperatures.append(temperature[row] if isinstance(temperature, torch.Tensor) else temperature)
            source_indices.extend(range(source_offset + len(prompt) - 1, source_offset + len(prompt) + count - 1))
            target_indices.extend(range(target_offset + start - 1, target_offset + start + count - 1))
            covered[start - 1 : start + count - 1] = True
            source_offset += tokens.numel()
        # This backend keeps loss_mask as a padded response-only tensor.
        # Translate its slots into unpadded next-token prediction positions.
        required = torch.zeros_like(covered)
        response_length = row_tokens.numel() - prompt_length
        required[prompt_length - 1 : row_tokens.numel() - 1] = batch["loss_mask"][row, :response_length].bool()
        if (required & ~covered).any():
            raise ValueError("A loss-bearing token has no matching rollout context")
    if not sequences:
        raise ValueError("No replayable ALFWorld responses")
    expanded = TensorDict({}, batch_size=[len(sequences)], device=device)
    expanded["input_ids"] = torch.nested.as_nested_tensor(sequences, layout=torch.jagged)
    # Sharing offsets preserves the nested symbolic sequence dimension.
    expanded["position_ids"] = torch.nested.nested_tensor_from_jagged(
        torch.cat(positions), expanded["input_ids"].offsets()
    )
    temperature_tensor = torch.as_tensor(temperatures, dtype=torch.float32, device=device)
    # The fused PPO head accepts one scalar temperature. ALFWorld normally uses
    # one rollout temperature for every trajectory, so preserve that value as a
    # scalar instead of materializing an equivalent per-sample tensor.
    if temperature_tensor.numel() and torch.all(temperature_tensor == temperature_tensor[0]):
        tu.assign_non_tensor_data(expanded, "temperature", float(temperature_tensor[0].item()))
    else:
        expanded["temperature"] = temperature_tensor
    for key, default in (
        ("use_remove_padding", True), ("use_fused_kernels", False),
        ("calculate_entropy", False), ("pad_mode", "no_padding"),
    ):
        tu.assign_non_tensor_data(expanded, key, tu.get_non_tensor_data(batch, key, default))
    if tu.get_non_tensor_data(batch, "distillation_use_topk", False):
        raise ValueError("Turn-context replay does not support distillation")
    return expanded, torch.tensor(source_indices, device=device), torch.tensor(target_indices, device=device)


def restore_trajectory_outputs(
    outputs: dict, original: TensorDict, source_indices: torch.Tensor, target_indices: torch.Tensor
) -> dict:
    """Differentiably scatter per-decision logprobs/entropy into original loss slots."""
    restored = {}
    input_ids = original["input_ids"]
    for key, value in outputs.items():
        if not isinstance(value, torch.Tensor) or not value.is_nested:
            raise ValueError(f"Unsupported turn-context output: {key}")
        values = value.values()
        flat = values.new_zeros(input_ids.values().numel()).index_copy(
            0, target_indices, values.index_select(0, source_indices)
        )
        restored[key] = torch.nested.nested_tensor_from_jagged(flat, input_ids.offsets())
    return restored
