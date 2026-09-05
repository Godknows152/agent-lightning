"""Replay compact ALFWorld decision contexts without changing trajectory loss weights."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


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
    for key in ("multi_modal_inputs", "decision_action_token_ids", "decision_first_token_ids"):
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
    expanded["temperature"] = torch.as_tensor(temperatures, dtype=torch.float32, device=device)
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
