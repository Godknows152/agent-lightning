# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import torch


def gather_tool_choice_candidate_logits(
    logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_leaf_counts: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    """Gather legal branch logits after no-padding micro-batch conversion.

    The rollout batch uses left-padded prompts and right-padded responses. The
    actor removes both paddings, then right-pads each micro-batch to its local
    maximum sequence length. Consequently, response token ``r`` is predicted
    at ``real_prompt_length - 1 + r`` for each sample, not at a shared padded
    prompt offset.
    """

    if logits.ndim != 3 or candidate_token_ids.ndim != 3:
        raise ValueError("Expected logits [B, S, V] and candidates [B, R, K]")
    if candidate_leaf_counts.shape != candidate_token_ids.shape:
        raise ValueError("Candidate token ids and leaf counts must have identical [B, R, K] shapes")

    batch_size, sequence_width, vocab_size = logits.shape
    response_width = candidate_token_ids.shape[1]
    if candidate_token_ids.shape[0] != batch_size:
        raise ValueError("Logits and candidates must have the same batch size")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
        raise ValueError("Expected attention mask [B, padded_prompt_and_response_width]")
    if prompt_width <= 0 or attention_mask.shape[1] < prompt_width + response_width:
        raise ValueError("Attention mask does not cover the padded prompt and response")

    prompt_lengths = attention_mask[:, :prompt_width].sum(dim=-1, dtype=torch.long)
    response_is_valid = attention_mask[:, prompt_width : prompt_width + response_width].to(bool)
    active_rows = candidate_leaf_counts.gt(0).any(dim=-1)
    if torch.any(active_rows & ~response_is_valid):
        raise ValueError("Tool-choice candidates are present on a padded response position")
    if torch.any(active_rows & prompt_lengths.unsqueeze(-1).eq(0)):
        raise ValueError("Tool-choice candidates require a non-empty prompt")

    response_offsets = torch.arange(response_width, device=logits.device)
    predictor_positions = prompt_lengths.to(logits.device).unsqueeze(-1) - 1 + response_offsets
    invalid_predictors = (predictor_positions < 0) | (predictor_positions >= sequence_width)
    if torch.any(active_rows.to(logits.device) & invalid_predictors):
        raise ValueError("Tool-choice predictor position is outside the actor logits")

    active_candidates = candidate_leaf_counts.gt(0).to(logits.device)
    candidate_token_ids = candidate_token_ids.to(logits.device)
    invalid_token_ids = (candidate_token_ids < 0) | (candidate_token_ids >= vocab_size)
    if torch.any(active_candidates & invalid_token_ids):
        raise ValueError("Tool-choice candidate token id is outside the actor vocabulary")

    safe_positions = predictor_positions.masked_fill(~active_rows.to(logits.device), 0)
    safe_token_ids = candidate_token_ids.masked_fill(~active_candidates, 0)
    batch_indices = torch.arange(batch_size, device=logits.device).view(-1, 1, 1)
    return logits[
        batch_indices,
        safe_positions.unsqueeze(-1).expand_as(safe_token_ids),
        safe_token_ids,
    ]


def restricted_tool_choice_entropy(
    logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_leaf_counts: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position entropy over legal tool-trie branches.

    The root entropy plus the conditional entropies on the sampled branch is a
    Monte Carlo estimate of the tool distribution's chain-rule entropy.
    """

    if logits.ndim != 3 or candidate_token_ids.ndim != 3:
        raise ValueError("Expected logits [B, R, V] and candidates [B, R, K]")
    safe_ids = candidate_token_ids.clamp(min=0)
    gathered = logits.gather(-1, safe_ids)
    return restricted_tool_choice_entropy_from_candidates(gathered, candidate_leaf_counts)


def restricted_tool_choice_entropy_from_candidates(
    candidate_logits: torch.Tensor,
    candidate_leaf_counts: torch.Tensor,
) -> torch.Tensor:
    """Compute tool entropy after the actor has gathered only candidate logits."""

    valid = candidate_leaf_counts > 0
    if candidate_logits.ndim != 3 or candidate_logits.shape != valid.shape:
        raise ValueError("Candidate logits and leaf counts must have identical [B, R, K] shapes")

    valid_count = valid.sum(dim=-1)
    masked_logits = candidate_logits.masked_fill(~valid, torch.finfo(candidate_logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~valid, 0.0)).sum(dim=-1)
    return entropy.masked_fill(valid_count < 2, 0.0)
