"""Align legal action branches with original thinking tokens (no model forward)."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from omegaconf import OmegaConf

from .action_mentions import MENTION_SELECTIONS, render_action_case, select_action_mention

FIELD = "alfworld_branch_entropy"


def validate_branch_entropy_config(config: Any) -> None:
    """Reject unsupported execution paths before allocating a model or starting Ray."""

    def get(path: str, default: Any = None) -> Any:
        return OmegaConf.select(config, path, default=default)

    if not get("actor_rollout_ref.branch_entropy.enabled", False):
        return
    selection = get("actor_rollout_ref.branch_entropy.mention_selection", "first")
    if selection not in MENTION_SELECTIONS:
        raise ValueError(f"Unsupported branch entropy mention_selection: {selection}")
    coefficient = float(get("actor_rollout_ref.branch_entropy.coefficient", 0))
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("branch_entropy.coefficient must be finite and nonnegative")
    temperature = float(get("actor_rollout_ref.rollout.temperature", 0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Action branch entropy requires positive finite rollout.temperature")
    requirements = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "sync",
        "variables.TRAINING_BACKEND": "trajectory",
        "data.apply_chat_template_kwargs.enable_thinking": True,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 1,
        "algorithm.rollout_correction.bypass_mode": True,
    }
    for path, expected in requirements.items():
        if get(path, expected) != expected:
            raise ValueError(f"Action branch entropy requires {path}={expected!r}")
    if get("actor_rollout_ref.actor.entropy_coeff", 0.0) != 0 and not get(
        "actor_rollout_ref.actor.calculate_entropy", False
    ):
        raise ValueError("Nonzero actor.entropy_coeff requires actor.calculate_entropy=True")
    if get("actor_rollout_ref.actor.strategy") not in {"fsdp", "fsdp2"}:
        raise ValueError("Action branch entropy currently requires the native FSDP engine")
    if get("actor_rollout_ref.actor.fsdp_config.tensor_parallel_size", 1) != 1:
        raise ValueError("Action branch entropy currently requires actor tensor_parallel_size=1")
    if get("distillation.enabled", False):
        raise ValueError("Action branch entropy does not support distillation")


def locate_action_branches(
    tokenizer: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    action: str | None,
    admissible_actions: list[str],
    *,
    mention_selection: str = "first",
) -> dict[str, Any]:
    """Build a context-tokenized trie and retain branches on the sampled action path.

    ``mention_selection`` defaults to the first case-insensitive full mention. The opt-in
    first_non_listing mode skips recognized candidate lists before trie alignment.
    Positions index *target response tokens*, not logits. The engine subtracts one
    after adding prompt length. Token IDs are never substituted into the rollout.
    Candidates include the observed continuation's first boundary token, allowing
    a command to be distinguished from a longer command with the same prefix.
    """
    actions = sorted(set(admissible_actions))
    result: dict[str, Any] = {
        "status": "no_action",
        "nodes": [],
        "candidate_count": len(actions),
    }
    if not action:
        return result
    result["action"] = action
    if action not in actions:
        return {**result, "status": "not_admissible"}
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Action branch alignment requires a fast tokenizer with offset mappings")

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    prompt_text, response_text = decode(prompt_ids), decode(response_ids)
    close = response_text.find("</think>")
    if close < 0:
        return {**result, "status": "no_thinking"}
    open_tag = response_text.find("<think>", 0, close)
    if open_tag < 0 and prompt_text.rfind("<think>") <= prompt_text.rfind("</think>"):
        return {**result, "status": "no_thinking"}
    start = open_tag + len("<think>") if open_tag >= 0 else 0
    match, mention_info = select_action_mention(response_text[start:close], action, actions, mention_selection)
    result.update(mention_info)
    if match is None:
        return {**result, "status": "only_listing_mentions" if mention_info["mention_count"] else "no_mention"}

    text = prompt_text + response_text
    begin = len(prompt_text) + start + match.start()
    end = len(prompt_text) + start + match.end()
    observed = match.group()
    actual_ids = prompt_ids + response_ids
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    pivot = next((i for i, (a, b) in enumerate(offsets) if a <= begin < b), None)
    if pivot is None or pivot < len(prompt_ids):
        return {**result, "status": "token_mismatch"}

    paths: dict[str, list[int]] = {}
    for candidate in actions:
        rendered = render_action_case(candidate, observed)
        candidate_end = begin + len(rendered)
        encoded_candidate = tokenizer(
            text[:begin] + rendered + text[end:],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids, spans = encoded_candidate["input_ids"], encoded_candidate["offset_mapping"]
        # Do not discard inconvenient candidates: that would change the action space.
        if ids[:pivot] != actual_ids[:pivot]:
            return {**result, "status": "unstable_prefix"}
        boundary = next((i for i, (_, b) in enumerate(spans) if b > candidate_end), None)
        if boundary is None:
            return {**result, "status": "token_mismatch"}
        paths[candidate] = ids[pivot : boundary + 1]
    sampled = paths[action]
    if encoded["input_ids"][:pivot] != actual_ids[:pivot] or actual_ids[pivot : pivot + len(sampled)] != sampled:
        return {**result, "status": "token_mismatch"}
    # An observed prose continuation can itself spell a longer command. In that
    # case the first boundary token does not uniquely terminate the sampled path.
    if any(
        candidate != action and path[: min(len(path), len(sampled))] == sampled[: min(len(path), len(sampled))]
        for candidate, path in paths.items()
    ):
        return {**result, "status": "ambiguous_boundary"}

    nodes = []
    for depth, token in enumerate(sampled):
        next_tokens = sorted(
            {path[depth] for path in paths.values() if len(path) > depth and path[:depth] == sampled[:depth]}
        )
        if len(next_tokens) > 1:
            assert token in next_tokens
            nodes.append({"position": pivot + depth - len(prompt_ids), "token_ids": next_tokens})
    return {**result, "status": "matched", "nodes": nodes}


def trajectory_weights(keys: list[str], tags: list[dict[str, Any]], mini_batch_size: int) -> tuple[list[float], float]:
    """Equal episode weighting, stable across DP splits, microbatches and padding.

    Native V1 averages update metrics over optimizer minibatches. Multiplying by
    total_rows / mini_batch_size makes their mean the whole-rollout episode mean.
    Padding contributes zero. Missing alignments retain their episode denominator.
    """
    if len(keys) != len(tags) or not keys or mini_batch_size <= 0 or len(keys) % mini_batch_size:
        raise ValueError("Branch entropy requires a complete, minibatch-padded V1 update batch")
    sessions = [key.rsplit("_", 1)[0] for key in keys]
    counts = Counter(session for session, tag in zip(sessions, tags, strict=True) if not tag.get("is_padding", False))
    if not counts:
        raise ValueError("Branch entropy cannot normalize a batch containing only padding")
    weights = [
        0.0 if tag.get("is_padding", False) else 1.0 / counts[session]
        for session, tag in zip(sessions, tags, strict=True)
    ]
    normalizer = len(counts) * mini_batch_size / len(keys)
    return weights, normalizer
