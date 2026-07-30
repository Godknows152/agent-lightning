# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import logging
import math
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated

logger = logging.getLogger(__name__)


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    if "prompt_length" in batch.batch and "response_length" in batch.batch:
        return dict(
            prompt_length=batch.batch["prompt_length"],
            response_length=batch.batch["response_length"],
        )

    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min, variance: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Actual tool-call statistics when available, otherwise chat-turn statistics
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_prompt_length = batch.batch["prompts"].shape[-1]
    max_response_length = batch.batch["responses"].shape[-1]

    response_mask = batch.batch["response_mask"].bool()

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    if non_aborted_sequence_score.numel() > 0:
        score_mean = torch.mean(non_aborted_sequence_score).detach().item()
        score_max = torch.max(non_aborted_sequence_score).detach().item()
        score_min = torch.min(non_aborted_sequence_score).detach().item()
    else:
        logger.warning("All samples are aborted, returning default score metrics")
        score_mean = score_max = score_min = float("nan")

    if non_aborted_sequence_reward.numel() > 0:
        reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
        reward_max = torch.max(non_aborted_sequence_reward).detach().item()
        reward_min = torch.min(non_aborted_sequence_reward).detach().item()
        reward_variance = torch.var(non_aborted_sequence_reward, correction=0).detach().item()
    else:
        logger.warning("All samples are aborted, returning default reward metrics")
        reward_mean = reward_max = reward_min = reward_variance = float("nan")

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if valid_adv.numel() > 0:
        adv_mean = torch.mean(valid_adv).detach().item()
        adv_max = torch.max(valid_adv).detach().item()
        adv_min = torch.min(valid_adv).detach().item()
    else:
        logger.warning("Response mask is all False, returning default advantage metrics")
        adv_mean = adv_max = adv_min = float("nan")

    if valid_returns.numel() > 0:
        returns_mean = torch.mean(valid_returns).detach().item()
        returns_max = torch.max(valid_returns).detach().item()
        returns_min = torch.min(valid_returns).detach().item()
    else:
        logger.warning("Response mask is all False, returning default return metrics")
        returns_mean = returns_max = returns_min = float("nan")

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        logger.warning("All samples are aborted, returning default response length metrics")
        non_aborted_response_length_mean = float("nan")
        non_aborted_response_length_max = float("nan")
        non_aborted_response_length_min = float("nan")
        non_aborted_response_length_clip_ratio = float("nan")

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        if valid_returns.numel() > 0 and valid_values.numel() > 0:
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            critic_value_metrics = {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
        else:
            logger.warning("Response mask is all False, returning default value metrics")
            critic_value_metrics = {
                "critic/values/mean": float("nan"),
                "critic/values/max": float("nan"),
                "critic/values/min": float("nan"),
                # vf explained var
                "critic/vf_explained_var": float("nan"),
            }
    else:
        critic_value_metrics = {}

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        "critic/rewards/variance": reward_variance,
        # adv
        "critic/advantages/mean": adv_mean,
        "critic/advantages/max": adv_max,
        "critic/advantages/min": adv_min,
        # returns
        "critic/returns/mean": returns_mean,
        "critic/returns/max": returns_max,
        "critic/returns/min": returns_min,
        **critic_value_metrics,
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Keep the existing SwanLab key stable while making it report actual tool calls
    # for agent-loop batches. Other workflows without tool-call data retain the
    # original chat-turn fallback.
    tool_call_counts = batch.non_tensor_batch.get("tool_call_counts")
    if tool_call_counts is not None:
        tool_call_counts = np.asarray(tool_call_counts, dtype=np.int64)
        metrics["num_turns/min"] = tool_call_counts.min()
        metrics["num_turns/max"] = tool_call_counts.max()
        metrics["num_turns/mean"] = tool_call_counts.mean()
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()
    elif "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    return metrics


RESTORATION_PENALTY_METRIC_NAMES = {
    "no_tool_call": "no_tool_call_count",
    "malformed_tool_call_xml": "malformed_tool_call_xml_count",
    "invalid_restoration_action": "invalid_restoration_action_count",
    "unknown_tool_name": "unknown_tool_name_count",
    "forged_user_or_assistant_role_after_tool_call": "forged_user_or_assistant_role_after_tool_call_count",
    "repeated_restoration_action": "repeated_restoration_action_count",
}


def _as_penalty_record_lists(value: Any, length: int) -> list[list[dict[str, Any]]]:
    """Normalize per-trajectory penalty records without inferring penalties from unrelated rewards."""

    if value is None:
        return [[] for _ in range(length)]
    array = np.asarray(value, dtype=object)
    if array.shape == ():
        array = np.full(length, array.item(), dtype=object)
    if len(array) != length:
        logger.warning("Expected penalty_records length %s, got %s; ignoring field", length, len(array))
        return [[] for _ in range(length)]

    records_by_trajectory: list[list[dict[str, Any]]] = []
    for item in array:
        if not isinstance(item, (list, tuple, np.ndarray)):
            records_by_trajectory.append([])
            continue
        records_by_trajectory.append([record for record in item if isinstance(record, dict)])
    return records_by_trajectory


def compute_restoration_penalty_metrics(batch: DataProto) -> dict[str, Any]:
    """Count actual negative reward events by their explicit cause for the current step."""

    batch_size = len(batch)
    if batch_size == 0:
        return {}

    reason_counts = {reason: 0 for reason in RESTORATION_PENALTY_METRIC_NAMES}
    for trajectory_records in _as_penalty_record_lists(
        batch.non_tensor_batch.get("penalty_records"),
        batch_size,
    ):
        for record in trajectory_records:
            reason = str(record.get("reason", "")).strip()
            try:
                value = float(record.get("value", 0.0))
                occurrences = max(1, int(record.get("occurrences", 1)))
            except (TypeError, ValueError):
                continue
            if not reason or value >= 0.0:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + occurrences

    metrics = {}
    for reason, count in reason_counts.items():
        metric_name = RESTORATION_PENALTY_METRIC_NAMES.get(reason, f"{reason}_count")
        metrics[f"restoration_penalty/{metric_name}"] = int(count)
    return metrics


def compute_restoration_reward_metrics(batch: DataProto) -> dict[str, Any]:
    """Aggregate the restoration reward components selected for tracking."""

    batch_size = len(batch)
    if batch_size == 0:
        return {}

    field_names = ("pure_image_restoration_reward", "other_penalty")
    arrays: dict[str, np.ndarray | None] = {}
    for field_name in field_names:
        values = batch.non_tensor_batch.get(field_name)
        if values is None:
            arrays[field_name] = None
            continue
        array = np.asarray(values, dtype=object)
        if array.shape == ():
            array = np.full(batch_size, array.item(), dtype=object)
        if len(array) != batch_size:
            logger.warning("Expected %s length %s, got %s; ignoring field", field_name, batch_size, len(array))
            arrays[field_name] = None
            continue
        arrays[field_name] = array

    if all(array is None for array in arrays.values()):
        return {}

    components = {field_name: [] for field_name in field_names}
    for index in range(batch_size):
        sample_values: dict[str, float] = {}
        has_component = False
        for field_name, array in arrays.items():
            item = None if array is None else array[index]
            if item is None:
                sample_values[field_name] = 0.0
                continue
            try:
                value = float(item)
            except (TypeError, ValueError):
                sample_values[field_name] = 0.0
                continue
            if np.isfinite(value):
                sample_values[field_name] = value
                has_component = True
            else:
                sample_values[field_name] = 0.0
        if has_component:
            for field_name in field_names:
                components[field_name].append(sample_values[field_name])

    if not components["pure_image_restoration_reward"]:
        return {}

    return {
        "restoration_reward/pure_image_reward_mean": float(np.mean(components["pure_image_restoration_reward"])),
        "restoration_reward/other_penalty_mean": float(np.mean(components["other_penalty"])),
    }


def _normalize_restoration_action_histories(value: Any, length: int) -> list[tuple[str, ...]]:
    """Normalize successfully executed action histories for batch-level reward shaping."""

    array = np.asarray(value, dtype=object)
    if array.shape == ():
        array = np.full(length, array.item(), dtype=object)
    if len(array) != length:
        raise ValueError(f"Expected action_history length {length}, got {len(array)}")

    histories: list[tuple[str, ...]] = []
    for item in array:
        if not isinstance(item, (list, tuple, np.ndarray)):
            histories.append(())
            continue
        histories.append(
            tuple(
                action.strip()
                for action in item
                if isinstance(action, str) and action.strip() and action.strip() != "stop"
            )
        )
    return histories


def _empirical_entropy(samples: list[Any]) -> float:
    """Return natural-log empirical Shannon entropy, or zero for no samples."""

    if not samples:
        return 0.0
    counts = Counter(samples)
    return float(-sum((count / len(samples)) * math.log(count / len(samples)) for count in counts.values()))


def compute_restoration_action_entropy_metrics(batch: DataProto) -> dict[str, Any]:
    """Measure batch-level entropy of successful non-stop restoration actions and paths."""

    if "action_history" not in batch.non_tensor_batch or len(batch) == 0:
        return {}
    try:
        histories = _normalize_restoration_action_histories(batch.non_tensor_batch["action_history"], len(batch))
    except ValueError as exc:
        logger.warning("Cannot compute restoration action entropy metrics: %s", exc)
        return {}

    valid_histories = [history for history in histories if history]
    action_choices = [action for history in valid_histories for action in history]
    return {
        "actor/tool_choice_entropy": _empirical_entropy(action_choices),
        "actor/action_path_entropy": _empirical_entropy(valid_histories),
    }


def apply_restoration_action_rarity_reward(
    batch: DataProto,
    *,
    coefficient: float,
    quality_gate: torch.Tensor,
    num_repair_actions: int = 16,
) -> tuple[DataProto, dict[str, Any]]:
    """Add a quality-gated empirical action-rarity reward to terminal response tokens.

    Each non-stop repair call receives normalized surprisal ``-log(c(a) / N) / log(K)``
    from the current global rollout batch. A trajectory uses the mean surprisal of its
    calls, so making more tool calls cannot increase the reward by itself.
    """

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("action_rarity_reward_coeff must be finite and non-negative")
    if coefficient == 0:
        return batch, {}
    if num_repair_actions < 2:
        raise ValueError("num_repair_actions must be at least 2")
    if "action_history" not in batch.non_tensor_batch:
        raise ValueError("action_history is required when action_rarity_reward_coeff is positive")
    if "token_level_rewards" not in batch.batch or "response_mask" not in batch.batch:
        raise ValueError("token_level_rewards and response_mask are required for action-rarity rewards")

    batch_size = len(batch)
    histories = _normalize_restoration_action_histories(batch.non_tensor_batch["action_history"], batch_size)
    counts = Counter(action for history in histories for action in history)
    sample_count = sum(counts.values())
    probabilities = {action: count / sample_count for action, count in counts.items()} if sample_count else {}
    empirical_entropy = _empirical_entropy([action for history in histories for action in history])

    normalizer = math.log(num_repair_actions)
    scores = np.zeros(batch_size, dtype=np.float32)
    valid_trajectory_mask = np.zeros(batch_size, dtype=bool)
    for index, history in enumerate(histories):
        if not history:
            continue
        valid_trajectory_mask[index] = True
        mean_surprisal = sum(-math.log(probabilities[action]) for action in history) / len(history)
        scores[index] = min(1.0, mean_surprisal / normalizer)

    token_level_rewards = batch.batch["token_level_rewards"]
    response_mask = batch.batch["response_mask"].to(device=token_level_rewards.device, dtype=torch.bool)
    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("token_level_rewards and response_mask must have identical [B, R] shapes")
    if quality_gate.ndim != 1 or quality_gate.numel() != batch_size:
        raise ValueError("quality_gate must have shape [B]")

    score_tensor = torch.as_tensor(scores, device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    gate_tensor = quality_gate.to(device=token_level_rewards.device, dtype=torch.bool)
    has_response = response_mask.any(dim=-1)
    sequence_rewards = coefficient * score_tensor * gate_tensor * has_response

    response_positions = torch.arange(response_mask.shape[1], device=response_mask.device).expand_as(response_mask)
    last_response_positions = response_positions.masked_fill(~response_mask, -1).max(dim=-1).values
    active_rows = sequence_rewards.ne(0) & last_response_positions.ge(0)
    bonus = torch.zeros_like(token_level_rewards)
    row_indices = torch.arange(batch_size, device=token_level_rewards.device)[active_rows]
    bonus[row_indices, last_response_positions[active_rows]] = sequence_rewards[active_rows]
    batch.batch["token_level_rewards"] = token_level_rewards + bonus

    batch.non_tensor_batch["action_rarity_score"] = scores
    batch.non_tensor_batch["action_rarity_reward"] = sequence_rewards.detach().float().cpu().numpy()

    valid_tensor = torch.as_tensor(valid_trajectory_mask, device=token_level_rewards.device, dtype=torch.bool)
    valid_count = int(valid_tensor.sum().item())
    gate_ratio = float((gate_tensor & valid_tensor).sum().item() / valid_count) if valid_count else 0.0
    valid_score_mean = float(score_tensor[valid_tensor].float().mean().item()) if valid_count else 0.0
    return batch, {
        "actor/tool_choice_entropy": float(empirical_entropy),
        "actor/tool_choice_sample_count": int(sample_count),
        "actor/tool_choice_unique_action_count": int(len(counts)),
        "actor/action_rarity_score_mean": valid_score_mean,
        "actor/action_rarity_reward_mean": float(sequence_rewards.float().mean().item()),
        "actor/action_rarity_reward_max": float(sequence_rewards.float().max().item()) if batch_size else 0.0,
        "actor/action_rarity_reward_coeff": float(coefficient),
        "actor/action_rarity_reward_gate_ratio": gate_ratio,
        "actor/action_rarity_valid_trajectory_rate": float(valid_count / batch_size) if batch_size else 0.0,
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val
