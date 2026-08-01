# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.diffusion.diffusion_algos import kl_penalty_image
from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
    decision_action_sequence_entropy = model_output.get("decision_action_sequence_entropy", None)
    decision_action_normalized_entropy = model_output.get("decision_action_normalized_entropy", None)
    decision_first_token_action_entropy = model_output.get("decision_first_token_action_entropy", None)
    decision_first_token_normalized_entropy = model_output.get("decision_first_token_action_entropy_normalized", None)
    decision_first_token_effective_action_count = model_output.get("decision_first_token_effective_action_count", None)
    decision_first_token_legal_mass = model_output.get("decision_first_token_legal_mass", None)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    # The mask carries the global count used to average per-decision action entropies.
    decision_action_entropy_enabled = config.decision_point_entropy_coeff > 0.0
    if decision_action_entropy_enabled and "decision_point_mask" not in data:
        raise ValueError("decision_point_mask is required when decision_point_entropy_coeff is positive")
    has_dp_mask = decision_action_entropy_enabled
    batch_num_decision_points = None
    if has_dp_mask:
        fields.append("decision_point_mask")
        batch_num_decision_points = tu.get_non_tensor_data(data=data, key="batch_num_decision_points", default=None)
        if batch_num_decision_points is None:
            raise ValueError("batch_num_decision_points is required when decision_point_mask is present")
        if decision_action_sequence_entropy is None or decision_action_normalized_entropy is None:
            raise ValueError(
                "current-policy decision action-sequence entropies are required when "
                "decision_point_entropy_coeff is positive"
            )
    decision_first_token_entropy_enabled = getattr(config, "decision_point_first_token_entropy_coeff", 0.0) > 0.0
    if decision_first_token_entropy_enabled and "decision_first_token_found" not in data:
        raise ValueError(
            "decision_first_token_found is required when decision_point_first_token_entropy_coeff is positive"
        )
    batch_num_decision_trajectories = None
    if decision_first_token_entropy_enabled:
        fields.append("decision_first_token_found")
        batch_num_decision_trajectories = tu.get_non_tensor_data(
            data=data,
            key="batch_num_decision_trajectories",
            default=None,
        )
        if batch_num_decision_trajectories is None:
            raise ValueError("batch_num_decision_trajectories is required for decision first-token entropy")
        if any(
            value is None
            for value in (
                decision_first_token_action_entropy,
                decision_first_token_normalized_entropy,
                decision_first_token_effective_action_count,
                decision_first_token_legal_mass,
            )
        ):
            raise ValueError(
                "current-policy first-token action entropy outputs are required when "
                "decision_point_first_token_entropy_coeff is positive"
            )
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # Add normalized categorical entropy over complete legal action strings.
    if has_dp_mask:
        if batch_num_decision_points > 0:
            dp_size = config.global_batch_info["dp_size"]
            dp_action_sequence_entropy = (
                decision_action_sequence_entropy.values().sum() / batch_num_decision_points * dp_size
            )
            dp_entropy_loss = decision_action_normalized_entropy.values().sum() / batch_num_decision_points * dp_size
            policy_loss -= config.decision_point_entropy_coeff * dp_entropy_loss
        else:
            dp_action_sequence_entropy = log_prob.sum() * 0.0
            dp_entropy_loss = log_prob.sum() * 0.0
        metrics["actor/decision_point_entropy"] = Metric(value=dp_entropy_loss, aggregation=metric_aggregation)
        metrics["actor/decision_point_action_sequence_entropy"] = Metric(
            value=dp_action_sequence_entropy, aggregation=metric_aggregation
        )

    # Average all valid decisions within each trajectory before the global trajectory mean.
    if decision_first_token_entropy_enabled:
        first_token_found = data["decision_first_token_found"].to(torch.bool)
        local_decision_counts = first_token_found.sum(dim=-1)
        local_valid_trajectories = local_decision_counts > 0

        def global_trajectory_mean(values: torch.Tensor) -> torch.Tensor:
            per_trajectory = (values * first_token_found).sum(dim=-1) / local_decision_counts.clamp(min=1)
            return (
                per_trajectory[local_valid_trajectories].sum()
                / batch_num_decision_trajectories
                * config.global_batch_info["dp_size"]
            )

        if batch_num_decision_trajectories > 0:
            first_token_raw_entropy = global_trajectory_mean(decision_first_token_action_entropy)
            first_token_normalized_entropy = global_trajectory_mean(decision_first_token_normalized_entropy)
            first_token_effective_action_count = global_trajectory_mean(decision_first_token_effective_action_count)
            first_token_legal_mass = global_trajectory_mean(decision_first_token_legal_mass)
            policy_loss -= config.decision_point_first_token_entropy_coeff * first_token_normalized_entropy
        else:
            zero = log_prob.sum() * 0.0
            first_token_raw_entropy = zero
            first_token_normalized_entropy = zero
            first_token_effective_action_count = zero
            first_token_legal_mass = zero

        metrics["actor/decision_point_first_token_action_entropy"] = Metric(
            value=first_token_raw_entropy,
            aggregation=metric_aggregation,
        )
        metrics["actor/decision_point_first_token_action_entropy_normalized"] = Metric(
            value=first_token_normalized_entropy,
            aggregation=metric_aggregation,
        )
        metrics["actor/decision_point_first_token_effective_action_count"] = Metric(
            value=first_token_effective_action_count,
            aggregation=metric_aggregation,
        )
        metrics["actor/decision_point_legal_first_token_mass"] = Metric(
            value=first_token_legal_mass,
            aggregation=metric_aggregation,
        )
        # Compatibility alias; v4.1.1 runs must use the explicit metric above for comparisons.
        metrics["actor/decision_point_entropy"] = Metric(
            value=first_token_normalized_entropy,
            aggregation=metric_aggregation,
        )

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics


def diffusion_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
