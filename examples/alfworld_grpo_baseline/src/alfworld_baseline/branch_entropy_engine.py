"""Opt-in native FSDP extension; reuse actor logits without patching shared veRL."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import partial
import math
from typing import Any, Callable

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import TrainingWorker
from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.config.model import HFModelConfig

from .branch_entropy import FIELD, trajectory_weights
from .workers import ALFWorldActorRolloutRefWorker


def branch_statistics(logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
    """Compute differentiable legal-branch entropy from packed raw logits, SP=TP=1.

    Advanced indexing copies only selected candidate logits. Backward through this
    indexing does not save/read the original logits values, so native Flash CE's
    in-place backward is safe. Legal mass is diagnostic only (detached).
    """
    packed = logits.squeeze(0)
    if packed.ndim != 2:
        raise ValueError("Branch entropy expects packed [1, total_tokens, vocabulary] logits")
    offsets = data["input_ids"].offsets().tolist()
    response_offsets = data["response_mask"].offsets().tolist()
    # One small host transfer per field, rather than a GPU synchronization per node.
    input_ids = data["input_ids"].values().detach().cpu().tolist()
    response_masks = data["response_mask"].values().detach().cpu().tolist()
    weights = data["branch_entropy_weight"].detach().cpu().tolist()
    rows = list(data["extra_fields"])
    temperatures = data["temperature"]
    if isinstance(temperatures, torch.Tensor):
        temperatures = temperatures.detach().cpu().tolist()
    else:
        temperatures = [float(temperatures)] * len(rows)
    node_rows, node_positions, node_temperatures, widths = [], [], [], []
    flat_positions, flat_candidates, flat_temperatures = [], [], []
    for row, extra in enumerate(rows):
        metadata = extra[FIELD]
        # Native V1 padding inherits metadata but replaces the original tokens.
        nodes = metadata["nodes"] if weights[row] > 0 else []
        response_length = response_offsets[row + 1] - response_offsets[row]
        prompt_length = offsets[row + 1] - offsets[row] - response_length
        temperature = float(temperatures[row])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Branch entropy requires positive finite training temperature")
        for node in nodes:
            target = int(node["position"])
            candidates = node["token_ids"]
            if len(candidates) < 2 or len(candidates) != len(set(candidates)):
                raise ValueError("Branch metadata must contain at least two unique next tokens")
            if not 0 <= target < response_length:
                raise ValueError("Branch target is outside the response")
            position = offsets[row] + prompt_length + target - 1
            if position < offsets[row]:
                raise ValueError("Branch target has no preceding causal logit")
            if input_ids[position + 1] not in candidates:
                raise ValueError("Sampled response token is missing from its legal branch")
            if not response_masks[response_offsets[row] + target]:
                continue
            node_rows.append(row)
            node_positions.append(position)
            node_temperatures.append(temperature)
            widths.append(len(candidates))
            flat_positions.extend([position] * len(candidates))
            flat_candidates.extend(candidates)
            flat_temperatures.extend([temperature] * len(candidates))

    if not node_rows:
        return {
            f"branch_{key}": packed.new_zeros(len(rows), dtype=torch.float32)
            for key in ("entropy", "normalized", "legal_mass")
        }
    positions = torch.tensor(flat_positions, dtype=torch.long, device=packed.device)
    candidates = torch.tensor(flat_candidates, dtype=torch.long, device=packed.device)
    temperatures_tensor = torch.tensor(flat_temperatures, dtype=torch.float32, device=packed.device)
    # Exactly ONE gather from [all_tokens, vocabulary] per microbatch. Doing this
    # inside the node loop would scatter a full logits-sized gradient per node.
    selected = packed[positions, candidates].float() / temperatures_tensor
    results: dict[str, list[torch.Tensor]] = {key: [] for key in ("entropy", "normalized", "legal_mass")}
    for legal_logits, position, temperature, width in zip(
        selected.split(widths), node_positions, node_temperatures, widths, strict=True
    ):
        log_q = legal_logits.log_softmax(dim=-1)
        entropy = -(log_q.exp() * log_q).sum()
        results["entropy"].append(entropy)
        results["normalized"].append(entropy / math.log(width))
        with torch.no_grad():
            full_log_z = (packed[position].float() / temperature).logsumexp(dim=-1)
            results["legal_mass"].append((legal_logits.detach().logsumexp(dim=-1) - full_log_z).exp())
    row_ids = torch.tensor(node_rows, dtype=torch.long, device=packed.device)
    counts = torch.bincount(row_ids, minlength=len(rows)).clamp_min(1)
    # Empty/no-match decisions stay in the trajectory denominator with value zero.
    return {
        f"branch_{key}": packed.new_zeros(len(rows), dtype=torch.float32).index_add(0, row_ids, torch.stack(values))
        / counts
        for key, values in results.items()
    }


@EngineRegistry.register(
    model_type="alfworld_branch_language_model",
    backend=["fsdp", "fsdp2"],
    device="cuda",
)
class BranchEntropyEngine(FSDPEngineWithLMHead):
    def __init__(self, model_config: HFModelConfig, *args: Any, **kwargs: Any) -> None:
        # The separate registry key selects this extension only for its actor.
        # The underlying model is still a native language model, including checkpoints.
        model_config.model_type = "language_model"
        super().__init__(model_config, *args, **kwargs)

    def _build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        config = self.optimizer_config
        initial_ratio = getattr(config, "init_lr_ratio", 0.0)
        if not initial_ratio:
            return super()._build_lr_scheduler(optimizer)
        from verl.utils.torch_functional import get_cosine_schedule_with_warmup

        if config.lr_scheduler_type != "cosine":
            raise ValueError("Nonzero init_lr_ratio requires the cosine scheduler")
        warmup = config.lr_warmup_steps
        if warmup is None or warmup <= 0:
            warmup = int(config.lr_warmup_steps_ratio * config.total_training_steps)
        if not 0 < warmup < config.total_training_steps:
            raise ValueError("Nonzero initial warmup requires 0 < warmup steps < total steps")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=config.total_training_steps,
            min_lr_ratio=config.min_lr_ratio,
            num_cycles=config.num_cycles,
            init_lr_ratio=initial_ratio,
            zero_indexed_step=config.zero_indexed_step,
        )
        if self.rank == 0:
            print(
                f"ALFWorld cosine LR: initial_lr={scheduler.get_last_lr()[0]:.12g}, "
                f"peak_lr={config.lr:.12g}, warmup_steps={warmup}, "
                f"total_steps={config.total_training_steps}",
                flush=True,
            )
        return scheduler

    def prepare_model_outputs(
        self,
        output: Any,
        output_args: dict[str, Any],
        micro_batch: TensorDict,
        logits_processor_func: Callable | None,
    ) -> dict[str, torch.Tensor]:
        branch_output = {}
        if torch.is_grad_enabled() and "branch_entropy_weight" in micro_batch:
            if self.ulysses_sequence_parallel_size != 1 or not tu.get(micro_batch, "use_remove_padding", True):
                raise ValueError("Branch entropy supports packed actor logits with SP=1 only")
            if tu.get(micro_batch, "use_fused_kernels", False):
                raise ValueError("Branch entropy requires materialized logits")
            branch_output = branch_statistics(output.logits, micro_batch)
        result = super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)
        result.update(branch_output)
        return result


class BranchEntropyTrainingWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig) -> None:
        super().__init__(replace(config, model_type="alfworld_branch_language_model"))


def branch_ppo_loss(
    *,
    base_loss: Callable,
    coefficient: float,
    model_output: dict[str, torch.Tensor],
    data: TensorDict,
    dp_group: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Preserve native PPO/KL, adding only a trajectory-weighted entropy term."""
    loss, metrics = base_loss(model_output=model_output, data=data, dp_group=dp_group)
    weights = data["branch_entropy_weight"].to(model_output["branch_entropy"])
    factor = float(data["dp_size"]) / float(data["branch_entropy_normalizer"])
    branch_loss = (weights * model_output["branch_entropy"]).sum() * factor
    loss = loss - coefficient * branch_loss
    for source, name in (
        ("entropy", "branch_entropy"),
        ("normalized", "branch_entropy_normalized"),
        ("legal_mass", "branch_legal_mass"),
    ):
        value = (weights * model_output[f"branch_{source}"]).sum() * factor
        metrics[name] = Metric(value=value.detach(), aggregation=AggregationType.SUM)
    metrics["branch_entropy_loss"] = Metric(
        value=(-coefficient * branch_loss).detach(), aggregation=AggregationType.SUM
    )
    metrics["branch_entropy_coefficient"] = coefficient
    return loss, metrics


class BranchEntropyActorWorker(ALFWorldActorRolloutRefWorker):
    actor_worker_cls = BranchEntropyTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        super().init_model()
        if self.actor is not None:
            self.actor.set_loss_fn(
                partial(
                    branch_ppo_loss,
                    base_loss=self.loss_fn,
                    coefficient=float(self.config.branch_entropy.coefficient),
                )
            )


class BranchEntropyTrainerMixin:
    """Attach complete-rollout weights before V1 shuffles/splits rows across GPUs."""

    def _update_actor(self, batch: Any, metrics: dict[str, Any]) -> Any:
        if not self.config.actor_rollout_ref.get("branch_entropy", {}).get("enabled", False):
            return super()._update_actor(batch, metrics)
        import transfer_queue as tq

        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        rows = list(data["extra_fields"])
        if any(FIELD not in row for row in rows):
            raise ValueError("Missing rollout branch metadata; check the ALFWorld agent loop configuration")
        mini_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        weights, normalizer = trajectory_weights(list(batch.keys), list(batch.tags), mini_size)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"branch_entropy_weight": torch.tensor(weights)},
                batch_size=[len(weights)],
            ),
        )
        # Existing KV metadata enumerates fields persisted before this write.
        # Keep its tags/extra_info, while exposing the new tensor to the bridge.
        if getattr(batch, "fields", None) is not None:
            batch.fields = list(dict.fromkeys([*batch.fields, "branch_entropy_weight"]))
        batch.extra_info["branch_entropy_normalizer"] = normalizer
        real = [row[FIELD] for row, weight in zip(rows, weights, strict=True) if weight > 0]
        counts = Counter(row["status"] for row in real)
        for status in (
            "matched",
            "no_action",
            "not_admissible",
            "no_thinking",
            "no_mention",
            "only_listing_mentions",
            "token_mismatch",
            "unstable_prefix",
            "ambiguous_boundary",
        ):
            metrics[f"alfworld_branch/{status}_count"] = counts[status]
        metrics["alfworld_branch/listing_skipped_row_count"] = sum(
            row.get("skipped_listing_count", 0) > 0 for row in real
        )
        metrics["alfworld_branch/listing_skipped_mention_count"] = sum(
            row.get("skipped_listing_count", 0) for row in real
        )
        metrics["alfworld_branch/later_mention_selected_count"] = sum(
            row.get("selected_mention_index", 0) > 1 for row in real
        )
        metrics["alfworld_branch/match_fraction"] = counts["matched"] / len(real)
        metrics["alfworld_branch/eligible_fraction"] = sum(bool(row["nodes"]) for row in real) / len(real)
        metrics["alfworld_branch/node_count"] = sum(len(row["nodes"]) for row in real)
        return super()._update_actor(batch, metrics)
