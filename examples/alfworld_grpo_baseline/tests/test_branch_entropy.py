"""CPU-only integration tests for alignment, native queue rows and actor gradients."""

import asyncio
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from alfworld_baseline.branch_entropy import (
    FIELD,
    locate_action_branches,
    trajectory_weights,
    validate_branch_entropy_config,
)
from alfworld_baseline.branch_entropy_engine import (
    BranchEntropyEngine,
    BranchEntropyTrainerMixin,
    branch_ppo_loss,
    branch_statistics,
)
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.utils.losses import ppo_loss

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/LXJ/Python_Projects/Models/Qwen3.5-2B")


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer.json").is_file():
        pytest.skip("Local Qwen3.5 tokenizer not installed")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def xml(action):
    return (
        f"</think>\n<tool_call><function=alfworld_action><parameter=action>{action}</parameter></function></tool_call>"
    )


def align(tokenizer, thought, action, actions):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prompt = encode("<|im_start|>assistant\n<think>\n")
    response = encode(thought + xml(action))
    return (
        locate_action_branches(tokenizer, prompt, response, action, actions),
        prompt,
        response,
    )


def test_real_tokenizer_first_mention_shared_prefixes_numbers_and_dedup(tokenizer):
    metadata, _, response = align(
        tokenizer,
        "I will go to cabinet 1. Then go to cabinet 1.\n",
        "go to cabinet 1",
        [
            "look",
            "go to cabinet 1",
            "go to cabinet 1",
            "go to cabinet 2",
            "go to drawer 1",
        ],
    )
    assert metadata["status"] == "matched"
    assert metadata["candidate_count"] == 4
    branches = [{tokenizer.decode([t]) for t in n["token_ids"]} for n in metadata["nodes"]]
    assert branches == [{" go", " look"}, {" cabinet", " drawer"}, {"1", "2"}]
    assert [tokenizer.decode([response[n["position"]]]) for n in metadata["nodes"]] == [
        " go",
        " cabinet",
        "1",
    ]
    assert all(n["position"] < response.index(tokenizer.convert_tokens_to_ids("</think>")) for n in metadata["nodes"])


@pytest.mark.parametrize(
    "thought,action,actions,status",
    [
        ("I should search.\n", "look", ["look", "inventory"], "no_mention"),
        ("Looking at the room.\n", "look", ["look"], "no_mention"),
        ("I will look.\n", "look", ["inventory"], "not_admissible"),
        ("I will look.\n", None, ["look"], "no_action"),
        ("I will look.\n", "look", ["look"], "matched"),
    ],
)
def test_missing_invalid_and_single_candidate(tokenizer, thought, action, actions, status):
    metadata, _, _ = align(tokenizer, thought, action, actions)
    assert metadata["status"] == status
    assert metadata["nodes"] == []


def test_command_termination_is_a_branch_and_context_space_is_preserved(tokenizer):
    metadata, _, response = align(tokenizer, "I choose look.\n", "look", ["look", "look around"])
    assert metadata["status"] == "matched"
    assert len(metadata["nodes"]) == 1
    node = metadata["nodes"][0]
    assert tokenizer.decode([response[node["position"]]]) == "."
    assert {tokenizer.decode([t]) for t in node["token_ids"]} == {".", " around"}


def test_noncanonical_generated_ids_are_not_replaced(tokenizer):
    _, prompt, response = align(tokenizer, "I will look.\n", "look", ["look", "inventory"])
    target = response.index(tokenizer.encode(" look", add_special_tokens=False)[0])
    split = tokenizer.encode(" ", add_special_tokens=False) + tokenizer.encode("look", add_special_tokens=False)
    changed = response[:target] + split + response[target + 1 :]
    assert tokenizer.decode(changed) == tokenizer.decode(response)
    snapshot = list(changed)
    metadata = locate_action_branches(tokenizer, prompt, changed, "look", ["look", "inventory"])
    assert metadata["status"] == "token_mismatch"
    assert changed == snapshot


def test_ambiguous_termination_does_not_silently_drop_candidate(tokenizer):
    metadata, _, _ = align(tokenizer, "I will look around the room.\n", "look", ["look", "look around"])
    assert metadata["status"] == "ambiguous_boundary"
    assert metadata["nodes"] == []


def test_engine_registration_is_isolated_from_reference_and_original_actor(monkeypatch):
    from alfworld_baseline.branch_entropy_engine import BranchEntropyActorWorker, BranchEntropyTrainingWorker
    from alfworld_baseline.workers import ALFWorldActorRolloutRefWorker
    from verl.workers.engine.base import EngineRegistry
    from verl.workers.engine_workers import TrainingWorker

    monkeypatch.setattr("verl.workers.engine.base.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl.workers.engine.base.get_vendor", lambda: "nvidia")
    assert EngineRegistry.get_engine_cls("language_model", "fsdp") is transformer_impl.FSDPEngineWithLMHead
    assert EngineRegistry.get_engine_cls("alfworld_branch_language_model", "fsdp") is BranchEntropyEngine
    assert BranchEntropyActorWorker.actor_worker_cls is BranchEntropyTrainingWorker
    assert BranchEntropyActorWorker.ref_worker_cls is TrainingWorker
    assert ALFWorldActorRolloutRefWorker.actor_worker_cls is TrainingWorker


def toy_batch():
    # Different prompt/response lengths exercise packed row offsets and causal shift.
    ids = torch.nested.nested_tensor([torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 1, 3, 2])], layout=torch.jagged)
    masks = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    rows = [
        {FIELD: {"status": "matched", "nodes": [{"position": 0, "token_ids": [2, 3]}]}},
        {FIELD: {"status": "matched", "nodes": [{"position": 1, "token_ids": [1, 3]}]}},
    ]
    data = TensorDict(
        {
            "input_ids": ids,
            "response_mask": masks,
            "loss_mask": masks,
            "prompts": torch.nested.nested_tensor([torch.tensor([0, 1]), torch.tensor([1, 0])], layout=torch.jagged),
            "responses": torch.nested.nested_tensor(
                [torch.tensor([2, 3]), torch.tensor([1, 3, 2])], layout=torch.jagged
            ),
            "extra_fields": NonTensorStack(*rows),
            "branch_entropy_weight": torch.tensor([1.0, 1.0]),
        },
        [2],
    )
    tu.assign_non_tensor(
        data,
        temperature=1.0,
        dp_size=1,
        branch_entropy_normalizer=2.0,
        global_batch_size=2,
        batch_num_tokens=5,
        use_remove_padding=True,
        calculate_entropy=False,
        use_fused_kernels=False,
    )
    return data


def test_entropy_gradient_targets_causal_legal_tokens_and_raises_diversity():
    data = toy_batch()
    logits = torch.zeros(1, 9, 5, requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 2] = 2.0
        logits[0, 6, 3] = 2.0
    stats = branch_statistics(logits, data)
    # All branch candidates must share one indexing operation on the large logits.
    visited, pending = set(), [stats["branch_entropy"].grad_fn]
    while pending:
        node = pending.pop()
        if node is not None and node not in visited:
            visited.add(node)
            pending.extend(child for child, _ in node.next_functions)
    assert sum(type(node).__name__.startswith("IndexBackward") for node in visited) == 1
    objective = -stats["branch_entropy"].mean()
    objective.backward()
    nonzero = logits.grad.nonzero().tolist()
    assert nonzero == [[0, 1, 2], [0, 1, 3], [0, 6, 1], [0, 6, 3]]
    assert logits.grad[0, 1, 2] > 0  # gradient descent reduces overconfident selected token
    assert logits.grad[0, 1, 3] < 0
    updated = logits.detach() - 0.1 * logits.grad
    assert branch_statistics(updated, data)["branch_entropy"].mean() > -objective.detach()
    assert (stats["branch_normalized"] <= 1).all()
    assert ((stats["branch_legal_mass"] > 0) & (stats["branch_legal_mass"] < 1)).all()


def test_temperature_and_nonlegal_logits_affect_distinct_statistics():
    data = toy_batch()
    logits = torch.randn(1, 9, 5)
    before = branch_statistics(logits, data)
    logits[0, [1, 6], 4] += 10
    after = branch_statistics(logits, data)
    torch.testing.assert_close(before["branch_entropy"], after["branch_entropy"])
    assert (after["branch_legal_mass"] < before["branch_legal_mass"]).all()
    tu.assign_non_tensor(data, temperature=2.0)
    assert (branch_statistics(logits, data)["branch_entropy"] >= after["branch_entropy"]).all()


def test_padding_and_missing_alignments_have_zero_entropy():
    data = toy_batch()
    # Mimic native synthetic padding: copied metadata no longer matches replacement IDs.
    data["branch_entropy_weight"][0] = 0
    data["extra_fields"] = NonTensorStack(
        {FIELD: {"nodes": [{"position": 999, "token_ids": [0, 1]}]}},
        {FIELD: {"status": "no_mention", "nodes": []}},
    )
    values = branch_statistics(torch.randn(1, 9, 5), data)
    assert all(torch.equal(value, torch.zeros(2)) for value in values.values())


def test_row_reordering_and_microbatch_split_preserve_entropy():
    data = toy_batch()
    logits = torch.randn(1, 9, 5, requires_grad=True)
    full = branch_statistics(logits, data)["branch_entropy"]
    # Split/reorder via real TensorDict indexing, as native token-budget batching does.
    split = [
        branch_statistics(logits[:, 4:, :], tu.index_select_tensor_dict(data, [1]))["branch_entropy"],
        branch_statistics(logits[:, :4, :], tu.index_select_tensor_dict(data, [0]))["branch_entropy"],
    ]
    torch.testing.assert_close(torch.cat(split).flip(0), full)
    (grad_full,) = torch.autograd.grad(full.sum(), logits, retain_graph=True)
    (grad_split,) = torch.autograd.grad(sum(v.sum() for v in split), logits)
    torch.testing.assert_close(grad_full, grad_split)


def test_trajectory_and_dp_normalization_not_length_weighted():
    keys = ["task_a_session1_0", "task_b_session2_0", "task_b_session2_1", "pad_0_0"]
    weights, denominator = trajectory_weights(keys, [{}, {}, {}, {"is_padding": True}], 2)
    assert weights == [1.0, 0.5, 0.5, 0.0]
    assert denominator == 1.0
    entropy = torch.tensor([0.2, 0.4, 0.8, 9.0], requires_grad=True)
    # Two minibatches, each split over two ranks; DDP averages gradients over ranks.
    per_micro = [entropy[i] * weights[i] * 2 / denominator for i in range(4)]
    aggregated = sum(per_micro) / 2 / 2
    expected = (0.2 + (0.4 + 0.8) / 2) / 2
    assert aggregated.item() == pytest.approx(expected)
    aggregated.backward()
    torch.testing.assert_close(entropy.grad, torch.tensor([0.5, 0.25, 0.25, 0.0]))


@pytest.mark.parametrize("entropy_coeff", [0.0, 0.001])
def test_native_logprobs_and_ppo_kl_unchanged_plus_branch_gradient(monkeypatch, entropy_coeff):
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    data = toy_batch()
    tu.assign_non_tensor(data, calculate_entropy=entropy_coeff != 0)
    logits = torch.randn(1, 9, 5, requires_grad=True)
    engine = object.__new__(BranchEntropyEngine)
    engine.ulysses_sequence_parallel_size = 1
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = lambda x: -(x.log_softmax(-1).exp() * x.log_softmax(-1)).sum(-1)
    engine._gather_and_unpad_packed = lambda value, pad_size: value
    args = {
        "temperature_is_one": True,
        "temperature_rmpad": torch.ones(9),
        "input_ids_rmpad_rolled": data["input_ids"].values().roll(-1),
        "pad_size": 0,
    }
    native = transformer_impl.FSDPEngineWithLMHead.prepare_model_outputs(
        engine, SimpleNamespace(logits=logits), args, data, None
    )
    combined = engine.prepare_model_outputs(SimpleNamespace(logits=logits), args, data, None)
    torch.testing.assert_close(native["log_probs"].values(), combined["log_probs"].values())
    assert ("entropy" in combined) == (entropy_coeff != 0)
    if entropy_coeff:
        torch.testing.assert_close(native["entropy"].values(), combined["entropy"].values())
    data["old_log_probs"] = torch.nested.nested_tensor(
        [torch.full((2,), -1.0), torch.full((3,), -1.0)], layout=torch.jagged
    )
    data["ref_log_prob"] = data["old_log_probs"].clone()
    data["advantages"] = torch.nested.nested_tensor([torch.ones(2), -torch.ones(3)], layout=torch.jagged)
    from verl.workers.config.actor import FSDPActorConfig

    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        use_kl_loss=True,
        kl_loss_coef=0.01,
        entropy_coeff=entropy_coeff,
        loss_agg_mode="token-mean",
    )
    base = partial(ppo_loss, config=config)
    original, _ = base(model_output=native, data=data)
    actual, _ = branch_ppo_loss(base_loss=base, coefficient=0.1, model_output=combined, data=data)
    torch.testing.assert_close(actual, original - 0.1 * combined["branch_entropy"].mean())
    expected = original - 0.1 * combined["branch_entropy"].mean()
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()


def test_native_inplace_ce_backward_cannot_corrupt_branch_gradient(monkeypatch):
    class DestructiveCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels):
            ctx.save_for_backward(logits, labels)
            return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)

        @staticmethod
        def backward(ctx, grad):
            logits, labels = ctx.saved_tensors
            result = -logits.softmax(-1)
            result.scatter_add_(1, labels[:, None], torch.ones_like(labels[:, None], dtype=result.dtype))
            result *= grad[:, None]
            logits.copy_(result)  # emulate Flash CE's storage-reusing backward
            return logits, None

    data = toy_batch()
    original = torch.randn(1, 9, 5)
    gradients = []
    for destructive in (False, True):
        leaf = original.clone().requires_grad_()
        logits = leaf * 1  # native model outputs are non-leaves
        stats = branch_statistics(logits, data)
        labels = data["input_ids"].values().roll(-1)
        if destructive:
            lp = DestructiveCE.apply(logits.squeeze(0), labels)
        else:
            lp = logits.squeeze(0).log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        (-lp.mean() - 0.1 * stats["branch_entropy"].mean()).backward()
        gradients.append(leaf.grad)
    torch.testing.assert_close(*gradients)


def config(profile="action_branch_entropy"):
    with initialize_config_dir(config_dir=str(ROOT / "config/alfworld/qwen35_2b" / profile), version_base=None):
        return compose(config_name="alfworld_config_2gpu")


def test_config_is_opt_in_and_preserves_rewards_and_sampling():
    new, old = config(), config("v1")
    validate_branch_entropy_config(new)
    validate_branch_entropy_config(old)
    assert "branch_entropy" not in old.actor_rollout_ref
    assert new.reward == old.reward
    assert new.actor_rollout_ref.actor.optim == old.actor_rollout_ref.actor.optim
    for field in ("temperature", "top_p", "top_k", "n", "val_kwargs"):
        assert new.actor_rollout_ref.rollout[field] == old.actor_rollout_ref.rollout[field]
    assert new.trainer.resume_mode == "disable"
    assert new.trainer.default_local_dir != old.trainer.default_local_dir
    assert new.actor_rollout_ref.actor.entropy_coeff == old.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert new.actor_rollout_ref.actor.calculate_entropy
    assert new.actor_rollout_ref.branch_entropy.coefficient == 0.001
    assert Path(new.actor_rollout_ref.model.path).is_dir()


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor_rollout_ref.model.use_fused_kernels", True),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.actor.calculate_entropy", False),
        ("actor_rollout_ref.branch_entropy.coefficient", float("nan")),
        ("data.apply_chat_template_kwargs.enable_thinking", False),
    ],
)
def test_unsupported_config_fails_before_ray(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        validate_branch_entropy_config(cfg)


@pytest.mark.parametrize("selection", ["first", "first_non_listing"])
def test_loop_snapshots_state_and_queue_retains_metadata(tokenizer, monkeypatch, selection):
    from test_native_step_pipeline import episode
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.trainer.ppo.v1 import AgentLoopWorkerTQ
    import transfer_queue as tq

    loop, _, _, _ = episode()
    loop._branch_entropy_enabled = loop._thinking_enabled = True
    loop._branch_mention_selection = selection
    loop.tokenizer = tokenizer
    loop.prompt_length = 2048
    loop.response_length = 768
    _, prompt, response = align(
        tokenizer, "Available actions:\n- look\n- inventory\n\nI will look.\n", "look", ["look", "inventory"]
    )
    loop.apply_chat_template = AsyncMock(return_value=prompt)
    loop.server_manager.generate = AsyncMock(
        return_value=SimpleNamespace(
            token_ids=response,
            log_probs=[-0.1] * len(response),
            num_preempted=0,
            routed_experts=None,
        )
    )
    loop.tools["alfworld_action"].get_state = lambda _: (
        "room\nYour task is to: find apple.",
        ("look", "inventory"),
    )
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert outputs[0].extra_fields[FIELD]["selected_mention_index"] == (2 if selection == "first_non_listing" else 1)
    assert len(outputs[0].extra_fields[FIELD]["nodes"]) == 1
    assert outputs[1].extra_fields[FIELD]["nodes"] == []  # next environment only allows 'look'
    assert all(output.response_ids == response and output.reward_score == 10 for output in outputs)
    captured = {}

    async def put(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tq, "async_kv_batch_put", put)
    worker = SimpleNamespace(
        reward_loop_worker_handles=[],
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(ids.shape[-1]).unsqueeze(0),
    )
    worker._compute_score = lambda outputs, kwargs: AgentLoopWorker._compute_score(worker, outputs, kwargs)
    method = AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess
    asyncio.run(
        method(
            worker,
            outputs,
            False,
            uid="task",
            session_id="session",
            global_steps=0,
            partition_id="train",
        )
    )
    rows = list(captured["fields"]["extra_fields"])
    assert [row[FIELD] for row in rows] == [out.extra_fields[FIELD] for out in outputs]


def test_trainer_attaches_weights_preserving_transfer_metadata(monkeypatch):
    import transfer_queue as tq
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    keys = ["task_s1_0", "task_s2_0", "task_s2_1", "padding_0_0"]
    batch = SimpleNamespace(
        keys=keys,
        tags=[{}, {}, {}, {"is_padding": True}],
        partition_id="train",
        extra_info={"sentinel": 7},
        fields=["input_ids", "extra_fields", "response_mask"],
    )
    metadata = {FIELD: {"status": "matched", "nodes": [{"position": 0, "token_ids": [1, 2]}],
                        "skipped_listing_count": 1, "selected_mention_index": 2}}
    monkeypatch.setattr(
        tq,
        "kv_batch_get",
        lambda **kwargs: list_of_dict_to_tensordict([{"extra_fields": metadata}] * 4),
    )
    stored = {}
    monkeypatch.setattr(tq, "kv_batch_put", lambda **kwargs: stored.update(kwargs))

    class Base:
        def _update_actor(self, batch, metrics):
            return batch

    class Trainer(BranchEntropyTrainerMixin, Base):
        pass

    trainer = Trainer()
    trainer.config = config()
    trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    trainer.config.actor_rollout_ref.rollout.n = 2
    metrics = {}
    result = trainer._update_actor(batch, metrics)
    assert result is batch and result.extra_info["sentinel"] == 7
    assert result.fields == ["input_ids", "extra_fields", "response_mask", "branch_entropy_weight"]
    assert result.extra_info["branch_entropy_normalizer"] == 1
    assert stored["fields"]["branch_entropy_weight"].tolist() == [1.0, 0.5, 0.5, 0.0]
    assert metrics["alfworld_branch/matched_count"] == 3
    assert metrics["alfworld_branch/listing_skipped_row_count"] == 3
    assert metrics["alfworld_branch/listing_skipped_mention_count"] == 3
    assert metrics["alfworld_branch/later_mention_selected_count"] == 3
    assert metrics["alfworld_branch/only_listing_mentions_count"] == 0


def test_non_listing_alignment_preserves_rollout_and_selects_later_tokens(tokenizer):
    thought = "Available actions:\n- 'go to cabinet 1'\n- 'look'\n\nI choose go to cabinet 1.\n"
    actions = ['go to cabinet 1', 'look']
    old, prompt, response = align(tokenizer, thought, actions[0], actions)
    snapshot = list(response)
    new = locate_action_branches(
        tokenizer, prompt, response, actions[0], actions, mention_selection='first_non_listing',
    )
    assert old['status'] == new['status'] == 'matched'
    assert new['selected_mention_index'] == 2
    assert new['skipped_listing_count'] == 1
    assert min(n['position'] for n in new['nodes']) > max(n['position'] for n in old['nodes'])
    assert response == snapshot
    assert all(n['position'] < response.index(tokenizer.convert_tokens_to_ids('</think>')) for n in new['nodes'])


def test_all_lists_never_align_to_xml_action(tokenizer):
    thought = "Available actions:\n- 'go to cabinet 1'\n- 'look'\n"
    actions = ['go to cabinet 1', 'look']
    _, prompt, response = align(tokenizer, thought, actions[0], actions)
    metadata = locate_action_branches(
        tokenizer, prompt, response, actions[0], actions, mention_selection='first_non_listing',
    )
    assert metadata['status'] == 'only_listing_mentions'
    assert metadata['nodes'] == []
    assert metadata['skipped_listing_count'] == 1
