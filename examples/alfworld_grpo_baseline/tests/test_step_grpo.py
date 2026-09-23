"""Local rewards and cross-step GRPO through native serialization and PPO."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from tensordict import TensorDict

from alfworld_baseline.budget import validate_native_step_config
from alfworld_baseline.step_advantage import ADV_ESTIMATOR, compute_step_grpo_advantage
from alfworld_baseline.step_agent_loop import ALFWorldStepGRPOAgentLoop
from alfworld_baseline.step_reward import compute_score
from alfworld_baseline.step_workers import (
    StepGRPOWorkerMixin,
    StepGRPOAgentLoopManager,
    StepGRPOAgentLoopWorkerTQ,
)
from alfworld_baseline.validation_logging import ALFWorldValidationLoggingMixin
from test_native_step_pipeline import episode
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.protocol import DataProto
from verl.tools.schemas import ToolResponse
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

ROOT = Path(__file__).resolve().parents[1]


def backend_config(profile="qwen35_2b", backend="gigpo_grpo"):
    with initialize_config_dir(
        config_dir=str(ROOT / "config/alfworld" / profile / "v1"), version_base=None
    ):
        overrides = ["+training_backend=gigpo_grpo"] if backend == "gigpo_grpo" else []
        return compose(config_name="alfworld_config_2gpu", overrides=overrides)


def xml(action):
    return f"</think><tool_call><function=alfworld_action><parameter=action>{action}</parameter></function></tool_call>"


def mixed_episode(*, success=True, legacy=False, texts=None):
    loop, _, released, _ = episode()
    if not legacy:
        loop.__class__ = ALFWorldStepGRPOAgentLoop
    loop._thinking_enabled = True
    outputs = texts or [
        xml("look"),
        "unclosed thought",
        xml("not admissible"),
        xml("look"),
        xml("inventory"),
        xml("look"),
    ]
    loop._environment_budget = type(loop._environment_budget)(len(outputs), 768)
    loop.response_length = 768
    emitted = iter(outputs)
    calls = []

    async def generate(**kwargs):
        ids = loop.tokenizer.encode(next(emitted))
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.1] * len(ids),
            num_preempted=0,
            routed_experts=None,
        )

    async def execute(instance, args, **kwargs):
        action = args["action"]
        if action not in {"look", "inventory"}:
            return (
                ToolResponse(text="rejected"),
                0.0,
                {
                    "action": action,
                    "error": "invalid_action",
                    "observation": "room",
                    "admissible_commands": ["look", "inventory"],
                },
            )
        calls.append(action)
        won = success and len(calls) == 4
        return (
            ToolResponse(text="room"),
            10.0 if won else 0.0,
            {
                "action": action,
                "won": won,
                "done": won,
                "observation": "room",
                "admissible_commands": ["look", "inventory"],
            },
        )

    loop.tools["alfworld_action"].execute = AsyncMock(side_effect=execute)
    loop.server_manager.generate = AsyncMock(side_effect=generate)
    rows = asyncio.run(loop.run({}, raw_prompt=[]))
    assert released == ["instance"]
    assert [loop.tokenizer.decode(row.response_ids) for row in rows] == outputs
    return rows


@pytest.mark.parametrize("success", [True, False])
def test_penalties_are_local_flat_and_retained_after_success(success):
    rows = mixed_episode(success=success)
    expected_kinds = [
        "none",
        "no_action",
        "invalid_action",
        "repeated_action",
        "none",
        "repeated_action",
    ]
    outcome = 10.0 if success else 0.0
    assert [
        r.extra_fields["alfworld_step_penalty_kind"] for r in rows
    ] == expected_kinds
    assert [r.reward_score for r in rows] == pytest.approx(
        [
            outcome,
            outcome - 0.2,
            outcome - 0.2,
            outcome - 0.2,
            outcome,
            outcome - 0.2,
        ]
    )
    assert [r.extra_fields["alfworld_step_penalty"] for r in rows] == [
        0,
        -0.2,
        -0.2,
        -0.2,
        0,
        -0.2,
    ]
    assert all(
        r.extra_fields["alfworld_episode_penalty_sum"] == pytest.approx(-0.8)
        for r in rows
    )
    assert all(r.extra_fields["alfworld_episode_reward"] == outcome for r in rows)
    final = rows[-1].extra_fields
    assert final["alfworld_repeated_action_penalty_count"] == 2
    assert final["alfworld_invalid_tool_call_penalty_count"] == 1
    assert final["alfworld_no_tool_call_penalty_count"] == 1
    assert final["alfworld_valid_tool_call_count"] == 4
    assert sum(r.extra_fields["alfworld_is_final_step"] for r in rows) == 1
    assert all(
        compute_score("alfworld", extra_info=r.extra_fields) == r.reward_score
        for r in rows
    )


@pytest.mark.parametrize("success,expected", [(True, 6.0), (False, -4.25)])
def test_legacy_backend_keeps_original_penalties_and_success_exemption(
    success, expected
):
    rows = mixed_episode(success=success, legacy=True)
    assert [r.reward_score for r in rows] == [expected] * 6
    assert all("alfworld_step_penalty_kind" not in r.extra_fields for r in rows)


def test_many_missing_actions_do_not_accumulate_into_step_score():
    rows = mixed_episode(success=False, texts=["no action"] * 50)
    assert [r.reward_score for r in rows] == [-0.2] * 50
    assert rows[-1].extra_fields["alfworld_no_tool_call_penalty_count"] == 50
    assert rows[-1].extra_fields["alfworld_episode_penalty_sum"] == pytest.approx(-10)


def test_step_reward_rejects_episode_only_metadata():
    with pytest.raises(ValueError, match="decision's own"):
        compute_score("alfworld", extra_info={"tool_rewards": [-2, 10]})


def test_step_advantages_golden_values_multiple_groups_padding_and_shuffling():
    # Group a: [0, -.2, 0, 0] => [.5, -1.5, .5, .5] (sample std=.1).
    # Group b has unequal trajectory lengths: [10] and [0, 0]. Its mean is
    # 10/3, not the episode-level mean of 5; no final-row deduplication is allowed.
    scores = torch.tensor([0.0, -0.2, 0.0, 0.0, 10.0, 0.0, 0.0, 777.0, 0.0, 0.0, -0.2])
    groups = np.array(
        ["task_a"] * 4
        + ["task_b"] * 3
        + ["task_a", "constant", "constant", "singleton"]
    )
    mask = torch.tensor([[1, 1]] * 7 + [[0, 0]] + [[1, 0]] * 3)
    rewards = torch.stack([scores, torch.zeros_like(scores)], dim=-1)
    expected = torch.tensor(
        [
            0.5,
            -1.5,
            0.5,
            0.5,
            2 / np.sqrt(3),
            -1 / np.sqrt(3),
            -1 / np.sqrt(3),
            0,
            0,
            0,
            -0.2,
        ],
        dtype=torch.float32,
    )
    order = torch.tensor([7, 2, 6, 9, 4, 0, 10, 3, 5, 1, 8])
    advantages, returns = compute_step_grpo_advantage(
        rewards[order], mask[order], groups[order]
    )
    assert torch.allclose(advantages, expected[order, None] * mask[order], atol=2e-5)
    assert torch.equal(advantages, returns)
    assert torch.isfinite(advantages).all()
    assert rewards[1, 0] == -0.2  # scoring must not mutate the input rewards


def test_mean_only_normalization_and_empty_batch():
    adv, _ = compute_step_grpo_advantage(
        torch.tensor([[0.0], [-0.2]]),
        torch.ones(2, 1),
        ["a", "a"],
        config={"norm_adv_by_std_in_grpo": False},
    )
    assert adv[:, 0].tolist() == pytest.approx([0.1, -0.1])
    adv, _ = compute_step_grpo_advantage(torch.zeros(0, 1), torch.zeros(0, 1), [])
    assert adv.shape == (0, 1)
    adv, _ = compute_step_grpo_advantage(
        torch.full((50, 1), -0.2), torch.ones(50, 1), ["a"] * 50
    )
    assert torch.count_nonzero(adv) == 0


def test_native_queue_dispatch_preserves_local_scores_through_cpu_ppo(monkeypatch):
    import transfer_queue as tq

    rows, keys, tags = [], [], []

    async def put(**kwargs):
        keys.extend(kwargs["keys"])
        tags.extend(kwargs["tags"])
        rows.extend(kwargs["fields"][i].to_dict() for i in range(len(kwargs["keys"])))

    monkeypatch.setattr(tq, "async_kv_batch_put", put)
    worker = SimpleNamespace(
        reward_loop_worker_handles=[],
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(
            ids.shape[-1]
        ).unsqueeze(0),
    )
    worker._compute_score = lambda outputs, kwargs: AgentLoopWorker._compute_score(
        worker, outputs, kwargs
    )
    postprocess = StepGRPOWorkerMixin._agent_loop_postprocess
    for session_id, success in enumerate((True, False)):
        asyncio.run(
            postprocess(
                worker,
                mixed_episode(success=success),
                False,
                uid="task_with_underscores",
                session_id=session_id,
                global_steps=3,
            )
        )
    assert len(rows) == 12
    assert [float(r["rm_scores"].sum()) for r in rows] == pytest.approx(
        [10, 9.8, 9.8, 9.8, 10, 9.8, 0, -0.2, -0.2, -0.2, 0, -0.2]
    )
    padding, tag = construct_minimal_padding_template(rows[0], tags[0], eos_token_id=0)
    # Even padding accidentally sharing a task uid must not affect its statistics.
    padding["uid"] = rows[0]["uid"]
    rows.append(padding)
    keys.append("pad_0_0")
    order = [12, 5, 7, 0, 11, 2, 9, 4, 6, 1, 8, 3, 10]
    rows, keys = [rows[i] for i in order], [keys[i] for i in order]
    masks = torch.nn.utils.rnn.pad_sequence(
        [r["response_mask"] for r in rows], batch_first=True
    )
    rewards = torch.nn.utils.rnn.pad_sequence(
        [r["rm_scores"] for r in rows], batch_first=True
    )
    batch = DataProto(
        batch=TensorDict(
            {"response_mask": masks, "token_level_rewards": rewards}, batch_size=[13]
        ),
        non_tensor_batch={"uid": np.array([r["uid"] for r in rows], dtype=object)},
    )
    config = OmegaConf.create({"norm_adv_by_std_in_grpo": True})
    result = compute_advantage_for_multi_trajectories(
        batch, keys, ADV_ESTIMATOR, config=config
    )
    scores = rewards.sum(-1)
    valid = masks.any(-1)
    expected = (scores - scores[valid].mean()) / (scores[valid].std() + 1e-6)
    for i in range(len(rows)):
        assert torch.allclose(
            result.batch["advantages"][i], expected[i] * masks[i], atol=1e-6
        )
    correct = keys.index("task_with_underscores_0_0")
    invalid = keys.index("task_with_underscores_0_2")
    assert (
        result.batch["advantages"][correct, 0] > result.batch["advantages"][invalid, 0]
    )
    parameter = torch.nn.Parameter(torch.zeros_like(rewards))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    policy_config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": None,
            "clip_ratio_high": None,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
        }
    )
    loss, _ = compute_policy_loss_vanilla(
        torch.zeros_like(parameter),
        parameter,
        result.batch["advantages"],
        masks,
        config=policy_config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter.grad[~masks.bool()].abs().sum() == 0
    assert parameter.grad[masks.bool()].abs().sum() > 0
    optimizer.step()
    assert parameter.detach().abs().sum() > 0


@pytest.mark.parametrize("profile", ["qwen35_2b", "qwen35_9b", "qwen25_1_5b"])
def test_config_selects_a_complete_backend_and_isolates_outputs(profile):
    current, legacy = backend_config(profile), backend_config(profile, "trajectory")
    validate_native_step_config(current)
    validate_native_step_config(legacy)
    assert current.algorithm.adv_estimator == ADV_ESTIMATOR
    assert legacy.algorithm.adv_estimator == "grpo"
    if profile == "qwen35_2b":
        assert current.trainer.experiment_name == "qwen3.5_2B_GiGPO后端"
        assert current.trainer.default_local_dir == str(ROOT / "outputs/qwen3.5_2B/GiGPO后端")
    else:
        assert "/gigpo_grpo/" in current.trainer.default_local_dir
    assert current.trainer.rollout_data_dir == f"{current.trainer.default_local_dir}/rollouts"
    assert current.trainer.validation_data_dir == f"{current.trainer.default_local_dir}/validation"
    assert current.ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR == (
        f"{current.trainer.default_local_dir}/swanlab"
    )
    assert current.trainer.default_local_dir != legacy.trainer.default_local_dir
    assert current.trainer.experiment_name != legacy.trainer.experiment_name
    assert (
        current.actor_rollout_ref.actor.entropy_coeff
        == legacy.actor_rollout_ref.actor.entropy_coeff
    )
    assert (
        current.actor_rollout_ref.actor.kl_loss_coef
        == legacy.actor_rollout_ref.actor.kl_loss_coef
    )
    loop_config = OmegaConf.load(
        current.actor_rollout_ref.rollout.agent.agent_loop_config_path
    )
    assert (
        loop_config[0]._target_
        == "alfworld_baseline.step_agent_loop.ALFWorldStepGRPOAgentLoop"
    )
    assert current.actor_rollout_ref.rollout.agent.agent_loop_manager_class == (
        "alfworld_baseline.step_workers.StepGRPOAgentLoopManager"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("algorithm.adv_estimator", "grpo"),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "alfworld_tool_agent"),
        ("reward.custom_reward_function.path", "alfworld_baseline/reward.py"),
        ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", None),
        (
            "actor_rollout_ref.rollout.agent.agent_loop_config_path",
            str(ROOT / "config/agent_loops.yaml"),
        ),
    ],
)
def test_mixed_backend_config_is_rejected(key, value):
    cfg = backend_config()
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_native_step_config(cfg)


@pytest.mark.parametrize("success", [True, False])
def test_validation_queue_uses_outcomes_and_keeps_full_trajectories(
    monkeypatch, success
):
    import transfer_queue as tq

    serialized = []

    async def put(**kwargs):
        assert kwargs["partition_id"] == "val"
        serialized.extend(
            kwargs["fields"][i].to_dict() for i in range(len(kwargs["keys"]))
        )

    monkeypatch.setattr(tq, "async_kv_batch_put", put)
    worker = SimpleNamespace(
        _compute_multi_modal_inputs=lambda *args: {},
        _compute_position_ids=lambda ids, mask, mm: torch.arange(
            ids.shape[-1]
        ).unsqueeze(0),
    )
    asyncio.run(
        StepGRPOWorkerMixin._agent_loop_postprocess(
            worker,
            mixed_episode(success=success),
            True,
            uid="task",
            session_id=0,
            global_steps=1,
        )
    )
    outcome = 10.0 if success else 0.0
    assert [float(r["rm_scores"].sum()) for r in serialized] == [outcome] * 6
    assert serialized[-1]["extra_fields"]["alfworld_step_penalty"] == -0.2
    assert serialized[-1]["extra_fields"]["reward_extra_info"]["score"] == outcome
    captured = {}

    class Parent:
        def _dump_generations(self, **kwargs):
            captured["dump"] = kwargs

        def _maybe_log_val_generations(self, **kwargs):
            captured["preview"] = kwargs

    class Trainer(ALFWorldValidationLoggingMixin, Parent):
        config = SimpleNamespace(trainer=SimpleNamespace(log_val_generations=1))

    trainer = Trainer()
    trainer._alfworld_full_validation = True
    info = {
        "episode_reward": [outcome, outcome],
        "reward": [outcome, outcome],
        "score": [outcome, outcome],
        "uid": ["task_0_0", "task_0_1"],
    }
    trainer._dump_generations(
        ["p0", "p1"], ["r0", "r1"], [None, None], [outcome, outcome], info, "val"
    )
    assert captured["dump"]["scores"] == [outcome, outcome]
    assert captured["preview"]["scores"] == [outcome]
    assert "Step 2" in captured["preview"]["outputs"][0]
    trainer._alfworld_full_validation = False
    trainer._dump_generations(
        ["p0", "p1"], ["r0", "r1"], [None, None], [10, 9.8], info, "train"
    )
    assert captured["dump"]["scores"] == [10, 9.8]


def test_existing_parquet_agent_name_is_routed_only_in_new_worker(monkeypatch):
    captured = {}

    class Parent:
        async def _run_agent_loop(self, *args, **kwargs):
            captured.update(kwargs)

    class Worker(StepGRPOWorkerMixin, Parent):
        pass

    asyncio.run(
        Worker()._run_agent_loop({}, {}, agent_name="alfworld_tool_agent", uid="a")
    )
    assert captured["agent_name"] == "alfworld_step_grpo_agent"
    assert captured["uid"] == "a"
    with pytest.raises(ValueError, match="Unexpected agent"):
        asyncio.run(Worker()._run_agent_loop({}, {}, agent_name="tool_agent"))
    monkeypatch.setattr(AgentLoopManagerTQ, "__init__", lambda self, *a, **kw: None)
    manager = StepGRPOAgentLoopManager()
    assert manager.agent_loop_workers_class is StepGRPOAgentLoopWorkerTQ
    assert issubclass(
        StepGRPOAgentLoopWorkerTQ.__ray_metadata__.modified_class, StepGRPOWorkerMixin
    )
