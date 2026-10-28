"""Verify eligibility rewards using real token alignment and the native episode loop."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from alfworld_baseline.branch_entropy import FIELD, validate_branch_entropy_config
from alfworld_baseline.metrics import compute_alfworld_rollout_metrics
from alfworld_baseline.reward import compute_score
from test_decision_budget import CALL
from test_native_step_pipeline import episode


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("/home/LXJ/Python_Projects/Models/Qwen3.5-2B", local_files_only=True)


@pytest.mark.parametrize("text,actions,status,expected", [
    ("I choose look.</think>" + CALL, ["look", "inventory"], "matched", .05),
    ("I choose look.</think>" + CALL, ["look", "look around", "inventory"], "matched", .05),
    ("I choose Look.</think>" + CALL, ["look", "inventory"], "matched", .05),
    ("look, inventory are available. I choose look.</think>" + CALL,
     ["look", "inventory"], "matched", .05),
    ("Available actions: look, inventory.</think>" + CALL,
     ["look", "inventory"], "only_listing_mentions", 0),
    ("I choose inventory.</think>" + CALL, ["look", "inventory"], "no_mention", 0),
    ("I choose look.</think>" + CALL, ["look"], "matched", 0),
    ("I choose look.</think>" + CALL, ["inventory"], "not_admissible", 0),
    ("I choose look.</think>No XML", ["look", "inventory"], "no_action", 0),
])
def test_real_alignment_bonus_once_per_decision(tokenizer, text, actions, status, expected):
    loop, calls, _, _ = episode()
    loop.tokenizer = tokenizer
    loop._thinking_enabled = loop._branch_entropy_enabled = True
    loop._branch_mention_selection = "first_non_listing"
    loop._branch_entropy_eligible_reward = .05
    prompt = tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    response = tokenizer.encode(text, add_special_tokens=False)
    loop.apply_chat_template = AsyncMock(return_value=prompt)
    loop.server_manager.generate = AsyncMock(return_value=SimpleNamespace(
        token_ids=response, log_probs=[-.1] * len(response), num_preempted=0, routed_experts=None,
    ))
    # The first decision sees `actions`; subsequent states only allow 'look'.
    loop.tools["alfworld_action"].get_state = lambda _: (
        "room 0\nYour task is to: find apple.", tuple(actions)
    )
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert outputs[0].extra_fields[FIELD]["status"] == status
    assert [out.extra_fields["alfworld_branch_entropy_eligible_step_reward"] for out in outputs] == [expected, 0, 0]
    assert all(out.response_ids == response for out in outputs)
    final = outputs[-1].extra_fields
    assert final["alfworld_branch_entropy_eligible_reward"] == expected
    assert final["alfworld_branch_entropy_eligible_count"] == int(expected > 0)
    # Final episode score includes the bonus once, not once per serialized row.
    base = compute_score("alfworld", extra_info={**final, "alfworld_branch_entropy_eligible_reward": 0})
    assert all(out.reward_score == pytest.approx(base + expected) for out in outputs)
    assert final["reward_extra_info"]["base_reward"] == pytest.approx(base)
    metrics = compute_alfworld_rollout_metrics(SimpleNamespace(
        non_tensor_batch={key: [value] for key, value in final.items()}
    ))
    assert metrics["alfworld_reward/branch_entropy_eligible_total"] == expected
    assert metrics["alfworld_reward/base_reward_mean"] == pytest.approx(base)


def test_bonus_preserves_failure_penalties_and_other_bonus():
    assert compute_score("alfworld", extra_info={
        "tool_rewards": [0, -.2, -.2], "alfworld_terminal_reason": "decision_limit",
        "alfworld_repeated_action_penalty_count": 2,
        "alfworld_thinking_action_reward": .1,
        "alfworld_branch_entropy_eligible_reward": .15,
    }) == pytest.approx(-.35)


def test_multiple_eligible_decisions_sum_into_one_episode_reward(tokenizer):
    loop, _, _, _ = episode()
    loop.tokenizer = tokenizer
    loop._thinking_enabled = loop._branch_entropy_enabled = True
    loop._branch_entropy_eligible_reward = .05
    loop._thinking_action_once_reward = .1
    response = tokenizer.encode("I choose look.</think>" + CALL, add_special_tokens=False)
    loop.apply_chat_template = AsyncMock(return_value=tokenizer.encode(
        "<|im_start|>assistant\n<think>\n", add_special_tokens=False
    ))
    loop.server_manager.generate = AsyncMock(return_value=SimpleNamespace(
        token_ids=response, log_probs=[-.1] * len(response), num_preempted=0, routed_experts=None,
    ))
    tool = loop.tools["alfworld_action"]
    tool.get_state = lambda _: ("room\nYour task is to: find apple.", ("look", "inventory"))
    execute = tool.execute

    async def execute_with_candidates(*args, **kwargs):
        result, reward, metrics = await execute(*args, **kwargs)
        metrics["admissible_commands"] = ["look", "inventory"]
        return result, reward, metrics

    tool.execute = execute_with_candidates
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert [out.extra_fields["alfworld_branch_entropy_eligible_step_reward"] for out in outputs] == [.05] * 3
    assert all(out.reward_score == pytest.approx(10.45) for out in outputs)
    final = outputs[-1].extra_fields
    assert final["alfworld_branch_entropy_eligible_count"] == 3
    assert final["reward_extra_info"]["branch_entropy_eligible_reward"] == pytest.approx(.15)
    assert final["reward_extra_info"]["thinking_action_reward"] == pytest.approx(.3)
    assert final["reward_extra_info"]["base_reward"] == pytest.approx(10)


def test_live_profile_and_invalid_eligibility_config():
    root = Path(__file__).resolve().parents[1] / "config/alfworld/qwen35_2b/action_entropy_1019"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        config = compose(config_name="alfworld_config_2gpu")
    validate_branch_entropy_config(config)
    assert config.variables.BRANCH_ENTROPY_ELIGIBLE_REWARD == .05
    assert config.actor_rollout_ref.branch_entropy.coefficient == .0007
    assert config.variables.THINKING_ACTION_REPEAT_REWARD == 0
    assert config.variables.get("THINKING_ACTION_ONCE_REWARD", 0) == 0
    for path, value in [
        ("actor_rollout_ref.branch_entropy.enabled", False),
        ("actor_rollout_ref.branch_entropy.coefficient", 0),
        ("variables.BRANCH_ENTROPY_ELIGIBLE_REWARD", -.05),
        ("variables.BRANCH_ENTROPY_ELIGIBLE_REWARD", float("nan")),
    ]:
        invalid = OmegaConf.create(OmegaConf.to_container(config))
        OmegaConf.update(invalid, path, value)
        with pytest.raises(ValueError, match="BRANCH_ENTROPY_ELIGIBLE_REWARD"):
            validate_branch_entropy_config(invalid)
