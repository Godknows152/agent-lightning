"""Check literal matching and the bonus through the real native episode loop."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from alfworld_baseline.metrics import compute_alfworld_rollout_metrics
from alfworld_baseline.reward import compute_score
from alfworld_baseline.thinking_action_reward import count_thinking_action_mentions
from test_decision_budget import CALL
from test_native_step_pipeline import episode


@pytest.mark.parametrize(
    "thinking,action,expected",
    [
        ("I will use `look`.</think>", "look", 1),
        ("<think>I will look, then look.</think>", "look", 2),
        ("Looking is useful.</think>", "look", 0),
        ("looking and outlook.</think>", "look", 0),
        ("go to cabinet 10</think>", "go to cabinet 1", 0),
        ("Use go to cabinet 1.</think>", "go to cabinet 1", 1),
        ("Use go to cabinet 1", "go to cabinet 1", 0),
        ("No action here.</think>", "look", 0),
        ("look</think>", None, 0),
        ("look</think>", "", 0),
    ],
)
def test_matches_only_complete_action_in_closed_thinking(thinking, action, expected):
    # The occurrence in final XML must never contribute to the count.
    assert count_thinking_action_mentions(thinking + CALL, action) == expected


@pytest.mark.parametrize("coefficient,expected_score", [(0.0, 10.0), (0.1, 10.1), (1.0, 11.0)])
def test_bonus_accumulates_once_and_preserves_native_step_data(coefficient, expected_score):
    loop, calls, released, _ = episode()
    loop._thinking_enabled = True
    loop._thinking_action_once_reward = coefficient
    texts = ["Use look.</think>" + CALL, "look or look?</think>" + CALL, "Inspect the room.</think>" + CALL]
    pending = iter(texts)

    async def generate(**kwargs):
        ids = loop.tokenizer.encode(next(pending))
        return SimpleNamespace(token_ids=ids, log_probs=[-.1] * len(ids),
                               num_preempted=0, routed_experts=None)

    loop.server_manager.generate = AsyncMock(side_effect=generate)
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert calls == ["look"] * 3
    assert released == ["instance"]
    assert [out.response_ids for out in outputs] == [loop.tokenizer.encode(text) for text in texts]
    assert all(out.reward_score == pytest.approx(expected_score) for out in outputs)
    assert all(out.extra_fields["tool_rewards"] == [0.0, 0.0, 10.0] for out in outputs)
    if coefficient:
        assert [out.extra_fields["alfworld_thinking_action_step_reward"] for out in outputs] == [coefficient, 0, 0]
        assert [out.extra_fields["alfworld_thinking_action_mention_count"] for out in outputs] == [1, 2, 0]
        final = outputs[-1].extra_fields
        assert final["alfworld_thinking_action_once_count"] == 1
        assert final["reward_extra_info"] == {
            "score": expected_score, "base_reward": 10.0, "thinking_action_reward": coefficient, "success": 1.0,
        }
        metrics = compute_alfworld_rollout_metrics(SimpleNamespace(
            non_tensor_batch={key: [value] for key, value in final.items()}
        ))
        assert metrics["alfworld_reward/thinking_action_once_total"] == coefficient
        assert metrics["alfworld_reward/base_reward_mean"] == 10
    else:
        assert "alfworld_thinking_action_reward" not in outputs[-1].extra_fields


@pytest.mark.parametrize("mode", ["once", "repeat"])
def test_missing_or_malformed_xml_does_not_earn_bonus(mode):
    loop, calls, released, _ = episode()
    loop._thinking_enabled = True
    setattr(loop, f"_thinking_action_{mode}_reward", 1.0)
    thinking = "look" if mode == "once" else "look, look"
    pending = iter([thinking + " without closing thought", thinking + "</think>" + CALL + CALL,
                    thinking + "</think>No XML"])

    async def generate(**kwargs):
        ids = loop.tokenizer.encode(next(pending))
        return SimpleNamespace(token_ids=ids, log_probs=[-.1] * len(ids),
                               num_preempted=0, routed_experts=None)

    loop.server_manager.generate = AsyncMock(side_effect=generate)
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    assert not calls
    assert released == ["instance"]
    assert all(out.reward_score == pytest.approx(-.6) for out in outputs)
    assert outputs[-1].extra_fields["alfworld_thinking_action_reward"] == 0


@pytest.mark.parametrize("mentions", [0, 1, 2, 3, 8])
def test_repeat_bonus_is_capped_per_decision_and_accumulates_per_episode(mentions):
    loop, calls, released, _ = episode()
    loop._thinking_enabled = True
    loop._thinking_action_repeat_reward = 0.1
    text = ", ".join(["look"] * mentions) + "</think>" + CALL

    async def generate(**kwargs):
        ids = loop.tokenizer.encode(text)
        return SimpleNamespace(token_ids=ids, log_probs=[-.1] * len(ids),
                               num_preempted=0, routed_experts=None)

    loop.server_manager.generate = AsyncMock(side_effect=generate)
    outputs = asyncio.run(loop.run({}, raw_prompt=[]))
    expected_step_bonus = 0.1 if mentions >= 2 else 0.0
    assert calls == ["look"] * 3
    assert released == ["instance"]
    assert all(out.extra_fields["alfworld_thinking_action_mention_count"] == mentions for out in outputs)
    assert all(out.extra_fields["alfworld_thinking_action_step_reward"] == expected_step_bonus for out in outputs)
    assert all(out.reward_score == pytest.approx(10 + 3 * expected_step_bonus) for out in outputs)
    assert all(out.extra_fields["tool_rewards"] == [0.0, 0.0, 10.0] for out in outputs)
    final = outputs[-1].extra_fields
    assert final["alfworld_thinking_action_repeat_count"] == (3 if mentions >= 2 else 0)
    assert "alfworld_thinking_action_once_count" not in final
    metrics = compute_alfworld_rollout_metrics(SimpleNamespace(
        non_tensor_batch={key: [value] for key, value in final.items()}
    ))
    assert metrics["alfworld_reward/thinking_action_repeat_total"] == pytest.approx(3 * expected_step_bonus)
    assert metrics["alfworld_reward/thinking_action_repeat_count"] == (3 if mentions >= 2 else 0)
    assert metrics["alfworld_reward/base_reward_mean"] == 10
    assert "alfworld_reward/thinking_action_once_total" not in metrics


def test_bonus_keeps_failure_penalties_and_does_not_require_success():
    assert compute_score("alfworld", extra_info={
        "tool_rewards": [0, -.2, -.2],
        "alfworld_terminal_reason": "decision_limit",
        "alfworld_repeated_action_penalty_count": 2,
        "alfworld_thinking_action_reward": 3,
    }) == pytest.approx(2.4)


def test_raw_profile_only_changes_requested_baseline_settings(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.delenv("ALFWORLD_SFT_MODEL", raising=False)
    root = Path(__file__).resolve().parents[1] / "config/alfworld/qwen35_2b"
    configs = []
    for name in ("no_branch_entropy", "raw_once_1017"):
        with initialize_config_dir(config_dir=str(root / name), version_base=None):
            configs.append(compose(config_name="alfworld_config_2gpu"))
    baseline, raw = configs
    assert raw.actor_rollout_ref.model.path == raw.variables.BASE_MODEL
    assert raw.variables.SFT_ADAPTER is None
    assert raw.variables.THINKING_ACTION_REPEAT_REWARD == 0.1
    assert raw.variables.get("THINKING_ACTION_ONCE_REWARD", 0) == 0
    assert raw.variables.SWANLAB_NEW_RUN_ON_RESUME is False
    assert raw.trainer.experiment_name == "原生无熵正则_1017"
    assert raw.trainer.resume_mode == "auto"
    # Normalize only intentional overrides, then compare every resolved field.
    for key in ("variables.SFT_ADAPTER", "variables.SFT_MODEL", "trainer.experiment_name",
                "trainer.default_local_dir", "trainer.resume_mode"):
        OmegaConf.update(raw, key, OmegaConf.select(baseline, key))
    del raw.variables.THINKING_ACTION_REPEAT_REWARD
    del raw.variables.SWANLAB_NEW_RUN_ON_RESUME
    assert OmegaConf.to_container(raw, resolve=True) == OmegaConf.to_container(baseline, resolve=True)
