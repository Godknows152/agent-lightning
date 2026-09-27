"""Run with the GiGPO checkout first in PYTHONPATH (not native VERL V1)."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from alfworld_baseline.gigpo_native import build_observations, configure_tokenizer, project_actions, sampled_response_mask
from alfworld_baseline.prompts_gigpo import QWEN3_ALFWORLD_CHAT_TEMPLATE, build_user_prompt


def response(action="look"):
    return "I should inspect.</think>\n<tool_call>\n<function=alfworld_action>\n<parameter=action>" + action + "</parameter>\n</function>\n</tool_call>"


@pytest.mark.parametrize("text,valid", [
    (response(), 1), (response("invented action"), 1),
    ("unfinished reasoning", 0), ("done.</think>", 0),
    (response().replace("alfworld_action", "wrong_tool"), 0),
    (response() + response(), 0), ("中文" + response(), 0),
])
def test_upstream_format_penalty_boundary(text, valid):
    actions, valids = project_actions([text], [["look"]])
    assert valids == [valid]
    assert len(actions) == 1


def test_repeats_are_valid():
    assert project_actions([response(), response()], [["look"], ["look"]]) == (["look", "look"], [1, 1])


def test_multiple_stop_ids_do_not_train_on_padding():
    # EOS used by the engine differs from tokenizer.eos; padding must stay masked.
    output = [{"meta_info": {"output_token_logprobs": [(-.2, 7, None), (-.1, 248046, None)]}},
              {"meta_info": {"output_token_logprobs": [(-.3, 8, None)]}}]
    mask = sampled_response_mask(output, 4, device="cpu", dtype=torch.long)
    torch.testing.assert_close(mask, torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]))


def test_prompt_is_existing_prompt():
    memory = [[{"text_obs": "room", "action": "look"}, {"text_obs": "drawer", "action": "open drawer 1"}]]
    manager = SimpleNamespace(memory=memory, tasks=["find a key"])
    actual = build_observations(manager, ["key inside"], [["take key 1 from drawer 1"]])[0]
    history = [f"[Observation {i}: '{r['text_obs']}', Action {i}: '{r['action']}']" for i, r in enumerate(memory[0], 1)]
    expected = build_user_prompt(mission="find a key", observation="key inside",
        admissible_actions=["take key 1 from drawer 1"], history=history,
        enable_thinking=True, avoid_repeated_actions=True)
    assert actual == expected
    assert configure_tokenizer(SimpleNamespace()).chat_template == QWEN3_ALFWORLD_CHAT_TEMPLATE


def test_native_reward_penalty_and_cross_step_advantage():
    import verl
    if not (Path(verl.__file__).resolve().parent.parent / "agent_system/reward_manager/episode.py").exists():
        pytest.skip("Run this integration test with ALFWORLD_VERL_ROOT pointing to the GiGPO checkout")
    from verl import DataProto
    from agent_system.reward_manager.episode import EpisodeRewardManager
    from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    data = DataProto(batch=TensorDict({
        "prompts": torch.ones(5, 2, dtype=torch.long),
        "responses": torch.ones(5, 3, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * 5),
    }, batch_size=[5]), non_tensor_batch={
        "data_source": np.array(["text"] * 5, dtype=object),
        "episode_rewards": np.array([10., 10., 0., 0., 0.]),
        "episode_lengths": np.array([2, 2, 3, 3, 3]),
        "is_action_valid": np.array([True, False, True, False, False]),
    })
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "example")
    data.batch["token_level_scores"] = EpisodeRewardManager(tokenizer, 0)(data)
    data, _ = apply_invalid_action_penalty(data, 0.1)
    expected = torch.tensor([10., 9.9, 0., -.1, -.1])
    torch.testing.assert_close(data.batch["token_level_scores"].sum(-1), expected)
    mask = data.batch["attention_mask"][:, 2:]
    adv, _ = compute_grpo_outcome_advantage(data.batch["token_level_scores"], mask,
        np.array(["task"] * 5), np.array(["a", "a", "b", "b", "b"]))
    expected_adv = ((expected - expected.mean()) / (expected.std() + 1e-6))[:, None] * mask
    torch.testing.assert_close(adv, expected_adv)
