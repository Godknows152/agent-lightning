"""Check complete validation previews against native V1 validation on CPU."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from alfworld_baseline.validation_logging import ALFWorldValidationLoggingMixin, trajectory_samples
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def test_transcripts_keep_all_steps_in_numeric_order_and_sessions_separate():
    indices = [10, 2, 0, 9, 1, 8, 3, 7, 4, 6, 5]
    keys = [f"task_with_underscores_0_{i}" for i in indices] + ["task_with_underscores_1_0"]
    inputs = [f"observation {i}" for i in indices] + ["second rollout"]
    outputs = [f"<think>reason {i}</think> action {i}" for i in indices] + ["malformed XML"]
    samples = trajectory_samples(keys, inputs, outputs, [float(i) for i in indices] + [-2.0])
    assert len(samples) == 2
    prompt, transcript, score = samples[0]
    assert prompt == "observation 0"
    assert score == 10.0
    assert transcript.startswith("Trajectory: 11 decision steps")
    positions = [transcript.index(f"===== Step {i + 1} =====") for i in range(11)]
    assert positions == sorted(positions)
    assert all(text in transcript for text in inputs[:-1] + outputs[:-1])
    assert "malformed XML" not in transcript
    assert samples[1][0] == "second rollout"
    assert "malformed XML" in samples[1][1]
    assert samples[1][2] == -2.0


class NativeValidationHarness:
    """Use native collection, sampling, and JSONL export without Ray or models."""

    _validate = PPOTrainer._validate
    _maybe_log_val_generations = PPOTrainer._maybe_log_val_generations
    _dump_generations = PPOTrainer._dump_generations
    _write_generations = staticmethod(PPOTrainer._write_generations)
    _init_dump_executor = PPOTrainer._init_dump_executor
    _shutdown_dump_executor = PPOTrainer._shutdown_dump_executor

    def __init__(self, dump_path, count):
        self.config = OmegaConf.create(
            {
                "trainer": {"validation_data_dir": dump_path, "log_val_generations": count, "logger": ["swanlab"]},
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
            }
        )
        self.global_steps = 40
        self.validation_generations_logger = SimpleNamespace(log=Mock())
        self.val_dataloader = [{"raw_prompt": np.array(["task", "other"], dtype=object)}]
        self.agent_loop_manager = SimpleNamespace(generate_sequences=Mock())
        self.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
        self._init_dump_executor()

    def _val_metrics_update(self, sources, uids, rewards, turns, **kwargs):
        self.collected_metrics = (sources, uids, rewards, turns)
        return {"reward_mean": float(np.mean(rewards["reward"]))}


class FullValidationHarness(ALFWorldValidationLoggingMixin, NativeValidationHarness):
    pass


@pytest.mark.parametrize("sample_count", [0, 2, 8])
def test_native_validation_logs_episodes_once_and_preserves_rewards_and_step_dump(monkeypatch, tmp_path, sample_count):
    import transfer_queue as tq

    # Deliberately interleave sessions and step order. One task has two rollouts.
    keys = ["task_0_2", "task_0_0", "other_0_0", "task_1_0", "task_0_1"]
    prompts = ["third observation", "initial room", "other room", "another rollout", "second observation"]
    responses = ["final action", "<think>first</think>look", "no parseable action", "success", "invalid XML"]
    scores = [6.0, 6.0, -2.0, 10.0, 6.0]
    batch = SimpleNamespace(keys=keys, partition_id="val")
    text_by_id = {i: text for i, text in enumerate(prompts + responses)}

    def get_rows(*, keys, partition_id, select_fields):
        assert partition_id == "val"
        if select_fields == ["prompts", "responses"]:
            return {
                "prompts": SimpleNamespace(to_padded_tensor=lambda **kw: [[i] for i in range(5)]),
                "responses": SimpleNamespace(to_padded_tensor=lambda **kw: [[i] for i in range(5, 10)]),
            }
        indices = [batch.keys.index(key) for key in keys]
        return {
            "uid": np.array([key.rsplit("_", 2)[0] for key in keys], dtype=object),
            "rm_scores": torch.tensor([[scores[i]] for i in indices]),
            "num_turns": np.ones(len(keys), dtype=int),
            "reward_model": np.array([{"ground_truth": "goal"}] * len(keys), dtype=object),
            "data_source": np.array(["alfworld"] * len(keys), dtype=object),
            "extra_fields": np.array([{"reward_extra_info": {"score": scores[i]}} for i in indices], dtype=object),
        }

    monkeypatch.setattr(tq, "kv_batch_put", Mock())
    monkeypatch.setattr(tq, "kv_batch_get", get_rows)
    monkeypatch.setattr(tq, "kv_clear", Mock())
    trainer = FullValidationHarness(str(tmp_path / "validation"), sample_count)
    trainer.replay_buffer = SimpleNamespace(sample=lambda **kw: (batch, None))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: text_by_id[ids[0]])
    try:
        assert trainer._validate() == {"reward_mean": pytest.approx(14.0 / 3)}
        assert not trainer._alfworld_full_validation
        # One score and one native turn count per episode, not per displayed step.
        assert trainer.collected_metrics[2]["reward"] == [6.0, -2.0, 10.0]
        assert trainer.collected_metrics[3] == [1, 1, 1]
        logger = trainer.validation_generations_logger.log
        if sample_count:
            logger.assert_called_once()
            loggers, samples, step = logger.call_args.args
            assert loggers == ["swanlab"] and step == 40
            assert len(samples) == min(sample_count, 3)
            assert all("Trajectory:" in output for _, output, _ in samples)
            if sample_count == 8:
                initial, transcript, score = next(row for row in samples if row[0] == "initial room")
                assert score == 6.0
                assert all(
                    text in transcript for text in [prompts[i] for i in (0, 1, 4)] + [responses[i] for i in (0, 1, 4)]
                )
                assert (
                    transcript.index("first</think>")
                    < transcript.index("invalid XML")
                    < transcript.index("final action")
                )
        else:
            logger.assert_not_called()
        # A subsequent training export must not produce another validation preview.
        trainer._dump_generations(prompts, responses, [None] * 5, scores, {"uid": keys}, str(tmp_path / "rollouts"))
        assert logger.call_count == int(bool(sample_count))
    finally:
        trainer._shutdown_dump_executor()
    saved = [json.loads(line) for line in (tmp_path / "validation/40.jsonl").read_text().splitlines()]
    assert len(saved) == 5
    assert [row["uid"] for row in saved] == ["other_0_0", "task_0_0", "task_0_1", "task_0_2", "task_1_0"]
    assert saved[2]["output"] == "invalid XML"
    assert saved[2]["score"] == 6.0


def test_validation_failure_clears_preview_state(tmp_path):
    trainer = FullValidationHarness(str(tmp_path), 8)
    trainer.val_dataloader = None
    try:
        with pytest.raises(TypeError):
            trainer._validate()
        assert not trainer._alfworld_full_validation
    finally:
        trainer._shutdown_dump_executor()
