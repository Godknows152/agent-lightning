"""Protocol and observability adapters for an isolated upstream GiGPO checkout.

This module deliberately does not implement rewards, advantages or PPO updates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from alfworld_baseline.prompts_gigpo import QWEN3_ALFWORLD_CHAT_TEMPLATE, build_user_prompt
from alfworld_baseline.xml_actions import parse_xml_decision

_latest_metrics: dict[str, float] = {}
_validation_preview: list[dict] = []


def sampled_response_mask(output: list[dict], width: int, *, device: Any, dtype: Any) -> Any:
    """Mask actual SGLang tokens, supporting multiple Qwen3.5 stop token IDs."""
    import torch
    lengths = [len(item["meta_info"]["output_token_logprobs"]) for item in output]
    if any(length < 1 or length > width for length in lengths):
        raise ValueError(f"Invalid sampled response lengths {lengths} for width={width}")
    return (torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]).to(dtype)


def project_actions(actions: list[str], action_pools: list[list[str]]) -> tuple[list[str], list[int]]:
    """Map the existing XML protocol to GiGPO's single format-validity flag.

    The prompt supplies the opening think tag. Do not check admissibility or
    repeats here: upstream does neither. Invalid fallback text still steps the
    environment, as in upstream's projection function.
    """
    projected, valids = [], []
    for response in actions:
        decision = parse_xml_decision(response, enable_thinking=True)
        action = decision.action
        projected.append(action.lower() if action is not None else response.lower()[-30:])
        valids.append(int(decision.status == "valid" and not re.search(r"[\u4e00-\u9fff]", response)))
    return projected, valids


def build_observations(manager: Any, text_obs: list[str], admissible_actions: list[list[str]], init=False) -> list[str]:
    """Use the current repository's prompt verbatim, including history syntax."""
    result = []
    for i, obs in enumerate(text_obs):
        history = [
            f"[Observation {j}: '{row['text_obs']}', Action {j}: '{row['action']}']"
            for j, row in enumerate(manager.memory[i], 1)
        ]
        result.append(build_user_prompt(
            mission=manager.tasks[i], observation=obs, admissible_actions=admissible_actions[i],
            history=history, enable_thinking=True, avoid_repeated_actions=True,
        ))
    return result


def configure_tokenizer(tokenizer: Any) -> Any:
    tokenizer.chat_template = QWEN3_ALFWORLD_CHAT_TEMPLATE
    return tokenizer


def record_trajectories(collector: Any, trajectory_data: tuple, is_train: bool) -> None:
    """Persist full episodes before upstream flattens/adjusts training rows."""
    batches, rewards, lengths, success, uids, _ = trajectory_data
    root = Path(collector.config.trainer.default_local_dir)
    label = "rollouts" if is_train else "validation"
    counts = getattr(collector, "_native_dump_counts", {"rollouts": 0, "validation": 0})
    target_dir = root / label / "trajectories"
    if counts[label] == 0 and target_dir.exists():
        counts[label] = max((int(p.stem.split("_")[-1]) for p in target_dir.glob("batch_*.jsonl")), default=0)
    counts[label] += 1
    collector._native_dump_counts = counts
    target = root / label / "trajectories" / f"batch_{counts[label]:04d}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    for i, rows in enumerate(batches):
        steps = []
        for row in rows:
            if not row["active_masks"]:
                continue
            prompt_len = row["prompts"].shape[-1]
            mask = row["attention_mask"]
            steps.append({
                "step": len(steps) + 1,
                "prompt": collector.tokenizer.decode(row["prompts"][mask[:prompt_len].bool()], skip_special_tokens=False),
                "response": collector.tokenizer.decode(row["responses"][mask[prompt_len:].bool()], skip_special_tokens=False),
                "is_action_valid": bool(row["is_action_valid"]),
                "environment_reward": float(row["rewards"]),
                "training_score": float(rewards[i]) - float(collector.config.actor_rollout_ref.actor.invalid_action_penalty_coef) * (not bool(row["is_action_valid"])),
            })
        episodes.append({"trajectory_id": str(uids[i]), "reward": float(rewards[i]),
                         "length": int(lengths[i]), "steps": steps})
    with target.open("w") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
    if is_train:
        _latest_metrics.update({
            "alfworld_termination/success_count": sum(float(r) > 0 for r in rewards),
            "alfworld_termination/episode_count": len(rewards),
        })
    else:
        _validation_preview[:] = episodes[:5]
    print(f"[GiGPO native] saved {len(episodes)} complete trajectories: {target}", flush=True)


def install_tracking() -> None:
    """Add durable local metrics/experiment identity without changing training."""
    from verl.utils import tracking
    original = tracking.Tracking
    if getattr(original, "_native_alfworld", False):
        return

    class NativeTracking(original):
        _native_alfworld = True

        def __init__(self, *args, **kwargs):
            config = kwargs.get("config") or {}
            self._root = Path(config["trainer"]["default_local_dir"])
            self._root.mkdir(parents=True, exist_ok=True)
            identity = self._root / ".swanlab_experiment.json"
            resumable = (self._root / "latest_checkpointed_iteration.txt").exists()
            restart_before_first_update = not (self._root / "last_completed_step.json").exists()
            if (config["trainer"].get("resume_mode") == "auto" and identity.exists()
                    and (resumable or restart_before_first_update)):
                os.environ["SWANLAB_RUN_ID"] = json.loads(identity.read_text())["run_id"]
                os.environ["SWANLAB_RESUME"] = "must"
            super().__init__(*args, **kwargs)
            if "swanlab" in self.logger:
                import swanlab
                run = swanlab.get_run()
                (self._root / ".swanlab_experiment.json").write_text(json.dumps({
                    "run_id": run.id, "project": config["trainer"]["project_name"],
                    "experiment_name": config["trainer"]["experiment_name"],
                }, ensure_ascii=False, indent=2))

        def log(self, data, step, backend=None):
            data = dict(data)
            if "actor/grad_norm" in data:
                data.update(_latest_metrics)
            super().log(data, step, backend)
            with (self._root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": int(step), **data}, default=float) + "\n")
            if _validation_preview and "swanlab" in self.logger:
                import swanlab
                for i, episode in enumerate(_validation_preview):
                    swanlab.log({f"validation/trajectory_{i}": swanlab.Text(
                        json.dumps(episode, ensure_ascii=False, indent=2))}, step=int(step))
                _validation_preview.clear()
            if step >= 1 and "actor/grad_norm" in data:
                (self._root / "last_completed_step.json").write_text(json.dumps({
                    "step": int(step), "actor_grad_norm": float(data["actor/grad_norm"]),
                    "pid": os.getpid(),
                }))

    tracking.Tracking = NativeTracking
