"""Deterministic, text-only ALFWorld and AGL adapter smoke test."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGL_ENVS = ROOT / "agl_envs"
os.environ.setdefault("ALFWORLD_DATA", str(AGL_ENVS / "alfworld" / "alfworld_source"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--agl", action="store_true", help="also exercise the AGL EnvironmentManager adapter")
    args = parser.parse_args()

    import pandas as pd
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    parquet = ROOT / "agl_envs" / "task_data" / "alfworld" / f"{args.split}.parquet"
    frame = pd.read_parquet(parquet)
    if frame.empty or "game_file" not in frame:
        raise RuntimeError(f"invalid dataset: {parquet}")
    task = frame.iloc[args.index].to_dict()
    game_file = Path(task["game_file"])
    if not game_file.is_absolute():
        game_file = ROOT / game_file
    if not game_file.is_file():
        raise FileNotFoundError(game_file)

    data_dir = Path(os.environ["ALFWORLD_DATA"])
    config = yaml.safe_load((data_dir / "base_config.yaml").read_text())
    AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
    raw = AlfredTWEnv(config, train_eval="train")
    raw.game_files, raw.num_games = [str(game_file)], 1
    raw = raw.init_env(batch_size=1)
    observation, info = raw.reset()
    observation = observation[0]
    actions = info["admissible_commands"][0]
    if not observation.strip() or not actions:
        raise AssertionError("reset returned empty observation or admissible actions")
    print(json.dumps({"task": str(game_file), "observation": observation[:300], "actions": actions[:5]}, ensure_ascii=False))
    done = False
    for step in range(args.steps):
        action = next((a for a in actions if a != "help"), actions[0])
        (observation,), (reward,), (done,), info = raw.step([action])
        actions = info["admissible_commands"][0]
        print(json.dumps({"step": step + 1, "action": action, "reward": reward, "done": done, "actions": actions[:5]}))
        if done:
            break
    raw.close()
    print(json.dumps({"status": "native_ok", "terminated": bool(done), "max_steps": args.max_steps}))

    if args.agl:
        from omegaconf import OmegaConf
        from agl_envs import make_env_manager
        from prompt_builder import HistoryPromptBuilder

        cfg = OmegaConf.load(ROOT / "config_env" / "alfworld.yaml")
        cfg.alfworld_kwargs.max_steps = args.max_steps
        manager = make_env_manager("alfworld", {"game_file": str(game_file)}, cfg)
        obs, info, hints = manager.reset()
        builder = HistoryPromptBuilder(max_history=cfg.captioner.max_history, prompt_type="single")
        builder.init(manager)
        builder.update_observation(obs)
        builder.update_admissible_actions(hints)
        prompt = builder.get_prompt()[0].content
        action = next(a for a in manager.env.available_actions if a != "help")
        result = manager.step(action, use_reasoning=False)
        manager.close()
        if not prompt.strip() or len(result) != 8:
            raise AssertionError("AGL adapter contract failed")
        print(json.dumps({"status": "agl_ok", "prompt": prompt[:300], "result_types": [type(x).__name__ for x in result]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
