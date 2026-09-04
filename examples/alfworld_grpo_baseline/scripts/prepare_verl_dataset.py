"""Materialize ALFWorld tasks into old-VERL prompt parquet files."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
ENV_ROOT = PROJECT_ROOT / "contrib" / "recipes" / "envs"
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--profile", choices=("qwen25", "qwen35"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    os.environ.setdefault("ALFWORLD_DATA", str(ENV_ROOT / "agl_envs" / "alfworld" / "alfworld_source"))

    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from alfworld_baseline.datasets import load_tasks
    from alfworld_baseline.prompt_profiles import get_prompt_profile

    prompt_profile = get_prompt_profile(args.profile)

    source = ENV_ROOT / "agl_envs" / "task_data" / "alfworld" / f"{args.split}.parquet"
    tasks = load_tasks(source, limit=args.limit)
    with (Path(os.environ["ALFWORLD_DATA"]) / "base_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
    rows: list[dict] = []
    for start in range(0, len(tasks), args.chunk_size):
        chunk = tasks[start : start + args.chunk_size]
        game_files = []
        for task in chunk:
            game_file = Path(str(task["game_file"]))
            if not game_file.is_absolute():
                game_file = ENV_ROOT / game_file
            game_files.append(str(game_file))
        env = AlfredTWEnv(config, train_eval="train")
        env.game_files, env.num_games = game_files, len(game_files)
        env = env.init_env(batch_size=len(game_files))
        observations, info = env.reset()
        for offset, (game_file, observation) in enumerate(zip(game_files, observations, strict=True)):
            actions = tuple(info["admissible_commands"][offset])
            observation = str(observation)
            mission = observation.split("Your task is to: ", 1)[-1] if "Your task is to: " in observation else observation
            prompt = [
                {"role": "system", "content": prompt_profile.SYSTEM_PROMPT},
                {"role": "user", "content": prompt_profile.build_user_prompt(mission=mission, observation=observation, admissible_actions=actions)},
            ]
            index = start + offset
            rows.append({"data_source": "alfworld", "agent_name": "alfworld_tool_agent", "prompt": copy.deepcopy(prompt), "reward_model": {"style": "rule", "ground_truth": {"game_file": game_file}}, "extra_info": {"index": index, "sample_id": f"{args.split}-{index:06d}", "game_file": game_file, "prompt_profile": args.profile, "prompt_version": prompt_profile.PROMPT_VERSION, "need_tools_kwargs": True, "tools_kwargs": {"alfworld_action": {"create_kwargs": {"game_file": game_file}}}}})
        env.close()
        print(f"processed={min(start + args.chunk_size, len(tasks))}/{len(tasks)}", flush=True)

    output = args.output or (ROOT / "data" / args.profile / f"{args.split}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, engine="pyarrow", index=False)
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
