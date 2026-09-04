"""Run a short ALFWorld text rollout through parser and validator."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_ROOT = PROJECT_ROOT / "contrib" / "recipes" / "envs"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("ALFWORLD_DATA", str(ENV_ROOT / "agl_envs" / "alfworld" / "alfworld_source"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    import pandas as pd, yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from alfworld_baseline.parser import parse_tool_call
    from alfworld_baseline.tool_registry import ALFWorldToolRegistry
    from alfworld_baseline.validator import validate_tool_call
    parquet = ENV_ROOT / "agl_envs" / "task_data" / "alfworld" / f"{args.split}.parquet"
    game_file = Path(str(pd.read_parquet(parquet).iloc[args.index]["game_file"]))
    if not game_file.is_absolute():
        game_file = ENV_ROOT / game_file
    with (Path(os.environ["ALFWORLD_DATA"]) / "base_config.yaml").open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
    env = AlfredTWEnv(config, train_eval="train"); env.game_files, env.num_games = [str(game_file)], 1; env = env.init_env(batch_size=1)
    observation, info = env.reset(); observation = observation[0]; actions = tuple(info["admissible_commands"][0])
    print(json.dumps({"task": str(game_file), "observation": observation[:300], "actions": actions[:5]}, ensure_ascii=False))
    done = False
    for step in range(args.steps):
        action = next((item for item in actions if item != "help"), actions[0])
        raw = (
            "<tool_call>\n<function=alfworld_action>\n<parameter=action>\n"
            f"{action}\n</parameter>\n</function>\n</tool_call>"
        )
        validation = validate_tool_call(parse_tool_call(raw), ALFWorldToolRegistry(actions))
        if not validation.is_valid: raise AssertionError(validation.error)
        (observation,), (reward,), (done,), info = env.step([validation.action]); actions = tuple(info["admissible_commands"][0])
        print(json.dumps({"step": step + 1, "action": validation.action, "reward": reward, "done": done}, ensure_ascii=False))
        if done: break
    env.close(); print(json.dumps({"status": "structured_tool_smoke_ok", "terminated": bool(done)}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
