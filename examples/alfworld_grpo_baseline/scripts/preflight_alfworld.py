"""Check dependencies and data in the isolated ALFWorld environment."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
ENV_ROOT = PROJECT_ROOT / "contrib" / "recipes" / "envs"
DATA_ROOT = ENV_ROOT / "agl_envs" / "alfworld" / "alfworld_source"
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    import importlib.util
    required = ["alfworld", "gymnasium", "stable_baselines3", "pandas", "pyarrow", "omegaconf"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"ALFWorld environment missing dependencies: {missing}")
    if not (DATA_ROOT / "base_config.yaml").is_file():
        raise FileNotFoundError(DATA_ROOT / "base_config.yaml")
    from alfworld_baseline.datasets import load_tasks
    for split in ("train", "test"):
        path = ENV_ROOT / "agl_envs" / "task_data" / "alfworld" / f"{split}.parquet"
        rows = load_tasks(path, limit=3)
        print(f"{split}_sampled={len(rows)} path={path}")
    print("status=alfworld_preflight_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
