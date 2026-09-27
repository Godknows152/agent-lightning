"""Materialize a pinned GiGPO worktree and the reviewable compatibility patch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("GIGPO_SOURCE_ROOT", "/home/LXJ/Python_Projects/GiGPO"))
TARGET = Path(os.environ.get("ALFWORLD_GIGPO_ROOT", str(ROOT / ".runtime/gigpo-native")))
REVISION = "20bd331"


def prepare() -> None:
    expected = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", REVISION], text=True).strip()
    if not TARGET.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(SOURCE), "worktree", "add", "--detach", str(TARGET), expected], check=True)
    actual = subprocess.check_output(["git", "-C", str(TARGET), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Unexpected GiGPO revision {actual}; expected {expected}")
    patch = ROOT / "patches/gigpo_native_qwen35.patch"
    command = ["git", "-C", str(TARGET), "apply"]
    if subprocess.run([*command, "--check", "--reverse", str(patch)], capture_output=True).returncode != 0:
        subprocess.run([*command, "--check", str(patch)], check=True)
        subprocess.run([*command, str(patch)], check=True)
    protected = ["verl/trainer/ppo/core_algos.py", "verl/trainer/ppo/ray_trainer.py",
                 "verl/workers/actor/dp_actor.py", "agent_system/reward_manager/episode.py",
                 "agent_system/environments/env_package/alfworld/envs.py"]
    manifest = {"source_commit": expected, "backend": str(TARGET), "unchanged_core": {}}
    for name in protected:
        upstream = subprocess.check_output(["git", "-C", str(TARGET), "show", f"HEAD:{name}"])
        content = (TARGET / name).read_bytes()
        if content != upstream:
            raise RuntimeError(f"Upstream algorithm/environment unexpectedly modified: {name}")
        manifest["unchanged_core"][name] = hashlib.sha256(content).hexdigest()
    (TARGET.parent / "gigpo_native_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Upstream's text dataset is only a modality/length placeholder, not task data.
    # Generate identical placeholder columns locally, avoiding an unrelated Hub download.
    import pandas as pd
    data = ROOT / "data/gigpo_native"
    data.mkdir(parents=True, exist_ok=True)
    for split, size, filename in [("train", 16, "train.parquet"), ("test", 128, "test.parquet")]:
        target = data / filename
        if not target.exists():
            pd.DataFrame([{"data_source": "text", "prompt": [{"role": "user", "content": ""}],
                           "ability": "agent", "extra_info": {"split": split, "index": i}}
                          for i in range(size)]).to_parquet(target, index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
