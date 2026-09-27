"""Launch the GiGPO TaskRunner directly with this repository's configuration."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import hydra
from omegaconf import OmegaConf

from alfworld_baseline.backend import assert_native_verl
from alfworld_baseline.ray_startup import install_ray_agent_port_guard


@hydra.main(config_path=None, config_name="alfworld_config_2gpu", version_base=None)
def main(config):
    assert_native_verl()
    from verl.trainer.main_ppo import run_ppo
    install_ray_agent_port_guard()
    OmegaConf.resolve(config)
    output = Path(config.trainer.default_local_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(os.environ["ALFWORLD_GIGPO_ROOT"]).parent / "gigpo_native_manifest.json",
                    output / "backend_manifest.json")
    run_ppo(config)


if __name__ == "__main__":
    main()
