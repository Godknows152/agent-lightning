"""CPU-only native veRL tracking checks; never initialize SwanLab or Ray."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config/alfworld/qwen35_2b/v1"
BACKEND = Path(os.environ.get("ALFWORLD_VERL_ROOT", "/home/LXJ/Python_Projects/verl"))


def launch_name(config_dir: Path, kind: str = "full") -> str:
    # Execute the actual name-selection block, excluding preflight/launch code.
    script = (ROOT / "scripts/run_alfworld_grpo_2gpu.sh").read_text()
    block = script.split('EXPERIMENT_NAME="alfworld_${MODEL_PROFILE}_v1_seed${SEED}"', 1)[1]
    block = 'EXPERIMENT_NAME="alfworld_${MODEL_PROFILE}_v1_seed${SEED}"' + block.split('\nfor override', 1)[0]
    env = dict(os.environ, PYTHON_BIN=sys.executable, CONFIG_PATH=str(config_dir),
               CONFIG_NAME="alfworld_config_2gpu", MODEL_PROFILE="qwen35_2b", SEED="0", RUN_KIND=kind)
    return subprocess.check_output(
        ["bash", "-eu", "-c", block + '\nprintf "%s" "$EXPERIMENT_NAME"'], env=env, text=True
    )


def test_full_launch_respects_versioned_name():
    assert launch_name(CONFIG_DIR) == "qwen3.5_2B_0920"


@pytest.mark.parametrize("kind", ["smoke", "pilot"])
def test_diagnostic_names_remain_isolated(kind):
    assert launch_name(CONFIG_DIR, kind) == f"alfworld_qwen35_2b_v1_{kind}_seed0"


def test_unspecified_name_keeps_existing_default(tmp_path):
    (tmp_path / "alfworld_config_2gpu.yaml").write_text("trainer:\n  resume_mode: auto\n")
    assert launch_name(tmp_path) == "alfworld_qwen35_2b_v1_seed0"


def test_native_tracking_uses_composed_run_identity(tmp_path, monkeypatch):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="alfworld_config_2gpu")
    assert cfg.trainer.resume_mode == "auto"
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = str(tmp_path / "global_step_20")
    assert cfg.trainer.experiment_name == launch_name(CONFIG_DIR)
    # Use a temporary marker and fake SDK: no cloud writes or artifact changes.
    cfg.trainer.default_local_dir = str(tmp_path)
    marker = {"experiment_name": cfg.trainer.experiment_name, "run_id": "test-original-run"}
    marker_path = tmp_path / ".swanlab_experiment.json"
    marker_path.write_text(json.dumps(marker))
    init_calls, log_calls = [], []
    fake = SimpleNamespace(init=lambda **kw: (init_calls.append(kw) or SimpleNamespace(id="test-original-run")),
                           log=lambda **kw: log_calls.append(kw), finish=lambda: None)
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("tracking_resume_test", BACKEND / "verl/utils/tracking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from alfworld_baseline.tracking import swanlab_resume
    with swanlab_resume(cfg):
        tracking = module.Tracking(cfg.trainer.project_name, cfg.trainer.experiment_name,
                                  default_backend=["swanlab"], config=OmegaConf.to_container(cfg, resolve=True))
    assert init_calls[0]["project"] == "ALFWorldRL"
    assert init_calls[0]["experiment_name"] == cfg.trainer.experiment_name
    assert init_calls[0]["resume"] == "must"
    assert init_calls[0]["id"] == "test-original-run"
    tracking.log({"test_metric": 1.0}, step=21)
    assert log_calls == [{"data": {"test_metric": 1.0}, "step": 21}]
    assert json.loads(marker_path.read_text()) == marker


def test_new_run_identity_is_saved_before_training_and_patch_is_restored(tmp_path, monkeypatch):
    from alfworld_baseline.tracking import swanlab_resume
    cfg = OmegaConf.create({'trainer': {'logger': ['swanlab'], 'default_local_dir': str(tmp_path),
                           'experiment_name': 'new-run', 'resume_mode': 'auto'}})
    calls = []
    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id='new-id')
    fake = SimpleNamespace(init=init, finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'swanlab', fake)
    with pytest.raises(RuntimeError, match='training failed'):
        with swanlab_resume(cfg):
            fake.init(project='test', experiment_name='new-run')
            assert json.loads((tmp_path / '.swanlab_experiment.json').read_text())['run_id'] == 'new-id'
            raise RuntimeError('training failed')
    assert fake.init is init
    assert 'resume' not in calls[0]


@pytest.mark.parametrize('case', ['missing_identity', 'wrong_name', 'orphan_identity', 'disable_existing'])
def test_resume_rejects_invalid_identity_before_sdk_init(tmp_path, monkeypatch, case):
    from alfworld_baseline.tracking import swanlab_resume
    cfg = OmegaConf.create({'trainer': {'logger': ['swanlab'], 'default_local_dir': str(tmp_path),
                           'experiment_name': 'run', 'resume_mode': 'auto'}})
    if case != 'missing_identity':
        (tmp_path / '.swanlab_experiment.json').write_text(json.dumps({
            'experiment_name': 'other' if case == 'wrong_name' else 'run', 'run_id': 'id'}))
    if case != 'orphan_identity':
        (tmp_path / 'latest_checkpointed_iteration.txt').write_text('10')
    if case == 'disable_existing':
        cfg.trainer.resume_mode = 'disable'
    fake = SimpleNamespace(init=lambda **kw: pytest.fail('must reject before SDK init'))
    monkeypatch.setitem(sys.modules, 'swanlab', fake)
    with pytest.raises(ValueError):
        with swanlab_resume(cfg):
            pass
