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


@pytest.mark.parametrize("error,state,exit_code", [
    (None, "success", 0),
    (RuntimeError("reference pinned allocation failed"), "crashed", 1),
    (KeyboardInterrupt(), "aborted", 1),
])
def test_native_tracking_preserves_terminal_state(monkeypatch, error, state, exit_code):
    from alfworld_baseline.tracking import finish_tracking

    calls = []
    fake = SimpleNamespace(init=lambda **kw: None, finish=lambda **kw: calls.append(kw))
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("tracking_finish_test", BACKEND / "verl/utils/tracking.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracking = module.Tracking("test", "test", default_backend=["swanlab"])
    other_calls = []
    tracking.logger["wandb"] = SimpleNamespace(finish=lambda **kw: other_calls.append(kw))
    finish_tracking(tracking, error=error)
    finish_tracking(tracking, error=error)
    tracking.finish()
    expected = {"state": state}
    if state == "crashed":
        expected["error"] = "RuntimeError: reference pinned allocation failed"
    assert calls == [expected]
    assert other_calls == [{"exit_code": exit_code}]


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
    config = OmegaConf.load(CONFIG_DIR / "alfworld_config_2gpu.yaml")
    assert launch_name(CONFIG_DIR) == config.trainer.experiment_name


@pytest.mark.parametrize("kind", ["smoke", "pilot"])
def test_diagnostic_names_remain_isolated(kind):
    assert launch_name(CONFIG_DIR, kind) == f"alfworld_qwen35_2b_v1_{kind}_seed0"


def test_unspecified_name_keeps_existing_default(tmp_path):
    (tmp_path / "alfworld_config_2gpu.yaml").write_text("trainer:\n  resume_mode: auto\n")
    assert launch_name(tmp_path) == "alfworld_qwen35_2b_v1_seed0"


@pytest.mark.parametrize("kind,explicit,expected", [
    ("full", None, None), ("full", "42", "42"), ("full", "null", "null"),
    ("smoke", None, "1"), ("pilot", None, "5"), ("pilot", "7", "7"),
])
def test_launcher_preserves_yaml_budget_and_diagnostic_caps(tmp_path, kind, explicit, expected):
    script = (ROOT / "scripts/run_alfworld_grpo_2gpu.sh").read_text()
    assignment = next(line for line in script.splitlines() if line.startswith("TOTAL_STEPS="))
    diagnostic = 'if [[ "${RUN_KIND}" == "smoke" ]]; then' + script.split(
        'if [[ "${RUN_KIND}" == "smoke" ]]; then', 1
    )[1].split('\nif [[ "${TRAINING_BACKEND}"', 1)[0]
    overrides = "overrides=(\n" + script.split("\noverrides=(\n", 1)[1].split("# Native veRL", 1)[0]
    env = dict(os.environ, RUN_KIND=kind, ROOT=str(tmp_path), MODEL_PROFILE="qwen35_2b",
               SEED="0", OUTPUT_DIR=str(tmp_path), LOG_DIR=str(tmp_path),
               SWANLAB_LOG_DIR=str(tmp_path), SWANLAB_MODE="offline", EXPERIMENT_NAME="test")
    env.pop("ALFWORLD_TOTAL_STEPS", None)
    if explicit is not None:
        env["ALFWORLD_TOTAL_STEPS"] = explicit
    args = subprocess.check_output(
        ["bash", "-eu", "-c", assignment + "\n" + diagnostic + "\n" + overrides
         + '\nprintf "%s\\n" "${overrides[@]}"'], env=env, text=True,
    ).splitlines()
    step_args = [arg for arg in args if arg.startswith("trainer.total_training_steps=")]
    assert step_args == ([] if expected is None else [f"trainer.total_training_steps={expected}"])


@pytest.mark.parametrize("resume_overlay", [False, True])
def test_native_tracking_uses_composed_run_identity(tmp_path, monkeypatch, resume_overlay):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        overrides = ["+training_backend=gigpo_grpo", "+resume_run=qwen35_2b_gigpo"] if resume_overlay else []
        cfg = compose(config_name="alfworld_config_2gpu", overrides=overrides)
    if resume_overlay:
        assert cfg.trainer.resume_mode == "auto"
        assert cfg.trainer.resume_from_path is None
        assert cfg.trainer.default_local_dir == str(ROOT / "log/alfworld/qwen35_2b/gigpo_grpo/full_sft")
        assert cfg.trainer.experiment_name == "qwen3.5_2B_GiGPO后端_full_sft"
        assert cfg.trainer.total_training_steps == 150
        assert cfg.ray_kwargs.ray_init._node_ip_address == os.environ.get("ALFWORLD_RAY_NODE_IP", "10.246.1.30")
    else:
        assert cfg.trainer.resume_mode == "auto"
        assert cfg.trainer.total_epochs == 3
        assert cfg.trainer.total_training_steps is None
        assert cfg.trainer.experiment_name == launch_name(CONFIG_DIR)
    if resume_overlay:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")
    else:
        cfg.trainer.resume_mode = "resume_path"
        cfg.trainer.resume_from_path = str(tmp_path / "global_step_20")
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


@pytest.mark.parametrize("kind,expected", [("full", True), ("preflight", True), ("smoke", False), ("pilot", False)])
def test_resume_overlay_selection_matches_preflight(kind, expected):
    script = (ROOT / "scripts/run_alfworld_grpo_2gpu.sh").read_text()
    block = "resume_preflight_args=()" + script.split("resume_preflight_args=()", 1)[1].split('\nOUTPUT_DIR=', 1)[0]
    env = dict(os.environ, ALFWORLD_RESUME_CONFIG="qwen35_2b_gigpo", MODEL_PROFILE="qwen35_2b",
               TRAINING_BACKEND="gigpo_grpo", RUN_KIND=kind)
    result = subprocess.check_output(
        ["bash", "-eu", "-c", 'backend_overrides=()\n' + block +
         '\nprintf "%s\\n" "${backend_overrides[@]}" "${resume_preflight_args[@]}"'], env=env, text=True)
    assert ("+resume_run=qwen35_2b_gigpo" in result) == expected
    assert ("--resume-config\nqwen35_2b_gigpo" in result) == expected


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


@pytest.mark.parametrize("mode", ["auto", "resume_path"])
def test_explicit_new_run_preserves_name_checkpoint_step_and_previous_identity(tmp_path, monkeypatch, mode):
    from alfworld_baseline.tracking import swanlab_resume

    cfg = OmegaConf.create({
        "variables": {"SWANLAB_NEW_RUN_ON_RESUME": True},
        "trainer": {"logger": ["swanlab"], "default_local_dir": str(tmp_path),
                    "experiment_name": "原生无熵正则_1017", "resume_mode": mode,
                    "resume_from_path": str(tmp_path / "global_step_40")},
    })
    marker = tmp_path / ".swanlab_experiment.json"
    previous = {"experiment_name": cfg.trainer.experiment_name, "run_id": "old-id"}
    marker.write_text(json.dumps(previous))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("40")
    calls, logs = [], []

    def init(**kwargs):
        calls.append(kwargs)
        if kwargs.get("resume") == "never":
            assert "id" not in kwargs  # Actual SwanLab 0.9 SDK contract.
        return SimpleNamespace(id=kwargs.get("id", "new-id"))

    fake = SimpleNamespace(init=init, log=lambda data, step: logs.append((data, step)), finish=lambda: None)
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    with swanlab_resume(cfg):
        fake.init(project="ALFWorldRL", experiment_name=cfg.trainer.experiment_name)
    new = json.loads(marker.read_text())
    assert new["run_id"] != previous["run_id"]
    assert new["experiment_name"] == previous["experiment_name"]
    assert calls[0]["resume"] == "never"
    assert "id" not in calls[0]
    history = json.loads(next((tmp_path / "swanlab_identity_history").glob("*.json")).read_text())
    assert history == {"previous": previous, "current": new, "resume_from_path": str(tmp_path / "global_step_40")}
    assert logs == [({"training/resumed_from_step": 40}, 40)]
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "40"
    assert fake.init is init
    # Ordinary future restarts reconnect to this new identity, without resetting again.
    cfg.variables.SWANLAB_NEW_RUN_ON_RESUME = False
    with swanlab_resume(cfg):
        fake.init(project="ALFWorldRL", experiment_name=cfg.trainer.experiment_name)
    assert calls[-1]["id"] == new["run_id"]
    assert calls[-1]["resume"] == "must"
    assert len(logs) == 1


def test_new_tracking_run_requires_checkpoint_resume(tmp_path, monkeypatch):
    from alfworld_baseline.tracking import swanlab_resume

    cfg = OmegaConf.create({
        "variables": {"SWANLAB_NEW_RUN_ON_RESUME": True},
        "trainer": {"logger": ["swanlab"], "default_local_dir": str(tmp_path),
                    "experiment_name": "run", "resume_mode": "disable"},
    })
    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=lambda **kw: pytest.fail("unexpected init")))
    with pytest.raises(ValueError, match="requires an existing checkpoint"):
        with swanlab_resume(cfg):
            pass


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
