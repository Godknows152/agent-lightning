"""Resume the original cloud run even when checkpoint migration changes output directories."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.tracking import Tracking


@pytest.fixture
def resume_config(tmp_path):
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(version_base=None, config_dir=str(root / "old_verl_grpo/config/fog/v4.1.4")):
        config = OmegaConf.to_container(compose(config_name="fog_config_2gpu"), resolve=True)
    config["trainer"]["default_local_dir"] = str(tmp_path / "new-output")
    return config


@pytest.fixture
def sdk(monkeypatch):
    fake = SimpleNamespace(init=MagicMock(), log=MagicMock(), finish=MagicMock(),
                           get_run=lambda: SimpleNamespace(id="g7vpp7ps"))
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    return fake


def make_tracking(config):
    return Tracking(config["trainer"]["project_name"], config["trainer"]["experiment_name"],
                    default_backend=["swanlab"], config=config)


def test_new_output_directory_resumes_original_id_and_logs_step_141(resume_config, sdk):
    tracking = make_tracking(resume_config)
    call = sdk.init.call_args.kwargs
    assert call["project"] == "FogRL"
    assert call["experiment_name"] == "fog_v4.1.4_0.4熵正则"
    assert call["id"] == "g7vpp7ps"
    assert call["resume"] == "must"
    tracking.log({"training/global_step": 141}, step=141)
    sdk.log.assert_called_once_with(data={"training/global_step": 141}, step=141)
    marker = Path(resume_config["trainer"]["default_local_dir"]) / ".swanlab_experiment.json"
    assert json.loads(marker.read_text()) == {"experiment_name": call["experiment_name"], "run_id": "g7vpp7ps"}


def test_conflicting_output_marker_fails_before_cloud_initialization(resume_config, sdk):
    output = Path(resume_config["trainer"]["default_local_dir"])
    output.mkdir()
    (output / ".swanlab_experiment.json").write_text(json.dumps({
        "experiment_name": resume_config["trainer"]["experiment_name"], "run_id": "wrong-run"}))
    with pytest.raises(ValueError, match="resume run ID does not match"):
        make_tracking(resume_config)
    sdk.init.assert_not_called()


def test_disabled_resume_does_not_attach_smoke_run_to_original(resume_config, sdk):
    resume_config["trainer"]["resume_mode"] = "disable"
    make_tracking(resume_config)
    assert sdk.init.call_args.kwargs["resume"] is None
    assert "id" not in sdk.init.call_args.kwargs


def test_existing_marker_still_resumes_without_explicit_id(resume_config, sdk):
    del resume_config["trainer"]["swanlab_resume_run_id"]
    output = Path(resume_config["trainer"]["default_local_dir"])
    output.mkdir()
    (output / ".swanlab_experiment.json").write_text(json.dumps({
        "experiment_name": resume_config["trainer"]["experiment_name"], "run_id": "g7vpp7ps"}))
    make_tracking(resume_config)
    assert sdk.init.call_args.kwargs["id"] == "g7vpp7ps"
    assert sdk.init.call_args.kwargs["resume"] == "must"
