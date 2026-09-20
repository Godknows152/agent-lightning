"""Regression checks for cross-world-size LoRA continuation."""

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "old_verl_grpo/scripts/export_LoRA"


@pytest.fixture
def converter(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("prepare_lora_resume_state", SCRIPTS / "prepare_lora_resume_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_resume_state


def make_checkpoint(tmp_path):
    source = tmp_path / "source/global_step_140"
    actor = source / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "huggingface/config.json").write_text("{}")
    (actor / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 1, "world_size": 4}))
    torch.save({"consumed_batches": 16}, source / "data.pt")
    for rank in range(4):
        torch.save({"layer.lora_A.default.weight": torch.zeros(2, 4)}, actor / f"model_world_size_4_rank_{rank}.pt")
        values = torch.arange(rank * 2, rank * 2 + 2, dtype=torch.float32)
        torch.save({"state": {0: {}, 1: {"exp_avg": values, "exp_avg_sq": values.square(), "step": torch.tensor(280.)}},
                    "param_groups": [{"params": [0, 1], "lr": 1.4708205381669757e-5}]},
                   actor / f"optim_world_size_4_rank_{rank}.pt")
        torch.save({"lr_scheduler": {"last_epoch": 140}, "rng": rank}, actor / f"extra_state_world_size_4_rank_{rank}.pt")
    return source


def test_reshards_adam_without_resetting_progress(tmp_path, converter):
    source = make_checkpoint(tmp_path)
    dest = tmp_path / "target/global_step_140"
    converter(source, dest, 2)
    for rank in range(2):
        state = torch.load(dest / "actor" / f"optim_world_size_2_rank_{rank}.pt", weights_only=False)
        assert torch.equal(state["state"][1]["exp_avg"], torch.arange(rank * 4, rank * 4 + 4).float())
        assert torch.equal(state["state"][1]["exp_avg_sq"], torch.arange(rank * 4, rank * 4 + 4).float().square())
        assert state["state"][1]["step"].item() == 280
        assert state["state"][0] == {}
        assert state["param_groups"][0]["lr"] == 1.4708205381669757e-5
        assert (dest / "actor" / f"extra_state_world_size_2_rank_{rank}.pt").read_bytes() == (
            source / "actor" / f"extra_state_world_size_4_rank_{rank}.pt").read_bytes()
    assert (dest / "data.pt").read_bytes() == (source / "data.pt").read_bytes()
    assert len(list((source / "actor").glob("model_world_size_4_rank_*.pt"))) == 4
    with pytest.raises(ValueError, match="new directory"):
        converter(source, dest, 2)


def test_rejects_inconsistent_optimizer_steps(tmp_path, converter):
    source = make_checkpoint(tmp_path)
    path = source / "actor/optim_world_size_4_rank_3.pt"
    state = torch.load(path, weights_only=False)
    state["state"][1]["step"] += 1
    torch.save(state, path)
    dest = tmp_path / "target/global_step_140"
    with pytest.raises(ValueError, match="Scalar optimizer state differs"):
        converter(source, dest, 2)
    assert not dest.exists()


@pytest.mark.parametrize("separate_reference", [False, True])
def test_reference_adapter_does_not_change_actor_adapter(monkeypatch, separate_reference):
    from verl.workers import engine_workers
    from verl.workers.config.model import HFModelConfig

    # Exercise the actual worker initialization branch, without loading a 9B model or using GPUs.
    monkeypatch.setattr(HFModelConfig, "__post_init__", lambda self: None)
    worker_factory = MagicMock()
    worker_factory.return_value.get_dispatch_collect.return_value = {}
    monkeypatch.setattr(engine_workers, "TrainingWorker", worker_factory)
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", lambda **kwargs: None)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "old_verl_grpo/config/fog/v4.1.4")):
        config = compose(config_name="fog_config_2gpu").actor_rollout_ref
    config.model.lora_adapter_path = "/actor/step140/fog"
    config.model.reference_lora_adapter_path = "/reference/sft/fog" if separate_reference else None
    worker = SimpleNamespace(role="ref", config=config, set_dispatch_collect=MagicMock())
    inspect.unwrap(engine_workers.ActorRolloutRefWorker.init_model)(worker)
    built = worker_factory.call_args.kwargs["config"]
    assert built.model_config.lora_adapter_path == ("/reference/sft/fog" if separate_reference else "/actor/step140/fog")
    assert config.model.lora_adapter_path == "/actor/step140/fog"
    assert built.model_config.mtp.enable is False
