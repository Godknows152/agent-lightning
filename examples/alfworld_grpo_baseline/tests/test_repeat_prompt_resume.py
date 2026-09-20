"""Prompt-only repeat guidance and safe latest-checkpoint selection."""
import json
from pathlib import Path
import sys

import pytest
from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alfworld_baseline.prompts_qwen35 import build_user_prompt, QWEN35_ALFWORLD_CHAT_TEMPLATE
from alfworld_baseline.resume import resolve_resume_checkpoint


@pytest.mark.parametrize("thinking", [True, False])
def test_repeat_guidance_preserves_history_and_legal_actions(thinking):
    prompt = build_user_prompt(mission="find a knife", observation="Cabinet 1 is closed.",
                               admissible_actions=["go to cabinet 1", "open cabinet 1"],
                               history=["go to cabinet 1 [executed]"],
                               enable_thinking=thinking, avoid_repeated_actions=True)
    assert "Do not reuse an exact action command" in prompt
    assert "1. go to cabinet 1 [executed]" in prompt
    assert "- go to cabinet 1\n- open cabinet 1" in prompt  # no action masking
    rendered = Environment().from_string(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(
        messages=[{"role": "user", "content": prompt}], enable_thinking=thinking,
        avoid_repeated_actions=True, add_generation_prompt=True)
    assert "Do not select an exact action command you have already used" in rendered
    assert rendered.endswith("<think>\n" if thinking else "<think>\n\n</think>\n\n")


def test_default_prompt_unchanged_for_other_profiles():
    prompt = build_user_prompt(mission="test", observation="test", admissible_actions=["look"])
    assert "Do not reuse an exact action command" not in prompt
    rendered = Environment().from_string(QWEN35_ALFWORLD_CHAT_TEMPLATE).render(
        messages=[], enable_thinking=True, add_generation_prompt=True)
    assert "Do not select an exact action command you have already used" not in rendered


def checkpoint_fixture(root, step):
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_text("fixture")
    (checkpoint / "actor/fsdp_config.json").write_text("{}")
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            (checkpoint / "actor" / f"{kind}_world_size_2_rank_{rank}.pt").write_text("fixture")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    (root / ".swanlab_experiment.json").write_text(json.dumps({"experiment_name": "original", "run_id": "run123"}))
    return checkpoint


def test_latest_checkpoint_changes_without_editing_config(tmp_path):
    assert resolve_resume_checkpoint(str(tmp_path), "original") is None
    p = checkpoint_fixture(tmp_path, 20)
    assert resolve_resume_checkpoint(str(tmp_path), "original") == str(p)
    p = checkpoint_fixture(tmp_path, 30)
    assert resolve_resume_checkpoint(str(tmp_path), "original") == str(p)


def test_incomplete_checkpoint_rejected(tmp_path):
    p = checkpoint_fixture(tmp_path, 20)
    (p / "actor/optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        resolve_resume_checkpoint(str(tmp_path), "original")


def test_wrong_identity_rejected(tmp_path):
    checkpoint_fixture(tmp_path, 20)
    with pytest.raises(ValueError, match="experiment name differs"):
        resolve_resume_checkpoint(str(tmp_path), "other")
    (tmp_path / ".swanlab_experiment.json").write_text('{"experiment_name":"original"}')
    with pytest.raises(ValueError, match="Missing SwanLab run ID"):
        resolve_resume_checkpoint(str(tmp_path), "original")


def test_existing_run_without_checkpoint_rejected(tmp_path):
    (tmp_path / ".swanlab_experiment.json").write_text('{}')
    with pytest.raises(ValueError, match="no published checkpoint"):
        resolve_resume_checkpoint(str(tmp_path), "original")


def test_native_resume_validates_all_checkpoint_shards(tmp_path):
    from omegaconf import OmegaConf
    from alfworld_baseline.resume import validate_native_resume
    cfg = OmegaConf.create({'trainer': {'default_local_dir': str(tmp_path), 'resume_mode': 'auto',
                           'logger': ['swanlab'], 'experiment_name': 'original',
                           'n_gpus_per_node': 2, 'nnodes': 1}})
    validate_native_resume(cfg)  # genuinely new output
    checkpoint = checkpoint_fixture(tmp_path, 20)
    validate_native_resume(cfg)
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(checkpoint)
    validate_native_resume(cfg)
    (checkpoint / 'actor/optim_world_size_2_rank_1.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        validate_native_resume(cfg)
