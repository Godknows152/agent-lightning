"""Reference residency must not turn reference inference into training."""
from dataclasses import dataclass

import pytest

from alfworld_baseline.reference_engine import ResidentReferenceEngine
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("fail", [False, True])
def test_forward_only_restored_after_wrapping(monkeypatch, fail):
    @dataclass
    class Config:
        forward_only: bool = True
        param_offload: bool = True

    engine = object.__new__(ResidentReferenceEngine)
    config = Config()
    engine.engine_config = config
    sentinel = object()

    def wrap(self, module):
        assert module is sentinel
        assert not self.engine_config.forward_only
        assert config.forward_only  # Never mutate the shared reference config.
        if fail:
            raise RuntimeError("wrapping failed")
        return sentinel

    monkeypatch.setattr(FSDPEngineWithLMHead, "_build_fsdp_module", wrap)
    if fail:
        with pytest.raises(RuntimeError, match="wrapping failed"):
            engine._build_fsdp_module(sentinel)
    else:
        assert engine._build_fsdp_module(sentinel) is sentinel
    assert engine.engine_config is config
    assert engine.engine_config.forward_only
