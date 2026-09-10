"""Diagnostic processor regressions; no production PPO enablement."""
from types import SimpleNamespace

import pytest
import torch

from alfworld_baseline.thinking_budget import (
    ALFWorldThinkingBudgetLogitProcessor, Qwen35TokenIdsOnlyBudget,
)

START, END = 248068, 248069


def params(output, budget=2, prompt=None):
    # Real v5 system instructions include closed tags before the open prefix.
    return {'thinking_budget':budget,'__req__':SimpleNamespace(
        origin_input_ids=[START, 10, END, 20, START, 198] if prompt is None else prompt,
        output_ids=output)}


def test_stock_scanning_fails_despite_corrected_qwen35_ids():
    logits = torch.zeros(1, END + 1)
    Qwen35TokenIdsOnlyBudget()(logits, [params([11, 12])])
    assert torch.isfinite(logits).all()  # incorrectly sees system's </think>
    ALFWorldThinkingBudgetLogitProcessor()(logits, [params([11, 12])])
    assert logits[0,END] == 0
    assert torch.isfinite(logits).sum() == 1
    assert torch.log_softmax(logits, dim=-1)[0,END] == 0  # forced sampler probability one


@pytest.mark.parametrize('budget,output,force', [(0,[],True),(2,[],False),(2,[11],False),
    (2,[11,12],True),(2,[11,12,END],False),(2,[END],False),(None,[11,12],False),
    (-1,[11,12],False),(True,[11,12],False),(2.0,[11,12],False)])
def test_budget_boundary_and_early_natural_closure(budget,output,force):
    logits=torch.zeros(1,END+1)
    ALFWorldThinkingBudgetLogitProcessor()(logits,[params(output,budget)])
    assert (torch.isfinite(logits).sum().item()==1) == force


def test_mixed_batch_independent_budgets_and_no_post_closure_forcing():
    logits=torch.zeros(4,END+1)
    batch=[params([11],1), params([11],4), params([11,END],1), params([],0,prompt=[START,END])]
    ALFWorldThinkingBudgetLogitProcessor()(logits,batch)
    assert torch.isfinite(logits).sum(dim=-1).tolist()==[1,END+1,END+1,END+1]


def test_native_serialization_roundtrip():
    processor=ALFWorldThinkingBudgetLogitProcessor.from_str(ALFWorldThinkingBudgetLogitProcessor.to_str())
    logits=torch.zeros(1,END+1)
    processor(logits,[params([],0)])
    assert logits.argmax().item()==END


def test_upstream_style_newline_prefix_adds_at_most_one_token():
    from alfworld_baseline.thinking_budget import ALFWorldNewlineThinkingBudgetLogitProcessor
    processor = ALFWorldNewlineThinkingBudgetLogitProcessor()
    logits = torch.zeros(1, END+1)
    processor(logits, [params([11,12],2)])
    assert logits.argmax().item() == 198
    logits.zero_()
    processor(logits, [params([11,12,198],2)])
    assert logits.argmax().item() == END
    logits.zero_()
    processor(logits, [params([11,12,198,END],2)])
    assert torch.isfinite(logits).all()
